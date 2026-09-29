import { describe, expect, it } from 'vitest';
import {
  evictOldStats,
  loadAllStats,
  statsCommand,
  STATS_COMMAND,
  updateSessionStats,
  type StatsStore,
} from '../hooks/fast-jev.ts';
import { recordCompaction, recordTurn, STORE_PREFIX } from '../hooks/stats.ts';

function memoryStore(initial: Record<string, unknown> = {}): StatsStore & { data: Map<string, unknown> } {
  const data = new Map(Object.entries(initial));
  return {
    data,
    get: async (key) => (data.has(key) ? JSON.parse(JSON.stringify(data.get(key))) : undefined),
    set: async (key, value) => void data.set(key, JSON.parse(JSON.stringify(value))),
    delete: async (key) => void data.delete(key),
    keys: async () => [...data.keys()],
  };
}

const usage = { input_tokens: 0, output_tokens: 10, cache_read_input_tokens: 100_000, cache_creation_input_tokens: 0 };

describe('stats store', () => {
  it('writes one key per session and ignores foreign keys', async () => {
    const store = memoryStore({ other: 1 });
    await updateSessionStats({ store }, 's1', 1, (s) => void recordTurn(s, { at: 1, usage, isAgent: false }));
    await updateSessionStats({ store }, 's1', 2, (s) => void recordTurn(s, { at: 2, usage, isAgent: false }));
    await updateSessionStats({ store }, 's2', 3, (s) => void recordTurn(s, { at: 3, usage, isAgent: true }));
    const all = await loadAllStats({ store }, 4);
    expect(all.map((s) => [s.id, s.main.turns, s.agents.turns])).toEqual([
      ['s1', 2, 0],
      ['s2', 0, 1],
    ]);
    expect(store.data.get('other')).toBe(1);
  });

  it('evicts nothing under the cap', async () => {
    const store = memoryStore();
    await updateSessionStats({ store }, 's1', 1, () => {});
    expect(await evictOldStats({ store }, 2)).toBe(0);
    expect(store.data.has(`${STORE_PREFIX}s1`)).toBe(true);
  });
});

describe(`/${STATS_COMMAND}`, () => {
  async function seeded() {
    const store = memoryStore();
    await updateSessionStats({ store }, 'mine', 1_790_000_000_000, (s) => {
      recordCompaction(s, { at: 1_790_000_000_000, outcome: 'jev', tokensBefore: 160_000, tokensDropped: 60_000 });
      recordTurn(s, { at: 1_790_000_001_000, usage, isAgent: false, contextTokens: 100_000 });
    });
    await updateSessionStats({ store }, 'theirs', 1_790_000_000_000, (s) => {
      recordCompaction(s, { at: 1_790_000_000_000, outcome: 'fallback-error', reason: 'timeout' });
    });
    return store;
  }

  it('reports all sessions by default and this session on request', async () => {
    const store = await seeded();
    const all = await statsCommand({ store }, '', 'mine', 0);
    expect(all).toContain('all sessions, 2 session(s)');
    expect(all).toContain('1 jev · 0 fallback (low reduction) · 1 fallback (error)');
    const own = await statsCommand({ store }, ' Session ', 'mine', 0);
    expect(own).toContain('this session, 1 session(s)');
    expect(own).toContain('~60K context tokens not re-sent');
  });

  it('resets only stats keys and rejects unknown arguments', async () => {
    const store = await seeded();
    await store.set('keep', true);
    expect(await statsCommand({ store }, 'bogus', 'mine', 0)).toMatch(/Unknown argument "bogus"/);
    expect(await statsCommand({ store }, 'reset', 'mine', 0)).toContain('2 session(s) cleared');
    expect([...store.data.keys()]).toEqual(['keep']);
  });

  it('does not let a subagent compaction move the main-loop credit', async () => {
    const store = memoryStore();
    await updateSessionStats({ store }, 's', 1, (s) => {
      recordCompaction(s, { at: 1, outcome: 'jev', tokensDropped: 50_000 });
      recordCompaction(s, { at: 2, outcome: 'fallback-error', agent: true });
      recordTurn(s, { at: 3, usage, isAgent: false, contextTokens: 100_000 });
    });
    const [s] = await loadAllStats({ store }, 4);
    expect(s?.savedTokens).toBe(50_000);
    expect(s?.compactions['fallback-error']).toBe(1);
  });
});
