import { describe, expect, it } from 'vitest';
import {
  aggregate,
  droppedTokens,
  emptySession,
  estimateCalls,
  evictable,
  formatReport,
  formatTokens,
  recordCompaction,
  recordTurn,
  reviveSession,
  RECENT_PER_SESSION,
  REASON_MAX_CHARS,
  STORE_PREFIX,
  weightedInput,
  weightedSaved,
  type TokenUsage,
} from '../hooks/stats.ts';

function usage(input: number, cacheRead: number, cacheWrite = 0, output = 100): TokenUsage {
  return {
    input_tokens: input,
    cache_read_input_tokens: cacheRead,
    cache_creation_input_tokens: cacheWrite,
    output_tokens: output,
  };
}

describe('stats', () => {
  it('estimates API calls per turn from the summed input side', () => {
    expect(estimateCalls(usage(0, 300_000, 0), 100_000)).toBe(3);
    expect(estimateCalls(usage(0, 50_000), 100_000)).toBe(1);
    expect(estimateCalls(usage(0, 300_000), undefined)).toBe(1);
  });

  it('splits main and subagent usage and keeps the highest cost reading', () => {
    const s = emptySession('a', 0);
    recordTurn(s, { at: 1, usage: usage(10, 1000, 50), isAgent: false, costUsd: 0.5 });
    recordTurn(s, { at: 2, usage: usage(5, 500), isAgent: true, costUsd: 0.4 });
    expect(s.main).toEqual({ turns: 1, input: 10, cacheRead: 1000, cacheWrite: 50, output: 100 });
    expect(s.agents.turns).toBe(1);
    expect(s.costUsd).toBe(0.5);
    expect(s.lastAt).toBe(2);
  });

  it('credits the last judge drop per main-loop call until a fallback resets it', () => {
    const s = emptySession('a', 0);
    recordCompaction(s, { at: 1, outcome: 'judge', tokensBefore: 160_000, tokensDropped: 60_000, ms: 500 });
    recordTurn(s, { at: 2, usage: usage(0, 200_000), isAgent: false, contextTokens: 100_000 });
    recordTurn(s, { at: 3, usage: usage(0, 200_000), isAgent: true, contextTokens: 100_000 });
    expect(s.savedTokens).toBe(120_000);
    recordCompaction(s, { at: 4, outcome: 'fallback-low', reason: 'below 25%' });
    recordTurn(s, { at: 5, usage: usage(0, 100_000), isAgent: false, contextTokens: 100_000 });
    expect(s.savedTokens).toBe(120_000);
    expect(s.compactions).toEqual({ judge: 1, 'fallback-low': 1, 'fallback-error': 0 });
    expect(s.droppedTokens).toBe(60_000);
  });

  it('caps the recent compactions kept per session', () => {
    const s = emptySession('a', 0);
    for (let i = 0; i < RECENT_PER_SESSION + 5; i++) recordCompaction(s, { at: i, outcome: 'fallback-error' });
    expect(s.recent).toHaveLength(RECENT_PER_SESSION);
    expect(s.recent[0]?.at).toBe(5);
  });

  it('derives dropped tokens from the character reduction', () => {
    expect(droppedTokens(100_000, 0.4)).toBe(40_000);
    expect(droppedTokens(undefined, 0.4)).toBeUndefined();
  });

  it('revives partial stored values and ignores garbage', () => {
    const s = reviveSession('x', { savedTokens: 7, main: { turns: 2 } }, 9);
    expect(s.savedTokens).toBe(7);
    expect(s.main.turns).toBe(2);
    expect(s.main.input).toBe(0);
    expect(reviveSession('x', 'nope', 9)).toEqual(emptySession('x', 9));
  });

  it('evicts the least recently active sessions beyond the cap', () => {
    const sessions = [3, 1, 2].map((t) => ({ ...emptySession(`s${t}`, 0), lastAt: t }));
    expect(evictable(sessions, 2)).toEqual([`${STORE_PREFIX}s1`]);
    expect(evictable(sessions, 3)).toEqual([]);
  });

  it('aggregates sessions and formats a report', () => {
    const a = emptySession('a', 1_790_000_000_000);
    recordCompaction(a, { at: 1_790_000_100_000, outcome: 'judge', tokensBefore: 167_000, tokensDropped: 67_000, ratio: 0.4, ms: 600 });
    recordTurn(a, { at: 1_790_000_200_000, usage: usage(1000, 99_000), isAgent: false, contextTokens: 100_000, costUsd: 1.25 });
    const b = emptySession('b', 1_790_001_000_000);
    recordCompaction(b, { at: 1_790_001_100_000, outcome: 'fallback-error', reason: 'timeout' });
    const total = aggregate([a, b]);
    expect(total.startedAt).toBe(1_790_000_000_000);
    expect(total.savedTokens).toBe(67_000);
    expect(total.recent.map((r) => r.outcome)).toEqual(['judge', 'fallback-error']);
    const report = formatReport(total, 'all sessions', 2);
    expect(report).toContain('1 by judge · judge avg 600 ms · 1 fallback (0 low reduction, 1 error)');
    expect(report).toContain('Saved        ~7K tokens at input price');
    expect(report).toContain('1 turns · avg 100K context per turn · cache hit 99.0%');
    expect(report).toContain('$1.25');
    expect(report).toContain('167K → 100K  40%  600 ms');
    expect(report).not.toContain('Timing');
  });

  it('reports the judge timing split when requests carried it', () => {
    const s = emptySession('a', 0);
    recordCompaction(s, {
      at: 1, outcome: 'judge', ms: 12_340, requests: 3,
      judge: { loadMs: 7_000, promptMs: 900, promptTokens: 6_000, genMs: 3_500, genTokens: 500 },
    });
    recordCompaction(s, {
      at: 2, outcome: 'judge', ms: 4_500, requests: 1,
      judge: { loadMs: 0, promptMs: 600, promptTokens: 4_000, genMs: 3_500, genTokens: 500 },
    });
    const total = aggregate([s]);
    expect(total.judgeRequests).toBe(4);
    expect(total.judgePromptTokens).toBe(10_000);
    const report = formatReport(total, 'all sessions', 1);
    expect(report).toContain(
      'Timing       avg per judge compaction: load 3500 ms + prefill 750 ms (5K tok) + gen 3500 ms (500 tok) + 2.0 req/compaction',
    );
    expect(report).toContain('3 req');
  });

  it('formats an empty report without failing', () => {
    expect(formatReport(aggregate([]), 'all sessions', 0)).toContain('0 session(s) since n/a');
  });

  it('formats token counts', () => {
    expect(formatTokens(950)).toBe('950');
    expect(formatTokens(167_400)).toBe('167K');
    expect(formatTokens(4_830_000)).toBe('4.8M');
  });

  it('weights input by cache price', () => {
    const u = { turns: 1, input: 1_000, cacheRead: 10_000, cacheWrite: 800, output: 0 };
    expect(weightedInput(u)).toBe(1_000 + 1_000 + 1_000);
  });

  it('prices saved tokens at the main loop cache-read share', () => {
    const main = { turns: 1, input: 0, cacheRead: 900, cacheWrite: 100, output: 0 };
    expect(weightedSaved(1_000, main)).toBeCloseTo(1_000 * (0.9 * 0.1 + 0.1));
    expect(weightedSaved(1_000, { ...main, cacheRead: 0, cacheWrite: 0 })).toBe(1_000);
  });

  it('stores a fallback reason on one line and capped', () => {
    const s = recordCompaction(emptySession('s', 0), {
      at: 1,
      outcome: 'fallback-error',
      reason: `boom\n\n  ${'x'.repeat(5_000)}`,
    });
    const reason = s.recent[0]?.reason ?? '';
    expect(reason.startsWith('boom x')).toBe(true);
    expect(reason.length).toBe(REASON_MAX_CHARS);
  });

  it('rejects prototype-key outcomes when reviving', () => {
    const s = reviveSession('s', { recent: [{ at: 1, outcome: 'toString' }] }, 0);
    expect(s.recent).toEqual([]);
  });
});
