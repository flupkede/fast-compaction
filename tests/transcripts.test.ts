import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  isRetryableLedgerError,
  isTooLargeError,
  listClaudeTranscriptFiles,
  loadClaudeTranscript,
  loadReplayIndex,
  opencodeToMessages,
  openOpencodeTranscripts,
  parseClaudeTranscript,
  planReplay,
  replaySession,
  runReplay,
  saveReplayIndex,
  type JevAsker,
  type JevQuestions,
  type Message,
  type ReplaySourceSpec,
  type TrainingRecord,
} from '../src/index.js';

const tmpDirs: string[] = [];

afterEach(async () => {
  while (tmpDirs.length > 0) await rm(tmpDirs.pop()!, { recursive: true, force: true });
});

async function tempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'fast-compaction-transcripts-'));
  tmpDirs.push(dir);
  return dir;
}

describe('parseClaudeTranscript', () => {
  const jsonl = [
    JSON.stringify({ type: 'system', subtype: 'init' }),
    JSON.stringify({
      type: 'user',
      message: { role: 'user', content: 'Fix the test' },
    }),
    '{not json',
    JSON.stringify({
      type: 'assistant',
      message: {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'hmm' },
          { type: 'text', text: 'reading the file' },
          { type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: 'a.ts' } },
        ],
      },
    }),
    JSON.stringify({
      type: 'user',
      message: {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'toolu_1', content: 'export const a = 1;' },
          {
            type: 'tool_result',
            tool_use_id: 'toolu_2',
            content: [{ type: 'text', text: 'boom' }],
            is_error: true,
          },
        ],
      },
    }),
    JSON.stringify({ type: 'ai-title', title: 'x' }),
  ].join('\n');

  it('keeps user and assistant lines with their tool blocks and skips the rest', () => {
    const messages = parseClaudeTranscript(jsonl);
    expect(messages).toHaveLength(3);
    expect(messages[0]).toEqual({ role: 'user', text: 'Fix the test', toolUses: [] });
    expect(messages[1]!.text).toBe('reading the file');
    expect(messages[1]!.toolUses).toEqual([
      { tool_use_id: 'toolu_1', tool: 'Read', input: { file_path: 'a.ts' } },
    ]);
    expect(messages[2]!.toolResults).toEqual([
      { tool_use_id: 'toolu_1', text: 'export const a = 1;' },
      { tool_use_id: 'toolu_2', text: 'boom', isError: true },
    ]);
  });

  it('loads from a file and lists transcript files recursively', async () => {
    const root = await tempDir();
    await writeFile(join(root, 'one.jsonl'), jsonl, 'utf8');
    const nested = join(root, 'nested');
    await (await import('node:fs/promises')).mkdir(nested, { recursive: true });
    await writeFile(join(nested, 'two.jsonl'), jsonl, 'utf8');
    expect((await listClaudeTranscriptFiles(root)).map((p) => p.split(/[\\/]/).pop())).toEqual([
      'two.jsonl',
      'one.jsonl',
    ]);
    const messages = await loadClaudeTranscript(join(root, 'one.jsonl'));
    expect(messages).toHaveLength(3);
  });
});

describe('opencodeToMessages', () => {
  it('maps text and tool parts, pairing results via a synthetic user message', () => {
    const messages = opencodeToMessages('assistant', [
      { type: 'step-start' },
      { type: 'text', text: 'checking' },
      {
        type: 'tool',
        callID: 'call_1',
        tool: 'read',
        state: { status: 'completed', input: { path: 'a.ts' }, output: { content: 'x' } },
      },
    ]);
    expect(messages).toEqual([
      { role: 'assistant', text: 'checking', toolUses: [{ tool_use_id: 'call_1', tool: 'read', input: { path: 'a.ts' } }] },
      {
        role: 'user',
        text: '',
        toolUses: [],
        toolResults: [{ tool_use_id: 'call_1', text: '{"content":"x"}' }],
      },
    ]);
  });

  it('marks error status, truncates huge outputs and skips tools without output', () => {
    const messages = opencodeToMessages('user', [
      { type: 'tool', callID: 'ok', tool: 'bash', state: { status: 'error', input: {}, output: 'nope' } },
      { type: 'tool', callID: 'pending', tool: 'bash', state: { status: 'running', input: {} } },
      { type: 'tool', callID: 'big', tool: 'read', state: { status: 'completed', input: {}, output: 'y'.repeat(30_000) } },
    ]);
    const results = messages[1]!.toolResults!;
    expect(results[0]).toMatchObject({ tool_use_id: 'ok', text: 'nope', isError: true });
    expect(results[1]!.text.length).toBe(20_000);
    expect(results.map((r) => r.tool_use_id)).toEqual(['ok', 'big']);
  });
});

describe('openOpencodeTranscripts', () => {
  async function fixtureDb(): Promise<string> {
    const { createRequire } = await import('node:module');
    const { DatabaseSync } = createRequire(import.meta.url)('node:sqlite') as typeof import('node:sqlite');
    const dir = await tempDir();
    const path = join(dir, 'opencode.db');
    const db = new DatabaseSync(path);
    db.exec(
      `CREATE TABLE session (id text PRIMARY KEY, project_id text, parent_id text, slug text,
         directory text, title text, version text, share_url text, time_created integer);
       CREATE TABLE message (id text PRIMARY KEY, session_id text, time_created integer,
         time_updated integer, data text);
       CREATE TABLE part (id text PRIMARY KEY, message_id text, session_id text,
         time_created integer, time_updated integer, data text);`,
    );
    db.prepare(
      'INSERT INTO session (id, project_id, parent_id, slug, directory, title, version, share_url, time_created) VALUES (?,?,?,?,?,?,?,?,?)',
    ).run('ses_a', 'p', null, 's', 'C:/x', 'Session A', '1', null, 2);
    db.prepare(
      'INSERT INTO session (id, project_id, parent_id, slug, directory, title, version, share_url, time_created) VALUES (?,?,?,?,?,?,?,?,?)',
    ).run('ses_b', 'p', null, 's', 'C:/x', 'Session B', '1', null, 1);
    const insertMessage = db.prepare(
      'INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?)',
    );
    const insertPart = db.prepare(
      'INSERT INTO part (id, message_id, session_id, time_created, time_updated, data) VALUES (?,?,?,?,?,?)',
    );
    insertMessage.run('m1', 'ses_a', 1, 1, JSON.stringify({ role: 'user' }));
    insertPart.run('p1', 'm1', 'ses_a', 1, 1, JSON.stringify({ type: 'text', text: 'do it' }));
    insertMessage.run('m2', 'ses_a', 2, 2, JSON.stringify({ role: 'assistant' }));
    insertPart.run(
      'p2',
      'm2',
      'ses_a',
      2,
      2,
      JSON.stringify({ type: 'tool', callID: 'c1', tool: 'read', state: { status: 'completed', input: { path: 'a' }, output: 'ok' } }),
    );
    insertMessage.run('m3', 'ses_b', 1, 1, JSON.stringify({ role: 'user' }));
    db.close();
    return path;
  }

  it('lists sessions by recency with a message-count filter and normalises messages', async () => {
    const transcripts = await openOpencodeTranscripts(await fixtureDb());
    try {
      expect(transcripts.sessions(2).map((s) => s.id)).toEqual(['ses_a']);
      expect(transcripts.sessions(1).map((s) => s.id)).toEqual(['ses_a', 'ses_b']);
      expect(transcripts.messages('ses_a')).toEqual([
        { role: 'user', text: 'do it', toolUses: [] },
        {
          role: 'assistant',
          text: '',
          toolUses: [{ tool_use_id: 'c1', tool: 'read', input: { path: 'a' } }],
        },
        { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'c1', text: 'ok' }] },
      ]);
    } finally {
      transcripts.close();
    }
  });
});

describe('planReplay and replaySession', () => {
  const transcript: Message[] = [
    { role: 'user', text: 'Fix the test', toolUses: [] },
    {
      role: 'assistant',
      text: '',
      toolUses: [{ tool_use_id: 't1', tool: 'Read', input: { file_path: 'a.ts' }, text: 'a' }],
    },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 't1', text: 'a'.repeat(2000) }] },
    { role: 'assistant', text: 'reading', toolUses: [] },
    { role: 'user', text: 'and then', toolUses: [] },
    { role: 'assistant', text: 'fixing', toolUses: [] },
    { role: 'user', text: 'ok', toolUses: [] },
    { role: 'assistant', text: 'done', toolUses: [] },
    { role: 'user', text: 'thanks', toolUses: [] },
    { role: 'assistant', text: 'bye', toolUses: [] },
  ];

  it('counts candidates and requests without asking', () => {
    const plan = planReplay(transcript);
    expect(plan).toMatchObject({ messages: 10, calls: 1, requests: 2 });
    expect(plan.tokens).toBeGreaterThan(0);
    expect(planReplay(transcript.slice(0, 1)).calls).toBe(0);
  });

  it('plans a pure-chat transcript through the prose path, all-protected as nothing', () => {
    const chat: Message[] = [
      { role: 'user', text: 'seed context', toolUses: [] },
      { role: 'assistant', text: 'old rambling', toolUses: [] },
      { role: 'user', text: 'latest question', toolUses: [] },
    ];
    // 3 messages, preserve 1 → protected: i=0 and i=2; only msg_1 judgeable
    expect(planReplay(chat, { preserveRecentMessages: 1 })).toMatchObject({
      calls: 0,
      prose: 1,
      requests: 1,
    });
    // everything inside the recent window: no request at all
    expect(planReplay(chat.slice(0, 2))).toMatchObject({
      calls: 0,
      prose: 0,
      requests: 0,
    });
  });

  it('reports an unfittable history as an error instead of throwing', () => {
    const many = (count: number, prefix: string) =>
      Array.from({ length: count }, (_, i) => ({
        tool_use_id: `${prefix}_${i}`,
        tool: 'Read',
        input: { file_path: `src/file_${prefix}_${i}.ts` },
        text: 'x',
      }));
    const messages: Message[] = [
      { role: 'user', text: 'start', toolUses: [] },
      {
        role: 'assistant',
        text: '',
        toolUses: [{ tool_use_id: 't1', tool: 'Read', input: { file_path: 'a.ts' } }],
      },
      {
        role: 'user',
        text: '',
        toolUses: [],
        toolResults: [{ tool_use_id: 't1', text: 'a' }],
      },
      { role: 'user', text: 'filler', toolUses: [] },
    ];
    // the newest six messages are pinned and their calls escape every shrink stage
    for (let round = 0; round < 3; round++) {
      messages.push({ role: 'assistant', text: '', toolUses: many(500, `p${round}`) });
      messages.push({
        role: 'user',
        text: '',
        toolUses: [],
        toolResults: many(500, `p${round}`).map((use) => ({
          tool_use_id: use.tool_use_id,
          text: 'ok',
        })),
      });
    }
    const plan = planReplay(messages);
    expect(plan.calls).toBe(1);
    expect(plan.error).toMatch(/too large/);
  });

  it('replays through compact with a replay-tagged record', async () => {
    const records: TrainingRecord[] = [];
    const asker: JevAsker = {
      async ask(_state, questions: JevQuestions) {
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.8 }]),
          ),
        };
      },
    };
    const result = await replaySession(transcript, asker, (record) => void records.push(record));
    expect(result.stats.requests).toBe(2); // call batch + message batch
    expect(records).toHaveLength(1);
    expect(records[0]!.source).toBe('replay');
  });
});

describe('replay index', () => {
  it('round-trips, defaults to empty for a missing file and rejects a corrupt one', async () => {
    const dir = await tempDir();
    expect(await loadReplayIndex(dir)).toEqual({});
    await saveReplayIndex(dir, { 'claude:x': { at: 1, records: 2 } });
    expect(await loadReplayIndex(dir)).toEqual({ 'claude:x': { at: 1, records: 2 } });
    await writeFile(join(dir, 'replay-index.json'), '{broken', 'utf8');
    await expect(loadReplayIndex(dir)).rejects.toThrow();
  });
});

describe('runReplay', () => {
  const good: Message[] = [
    { role: 'user', text: 'Fix the test', toolUses: [] },
    {
      role: 'assistant',
      text: '',
      toolUses: [{ tool_use_id: 't1', tool: 'Read', input: { file_path: 'a.ts' } }],
    },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 't1', text: 'a'.repeat(2000) }] },
    { role: 'assistant', text: 'reading', toolUses: [] },
    { role: 'user', text: 'and then', toolUses: [] },
    { role: 'assistant', text: 'fixing', toolUses: [] },
    { role: 'user', text: 'ok', toolUses: [] },
    { role: 'assistant', text: 'done', toolUses: [] },
    { role: 'user', text: 'thanks', toolUses: [] },
    { role: 'assistant', text: 'bye', toolUses: [] },
  ];

  function asker(fail = false, message = 'api down'): JevAsker {
    return {
      async ask(_state, questions: JevQuestions) {
        if (fail) throw new Error(message);
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
        };
      },
    };
  }

  function deps(dir: string) {
    return {
      dir,
      delayMs: 0,
      sleep: async () => {},
      now: () => 1234,
    };
  }

  it('replays new sources, skips ledgered ones, records no-candidates and load errors', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, { 's:done': { at: 1, records: 1 } });
    const records: TrainingRecord[] = [];
    const sources: ReplaySourceSpec[] = [
      { key: 's:done', label: 'done', load: async () => good },
      { key: 's:good', label: 'good', load: async () => good },
      { key: 's:tiny', label: 'tiny', load: async () => [{ role: 'user', text: 'hi', toolUses: [] }] },
      { key: 's:broken', label: 'broken', load: async () => { throw new Error('gone'); } },
    ];
    const summary = await runReplay(sources, asker(), (r) => void records.push(r), deps(dir));
    expect(summary).toEqual({
      sessions: 3,
      replayed: 1,
      records: 1,
      skippedDone: 1,
      noCandidates: 1,
      errors: 1,
      stoppedAfterFailures: false,
    });
    expect(records).toHaveLength(1);
    expect(records[0]!.source).toBe('replay');
    const index = await loadReplayIndex(dir);
    expect(index['s:good']).toEqual({ at: 1234, records: 1 });
    expect(index['s:tiny']).toEqual({ at: 1234, records: 0 });
    expect(index['s:done']).toEqual({ at: 1, records: 1 });
    expect(index['s:broken']).toBeUndefined();
  });

  it('leaves a failed Jev call unledgered so a re-run retries it, and honors the limit', async () => {
    const dir = await tempDir();
    const sources: ReplaySourceSpec[] = [
      { key: 'a', label: 'a', load: async () => good },
      { key: 'b', label: 'b', load: async () => good },
    ];
    const failing = await runReplay(sources, asker(true), () => {}, deps(dir));
    expect(failing).toEqual({
      sessions: 2,
      replayed: 0,
      records: 0,
      skippedDone: 0,
      noCandidates: 0,
      errors: 2,
      stoppedAfterFailures: false,
    });
    expect(await loadReplayIndex(dir)).toEqual({});

    const records: TrainingRecord[] = [];
    const limited = await runReplay(sources, asker(), (r) => void records.push(r), { ...deps(dir), limit: 1 });
    expect(limited.replayed).toBe(1);
    expect(records).toHaveLength(1);
  });

  it('stops after five consecutive Jev failures instead of hammering the API', async () => {
    const dir = await tempDir();
    const sources: ReplaySourceSpec[] = Array.from({ length: 7 }, (_, i) => ({
      key: `f${i}`,
      label: `f${i}`,
      load: async () => good,
    }));
    const events: string[] = [];
    const summary = await runReplay(
      sources,
      asker(true),
      () => {},
      { ...deps(dir), delayMs: 0 },
      (event) => events.push(event.outcome),
    );
    expect(summary.stoppedAfterFailures).toBe(true);
    expect(summary.sessions).toBe(5);
    expect(events.filter((outcome) => outcome === 'error')).toHaveLength(5);
    expect(events[events.length - 1]).toBe('stopped');
  });

  it('treats a failed record write as retriable: no ledger entry, counted as error', async () => {
    const dir = await tempDir();
    const sources: ReplaySourceSpec[] = [{ key: 'a', label: 'a', load: async () => good }];
    const summary = await runReplay(
      sources,
      asker(),
      () => {
        throw new Error('disk full');
      },
      deps(dir),
    );
    expect(summary).toEqual({
      sessions: 1,
      replayed: 0,
      records: 0,
      skippedDone: 0,
      noCandidates: 0,
      errors: 1,
      stoppedAfterFailures: false,
    });
    expect(await loadReplayIndex(dir)).toEqual({});
  });

  it('ledgers a deterministic input-ceiling 400 like a plan error; transient failures stay retriable', async () => {
    const dir = await tempDir();
    const sources: ReplaySourceSpec[] = [
      { key: 'sized', label: 'sized', load: async () => good },
      { key: 'huge', label: 'huge', load: async () => good },
      { key: 'flaky', label: 'flaky', load: async () => good },
      { key: 'crashy', label: 'crashy', load: async () => good },
      { key: 'ok', label: 'ok', load: async () => good },
    ];
    const deterministic = [
      'Jev request failed (400): {"error":"prompt 0 has 2355 tokens; expected 1–2050 (input is never truncated)"}',
      'Jev request failed (413): {"error":"text and schema must not exceed 64 KiB"}',
      'Jev request timed out (no answer within 240s)',
      'Jev request failed (500): {"error":"llama-server process has terminated: exit status 0xc0000005"}',
    ];
    let attempt = 0;
    const summary = await runReplay(
      sources,
      {
        async ask(_state, questions: JevQuestions) {
          if (attempt < deterministic.length) {
            throw new Error(deterministic[attempt++]);
          }
          return {
            answers: Object.fromEntries(
              Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
            ),
          };
        },
      },
      () => {},
      deps(dir),
    );
    expect(summary.errors).toBe(4);
    expect(summary.replayed).toBe(1);
    expect(summary.stoppedAfterFailures).toBe(false);
    const index = await loadReplayIndex(dir);
    expect(index.sized?.error).toMatch(/input is never truncated/);
    expect(index.huge?.error).toMatch(/64 KiB/);
    // timeouts are infra-bound: unledgered, retried next run
    expect(index.flaky).toBeUndefined();
    expect(index.crashy?.error).toMatch(/llama-server/);
    expect(index.ok?.records).toBe(1);
  });

  it('ledgers a live too-large request failure so the next run segments it', async () => {
    const dir = await tempDir();
    const sources: ReplaySourceSpec[] = [{ key: 'drift', label: 'drift', load: async () => good }];
    const tooLarge: JevAsker = {
      async ask() {
        throw new Error('history too large for Jev (~8123 tokens after truncation, limit 8000)');
      },
    };
    const summary = await runReplay(sources, tooLarge, () => {}, deps(dir));
    expect(summary.errors).toBe(1);
    expect((await loadReplayIndex(dir)).drift?.error).toMatch(/history too large/);
  });

  it('segments a session whose ledger says too-large even when the plan fits', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, {
      drift: { at: 1, error: 'history too large for Jev (~8123 tokens after truncation, limit 8000)' },
    });
    const sources: ReplaySourceSpec[] = [{ key: 'drift', label: 'drift', load: async () => good }];
    const details: string[] = [];
    const summary = await runReplay(
      sources,
      asker(),
      () => {},
      { ...deps(dir), segmentOversized: true },
      (event) => details.push(event.detail ?? ''),
    );
    expect(summary.replayed).toBe(1);
    expect(details[0]).toMatch(/segment/);
    expect((await loadReplayIndex(dir)).drift?.records).toBeGreaterThan(0);
  });

  it('planning sees compactOptions: an oversized session is a ledgered plan error, not a breaker trip', async () => {
    const dir = await tempDir();
    const sources: ReplaySourceSpec[] = [
      { key: 'big', label: 'big', load: async () => good },
      { key: 'next', label: 'next', load: async () => good },
    ];
    const summary = await runReplay(sources, asker(), () => {}, {
      ...deps(dir),
      compactOptions: { maxStateTokens: 10 },
    });
    expect(summary.errors).toBe(2);
    expect(summary.replayed).toBe(0);
    expect(summary.stoppedAfterFailures).toBe(false);
    expect((await loadReplayIndex(dir)).big?.error).toMatch(/too large/);
  });

  it('sleeps between replayed sessions when delayMs is set', async () => {
    const dir = await tempDir();
    const slept: number[] = [];
    const sources: ReplaySourceSpec[] = [
      { key: 'a', label: 'a', load: async () => good },
      { key: 'b', label: 'b', load: async () => good },
    ];
    const summary = await runReplay(sources, asker(), () => {}, {
      ...deps(dir),
      delayMs: 7,
      sleep: async (ms) => {
        slept.push(ms);
      },
    });
    expect(summary.replayed).toBe(2);
    expect(slept).toEqual([7, 7]);
  });
});

describe('parseReplayArgs', () => {
  it('parses flags and rejects unknown flags, bad sources and malformed numbers', async () => {
    const { parseReplayArgs } = await import('../src/index.js');
    expect(parseReplayArgs(['--dry-run', '--source', 'claude', '--limit', '3'])).toEqual({
      dryRun: true,
      source: 'claude',
      limit: 3,
      asker: 'ollama',
      segment: true,
    });
    expect(() => parseReplayArgs(['--nope'])).toThrow(/unknown argument/);
    expect(() => parseReplayArgs(['--source', 'cursor'])).toThrow(/--source/);
    expect(() => parseReplayArgs(['--limit'])).toThrow(/needs a value/);
    expect(() => parseReplayArgs(['--limit', '-1'])).toThrow(/>= 0/);
  });

  it('parses the local-judge flags and rejects a bad asker', async () => {
    const { parseReplayArgs } = await import('../src/index.js');
    expect(
      parseReplayArgs(['--asker', 'ollama', '--base-url', 'http://pc:11434/v1/systemone', '--model', 'clef-flash']),
    ).toMatchObject({
      asker: 'ollama',
      baseUrl: 'http://pc:11434/v1/systemone',
      model: 'clef-flash',
    });
    expect(parseReplayArgs(['--asker', 'cloud'])).toMatchObject({ asker: 'cloud' });
    expect(() => parseReplayArgs(['--asker', 'chatgpt'])).toThrow(/--asker/);
  });

  it('parses --no-segment as disabling the default segmentation', async () => {
    const { parseReplayArgs } = await import('../src/index.js');
    expect(parseReplayArgs(['--no-segment'])).toMatchObject({ segment: false });
    expect(parseReplayArgs(['--segment'])).toMatchObject({ segment: true });
  });

  it('parses --max-state-tokens and validates the number', async () => {
    const { parseReplayArgs } = await import('../src/index.js');
    expect(parseReplayArgs(['--max-state-tokens', '2000'])).toMatchObject({ maxStateTokens: 2000 });
    expect(() => parseReplayArgs(['--max-state-tokens'])).toThrow(/needs a value/);
    expect(() => parseReplayArgs(['--max-state-tokens', 'abc'])).toThrow(/>= 0/);
    expect(() => parseReplayArgs(['--max-state-tokens', '-5'])).toThrow(/>= 0/);
  });

  it('parses --request-timeout-ms and validates the number', async () => {
    const { parseReplayArgs } = await import('../src/index.js');
    expect(parseReplayArgs(['--request-timeout-ms', '900000'])).toMatchObject({ requestTimeoutMs: 900000 });
    expect(() => parseReplayArgs(['--request-timeout-ms'])).toThrow(/needs a value/);
    expect(() => parseReplayArgs(['--request-timeout-ms', 'abc'])).toThrow(/>= 0/);
  });

  it('resolves the request timeout as flag > OLLAMA_REQUEST_TIMEOUT_MS env > default', async () => {
    const { resolveRequestTimeoutMs } = await import('../scripts/replay-training.ts');
    const saved = process.env.OLLAMA_REQUEST_TIMEOUT_MS;
    try {
      delete process.env.OLLAMA_REQUEST_TIMEOUT_MS;
      expect(resolveRequestTimeoutMs({})).toBe(600_000);
      process.env.OLLAMA_REQUEST_TIMEOUT_MS = '900000';
      expect(resolveRequestTimeoutMs({})).toBe(900_000);
      expect(resolveRequestTimeoutMs({ requestTimeoutMs: 120000 })).toBe(120_000);
    } finally {
      if (saved === undefined) delete process.env.OLLAMA_REQUEST_TIMEOUT_MS;
      else process.env.OLLAMA_REQUEST_TIMEOUT_MS = saved;
    }
  });
});

describe('buildAsker', () => {
  const realFetch = globalThis.fetch;
  const posted: { url: string; model?: string }[] = [];
  const realEnvUrl = process.env.OLLAMA_SYSTEM_ONE_URL;
  const realEnvModel = process.env.OLLAMA_MODEL;

  beforeEach(() => {
    delete process.env.OLLAMA_SYSTEM_ONE_URL;
    delete process.env.OLLAMA_MODEL;
  });

  afterEach(() => {
    globalThis.fetch = realFetch;
    if (realEnvUrl !== undefined) process.env.OLLAMA_SYSTEM_ONE_URL = realEnvUrl;
    if (realEnvModel !== undefined) process.env.OLLAMA_MODEL = realEnvModel;
  });

  async function loadBuildAsker() {
    return (await import('../scripts/replay-training.ts')).buildAsker;
  }

  it('routes the ollama asker to the local endpoint with the clef-flash model', async () => {
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      posted.push({ url: String(url), model: (JSON.parse(String(init?.body)) as { model?: string }).model });
      return new Response(JSON.stringify({ answers: { q1: { type: 'noul', noul: 0.5 } } }), { status: 200 });
    }) as typeof fetch;
    const buildAsker = await loadBuildAsker();
    const asker = buildAsker({ dryRun: false, source: 'both', asker: 'ollama' });
    const response = await asker.ask('state', {
      q1: { type: 'noul', instructions: 'irrelevant' },
    });
    expect(posted[0]!.url).toBe('http://localhost:11434/v1/systemone');
    expect(posted[0]!.model).toBe('clef-flash');
    expect(response.answers.q1).toEqual({ type: 'noul', noul: 0.5 });
  });

  it('defaults to the local ollama judge when no asker flag is given', async () => {
    posted.length = 0;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      posted.push({ url: String(url), model: (JSON.parse(String(init?.body)) as { model?: string }).model });
      return new Response(JSON.stringify({ answers: { q1: { type: 'noul', noul: 0.5 } } }), { status: 200 });
    }) as typeof fetch;
    const buildAsker = await loadBuildAsker();
    const asker = buildAsker({ dryRun: false, source: 'both' });
    await asker.ask('state', {
      q1: { type: 'noul', instructions: 'irrelevant' },
    });
    expect(posted[0]!.url).toBe('http://localhost:11434/v1/systemone');
    expect(posted[0]!.model).toBe('clef-flash');
  });

  it('takes the endpoint from OLLAMA_SYSTEM_ONE_URL when no flag is given', async () => {
    posted.length = 0;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      posted.push({ url: String(url), model: (JSON.parse(String(init?.body)) as { model?: string }).model });
      return new Response(JSON.stringify({ answers: { q1: { type: 'noul', noul: 0.5 } } }), { status: 200 });
    }) as typeof fetch;
    const savedUrl = process.env.OLLAMA_SYSTEM_ONE_URL;
    process.env.OLLAMA_SYSTEM_ONE_URL = 'http://ollama.example:11434/v1/systemone';
    try {
      const buildAsker = await loadBuildAsker();
      const asker = buildAsker({ dryRun: false, source: 'both', asker: 'ollama' });
      await asker.ask('state', {
        q1: { type: 'noul', instructions: 'irrelevant' },
      });
      expect(posted[0]!.url).toBe('http://ollama.example:11434/v1/systemone');
    } finally {
      if (savedUrl === undefined) delete process.env.OLLAMA_SYSTEM_ONE_URL;
      else process.env.OLLAMA_SYSTEM_ONE_URL = savedUrl;
    }
  });

  it('lets the --base-url flag beat OLLAMA_SYSTEM_ONE_URL', async () => {
    posted.length = 0;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      posted.push({ url: String(url), model: (JSON.parse(String(init?.body)) as { model?: string }).model });
      return new Response(JSON.stringify({ answers: { q1: { type: 'noul', noul: 0.5 } } }), { status: 200 });
    }) as typeof fetch;
    const savedUrl = process.env.OLLAMA_SYSTEM_ONE_URL;
    process.env.OLLAMA_SYSTEM_ONE_URL = 'http://env-lan:11434/v1/systemone';
    try {
      const buildAsker = await loadBuildAsker();
      const asker = buildAsker({ dryRun: false, source: 'both', asker: 'ollama', baseUrl: 'http://flag-lan:11434/v1/systemone' });
      await asker.ask('state', {
        q1: { type: 'noul', instructions: 'irrelevant' },
      });
      expect(posted[0]!.url).toBe('http://flag-lan:11434/v1/systemone');
    } finally {
      if (savedUrl === undefined) delete process.env.OLLAMA_SYSTEM_ONE_URL;
      else process.env.OLLAMA_SYSTEM_ONE_URL = savedUrl;
    }
  });

  it('takes the model from OLLAMA_MODEL when no flag is given', async () => {
    posted.length = 0;
    globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
      posted.push({ url: String(url), model: (JSON.parse(String(init?.body)) as { model?: string }).model });
      return new Response(JSON.stringify({ answers: { q1: { type: 'noul', noul: 0.5 } } }), { status: 200 });
    }) as typeof fetch;
    const savedModel = process.env.OLLAMA_MODEL;
    process.env.OLLAMA_MODEL = 'clef-flash-lan';
    try {
      const buildAsker = await loadBuildAsker();
      const asker = buildAsker({ dryRun: false, source: 'both', asker: 'ollama' });
      await asker.ask('state', {
        q1: { type: 'noul', instructions: 'irrelevant' },
      });
      expect(posted[0]!.model).toBe('clef-flash-lan');
    } finally {
      if (savedModel === undefined) delete process.env.OLLAMA_MODEL;
      else process.env.OLLAMA_MODEL = savedModel;
    }
  });

  it('serializes local judge asks so one slow GPU is never asked in parallel', async () => {
    const events: string[] = [];
    globalThis.fetch = (async (_url: string | URL | Request, _init?: RequestInit) => {
      events.push('start');
      await new Promise((resolve) => setTimeout(resolve, 60));
      events.push('end');
      return new Response(JSON.stringify({ answers: { q1: { type: 'noul', noul: 0.5 } } }), { status: 200 });
    }) as typeof fetch;
    try {
      const buildAsker = await loadBuildAsker();
      const asker = buildAsker({ dryRun: false, source: 'both', asker: 'ollama' });
      const first = asker.ask('state one', { q1: { type: 'noul', instructions: 'irrelevant' } });
      const second = asker.ask('state two', { q1: { type: 'noul', instructions: 'irrelevant' } });
      await Promise.all([first, second]);
      // the second request may only start after the first has fully answered
      expect(events).toEqual(['start', 'end', 'start', 'end']);
    } finally {
      events.length = 0;
    }
  });

  it('keeps the cloud path key-gated', async () => {
    const buildAsker = await loadBuildAsker();
    const saved = process.env.TYPESAFE_API_KEY;
    delete process.env.TYPESAFE_API_KEY;
    try {
      expect(() => buildAsker({ dryRun: false, source: 'both', asker: 'cloud' })).toThrow(/--asker ollama/);    } finally {
      if (saved !== undefined) process.env.TYPESAFE_API_KEY = saved;
    }
  });

  it('passes baseUrl through resolveHookConfig', async () => {
    const { resolveHookConfig } = await import('../hooks/fast-compaction.ts');
    expect(resolveHookConfig({ baseUrl: 'http://pc:11434/v1/systemone' }).baseUrl).toBe(
      'http://pc:11434/v1/systemone',
    );
    expect(resolveHookConfig({}).baseUrl).toBeUndefined();
  });
});

describe('buildCompactOptions', () => {
  it('keeps maxStateTokens and stripping composed', async () => {
    const { buildCompactOptions } = await import('../scripts/replay-training.ts');
    expect(buildCompactOptions({ maxStateTokens: 1500, blocklistPath: undefined }, ['APRI'])).toEqual({
      maxStateTokens: 1500,
      strip: { blocklist: ['APRI'] },
    });
    // the pre-fix bug: strip overwrote the token ceiling wholesale
    expect(buildCompactOptions({ maxStateTokens: 1500 }, [])).toEqual({
      maxStateTokens: 1500,
      strip: {},
    });
    // no asker means local, so the local ceiling applies even with --no-strip
    expect(buildCompactOptions({ noStrip: true }, [])).toEqual({ maxStateTokens: 8000 });
  });

  it('defaults the local asker to 8000 state tokens; explicit flag and cloud do not', async () => {
    const { buildCompactOptions } = await import('../scripts/replay-training.ts');
    expect(buildCompactOptions({}, [])).toEqual({ maxStateTokens: 8000, strip: {} });
    expect(buildCompactOptions({ asker: 'ollama' }, [])).toEqual({ maxStateTokens: 8000, strip: {} });
    expect(buildCompactOptions({ asker: 'cloud' }, [])).toEqual({ strip: {} });
    expect(buildCompactOptions({ maxStateTokens: 500, asker: 'ollama' }, [])).toEqual({
      maxStateTokens: 500,
      strip: {},
    });
  });
});

describe('fetchWithTimeout', () => {
  it('aborts a hanging judge and surfaces a ledgerable timeout error', async () => {
    const { fetchWithTimeout, REQUEST_TIMEOUT_MS } = await import('../scripts/replay-training.ts');
    const realFetch = globalThis.fetch;
    globalThis.fetch = ((_url, init) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(new Error('aborted by test harness')));
      })) as typeof fetch;
    try {
      await expect(fetchWithTimeout(10)('http://judge', undefined)).rejects.toThrow(/timed out/);
    } finally {
      globalThis.fetch = realFetch;
    }
    expect(REQUEST_TIMEOUT_MS).toBeGreaterThan(0);
  });
});

describe('segmentByFit', () => {
  const resolved = {
    goal: '',
    keepThreshold: 0.5,
    preserveRecentMessages: 6,
    preserveStartMessages: 2,
    maxStateTokens: 30000,
    maxRequestTokens: 35000,
    maxQuestionsPerRequest: 12,
    askConcurrency: 2,
    truncateHeadChars: 300,
    trainingSource: 'replay' as const,
  };

  it('keeps a fitting history in one window', async () => {
    const { segmentByFit } = await import('../src/index.js');
    const messages: Message[] = [
      { role: 'user', text: 'hello world', toolUses: [] } as Message,
      { role: 'assistant', text: 'hi there', toolUses: [] } as Message,
    ];
    expect(segmentByFit(messages, resolved)).toEqual([messages]);
  });

  it('returns null when even a single message cannot fit', async () => {
    const { segmentByFit } = await import('../src/index.js');
    const giant: Message = { role: 'user', text: 'x'.repeat(5000), toolUses: [] } as Message;
    expect(segmentByFit([giant], { ...resolved, maxStateTokens: 10 })).toBeNull();
  });

  it('splits an oversized history into fitting windows at user prompts', async () => {
    const { segmentByFit } = await import('../src/index.js');
    const big: Message[] = [];
    for (let i = 0; i < 20; i++) {
      big.push({
        role: i % 3 === 0 ? 'user' : 'assistant',
        text: i % 3 === 0 ? `prompt ${i}` : 'y'.repeat(700),
        toolUses: [],
      } as Message);
    }
    const segments = segmentByFit(big, { ...resolved, maxStateTokens: 600, preserveRecentMessages: 6 });
    expect(segments).not.toBeNull();
    expect(segments!.length).toBeGreaterThan(1);
    expect(segments!.flat()).toEqual(big); // lossless, ordered
    for (const segment of segments!.slice(1)) {
      // every later window begins at one of the user's prompts
      expect(segment[0]!.role).toBe('user');
    }
  });

  it('never snaps a cut to a user prompt whose shorter window no longer fits', async () => {
    const { segmentByFit } = await import('../src/index.js');
    const small = (): Message => ({ role: 'assistant', text: 'x'.repeat(30), toolUses: [] }) as Message;
    const fat = (): Message => ({ role: 'assistant', text: 'F'.repeat(1200), toolUses: [] }) as Message;
    const prompt = (t: string): Message => ({ role: 'user', text: t, toolUses: [] }) as Message;
    // fitState is not monotonic under prefixes: in the SHORT prompt-cut window
    // the fat message sits in the pinned tail (text abridged to ~550 chars),
    // while in the LARGER binary-search window it is old (collapsed to ~25
    // chars) — so the naive snap produces a SHORTER window with a BIGGER state
    const messages: Message[] = [
      small(), small(), prompt('p0'), small(), small(), small(),
      fat(), small(), small(), small(),
      prompt('cut'), small(), small(), small(), small(),
      fat(), small(), small(), small(), small(), small(), small(), small(), small(),
    ];
    // calibrate the ceiling from the real token count of the trapped window
    const trap = planReplay(messages.slice(0, 10), { maxStateTokens: 100, preserveRecentMessages: 6 });
    expect(trap.error).toMatch(/~(\d+) tokens/);
    const ceiling = Number(/~(\d+) tokens/.exec(trap.error!)![1]) - 10;
    const segments = segmentByFit(messages, { ...resolved, maxStateTokens: ceiling });
    expect(segments).not.toBeNull();
    expect(segments!.length).toBeGreaterThan(1);
    expect(segments!.flat()).toEqual(messages); // lossless, ordered
    for (const segment of segments!) {
      // every window fits by construction — the snap-back cannot break that
      expect(
        planReplay(segment, { maxStateTokens: ceiling, preserveRecentMessages: 6 }).error,
      ).toBeUndefined();
    }
  });
});

describe('segmented replay of oversized sessions', () => {
  function localAsker(): JevAsker {
    return {
      async ask(_state, questions: JevQuestions) {
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
        };
      },
    };
  }

  function localDeps(dir: string) {
    return { dir, delayMs: 0, sleep: async () => {}, now: () => 1234 };
  }

  it('replays a too-large session in windows and ledgers the segment count', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, {
      big: { at: 1, error: 'history too large for Jev (~900 tokens after truncation, limit 100)' },
    });
    const big: Message[] = [];
    for (let i = 0; i < 40; i++) {
      big.push({
        role: i % 3 === 0 ? 'user' : 'assistant',
        text: i % 3 === 0 ? `prompt ${i}` : 'y'.repeat(400),
        toolUses: [],
      } as Message);
    }
    const sources: ReplaySourceSpec[] = [{ key: 'big', label: 'big', load: async () => big }];
    const summary = await runReplay(sources, localAsker(), () => {}, {
      ...localDeps(dir),
      segmentOversized: true,
      compactOptions: { maxStateTokens: 950, preserveRecentMessages: 6 },
    });
    expect(summary.replayed).toBe(1);
    expect(summary.records).toBeGreaterThanOrEqual(1);
    const index = await loadReplayIndex(dir);
    expect(index.big?.records).toBe(summary.records);
    expect(index.big?.error).toBeUndefined();
  });

  it('treats a ledgered 413 64 KiB rejection as too large, so segmentation retries it', () => {
    expect(
      isTooLargeError({
        error: 'Jev request failed (413): {"error":"text and schema must not exceed 64 KiB"}',
      }),
    ).toBe(true);
  });

  it('treats a ledgered 400 context rejection as too large, so segmentation retries it', () => {
    expect(
      isTooLargeError({
        error:
          'Jev request failed (400): {"error":"decision prompt exceeds the model context (input is never truncated)"}',
      }),
    ).toBe(true);
  });

  it('classifies the post-rename "judge" phrasings the same as the old "Jev" ones', async () => {
    const { isDeterministicJevError, isRetryableLedgerError } = await import('../src/index.js');
    expect(
      isDeterministicJevError('judge request failed (413): {"error":"text and schema must not exceed 64 KiB"}'),
    ).toBe(true);
    expect(isRetryableLedgerError({ error: 'judge request timed out (no answer within 600s)' })).toBe(true);
    expect(isRetryableLedgerError({ error: 'history too large for the judge (~100 tokens after truncation, limit 90)' })).toBe(true);
  });

  it('leaves other errors ledgered even in segment mode', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, {
      crashed: {
        at: 1,
        error: 'Jev request failed (500): {"error":"llama-server process has terminated: exit status 0xc0000005"}',
      },
    });
    const sources: ReplaySourceSpec[] = [
      {
        key: 'crashed',
        label: 'crashed',
        load: async () => [{ role: 'user', text: 'anything', toolUses: [] } as Message],
      },
    ];
    const summary = await runReplay(sources, localAsker(), () => {}, {
      ...localDeps(dir),
      segmentOversized: true,
    });
    expect(summary.skippedDone).toBe(1);
    expect(summary.sessions).toBe(0);
  });

  it('retries a ledgered timeout entry in segment mode (infra-bound, not session-bound)', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, {
      slow: { at: 1, error: 'Jev request timed out (no answer within 240s)' },
    });
    const sources: ReplaySourceSpec[] = [
      {
        key: 'slow',
        label: 'slow',
        load: async () => [{ role: 'user', text: 'anything', toolUses: [] } as Message],
      },
    ];
    const summary = await runReplay(sources, localAsker(), () => {}, {
      ...localDeps(dir),
      segmentOversized: true,
    });
    expect(summary.skippedDone).toBe(0);
    const index = await loadReplayIndex(dir);
    expect(index.slow?.error).toBeUndefined();
  });
});

describe('segmented replay completion', () => {
  function localAsker(): JevAsker {
    return {
      async ask(_state, questions: JevQuestions) {
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
        };
      },
    };
  }

  it('completes a session whose tail segment is a legitimate no-op', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, {
      tail: { at: 1, error: 'history too large for Jev (~5000 tokens after truncation, limit 600)' },
    });
    // fat messages force splits; the short tail after the last cut is a no-op window
    const big: Message[] = [];
    for (let i = 0; i < 12; i++) {
      big.push({
        role: i % 3 === 0 ? 'user' : 'assistant',
        text: i % 3 === 0 ? `prompt ${i}` : 'z'.repeat(700),
        toolUses: [],
      } as Message);
    }
    big.push({ role: 'user', text: 'final short question', toolUses: [] } as Message);
    const sources: ReplaySourceSpec[] = [{ key: 'tail', label: 'tail', load: async () => big }];
    const summary = await runReplay(sources, localAsker(), () => {}, {
      dir,
      delayMs: 0,
      sleep: async () => {},
      now: () => 1234,
      segmentOversized: true,
      compactOptions: { maxStateTokens: 900, preserveRecentMessages: 6 },
    });
    expect(summary.replayed).toBe(1);
    expect(summary.records).toBeGreaterThanOrEqual(1);
    const index = await loadReplayIndex(dir);
    expect(index.tail?.records).toBe(summary.records);
    expect(index.tail?.error).toBeUndefined();
  });
});

describe('segmented replay transient failures', () => {
  function localDeps(dir: string) {
    return { dir, delayMs: 0, sleep: async () => {}, now: () => 1234 };
  }

  it('feeds the breaker on a transient segment failure and stays retriable', async () => {
    const dir = await tempDir();
    await saveReplayIndex(dir, {
      flaky: { at: 1, error: 'history too large for Jev (~5000 tokens after truncation, limit 600)' },
    });
    const big: Message[] = [];
    for (let i = 0; i < 12; i++) {
      big.push({
        role: i % 3 === 0 ? 'user' : 'assistant',
        text: i % 3 === 0 ? `prompt ${i}` : 'z'.repeat(700),
        toolUses: [],
      } as Message);
    }
    let asks = 0;
    const asker: JevAsker = {
      async ask(_state, questions: JevQuestions) {
        asks++;
        // fail the first segment that actually asks (later windows can be no-ops)
        if (asks === 1) throw new Error('fetch failed: connection reset');
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
        };
      },
    };
    const sources: ReplaySourceSpec[] = [{ key: 'flaky', label: 'flaky', load: async () => big }];
    const summary = await runReplay(sources, asker, () => {}, {
      ...localDeps(dir),
      segmentOversized: true,
      compactOptions: { maxStateTokens: 700, preserveRecentMessages: 6 },
    });
    expect(summary.errors).toBe(1);
    expect(summary.stoppedAfterFailures).toBe(false);
    const index = await loadReplayIndex(dir);
    expect(index.flaky?.error).toMatch(/history too large/); // unchanged: retriable, will re-run
  });
});
