import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
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
  const dir = await mkdtemp(join(tmpdir(), 'fast-jev-transcripts-'));
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
    expect(index.flaky?.error).toMatch(/timed out/);
    expect(index.crashy?.error).toMatch(/llama-server/);
    expect(index.ok?.records).toBe(1);
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

  it('parses --max-state-tokens and validates the number', async () => {
    const { parseReplayArgs } = await import('../src/index.js');
    expect(parseReplayArgs(['--max-state-tokens', '2000'])).toMatchObject({ maxStateTokens: 2000 });
    expect(() => parseReplayArgs(['--max-state-tokens'])).toThrow(/needs a value/);
    expect(() => parseReplayArgs(['--max-state-tokens', 'abc'])).toThrow(/>= 0/);
    expect(() => parseReplayArgs(['--max-state-tokens', '-5'])).toThrow(/>= 0/);
  });
});

describe('buildAsker', () => {
  const realFetch = globalThis.fetch;
  const posted: { url: string; model?: string }[] = [];

  afterEach(() => {
    globalThis.fetch = realFetch;
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

  it('keeps the cloud path key-gated', async () => {
    const buildAsker = await loadBuildAsker();
    const saved = process.env.TYPESAFE_API_KEY;
    delete process.env.TYPESAFE_API_KEY;
    try {
      expect(() => buildAsker({ dryRun: false, source: 'both' })).toThrow(/--asker ollama/);
    } finally {
      if (saved !== undefined) process.env.TYPESAFE_API_KEY = saved;
    }
  });

  it('passes baseUrl through resolveHookConfig', async () => {
    const { resolveHookConfig } = await import('../hooks/fast-jev.ts');
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
    expect(buildCompactOptions({ noStrip: true }, [])).toEqual({});
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
