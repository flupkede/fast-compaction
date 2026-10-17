import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  compact,
  createTrainingLogger,
  defaultTrainingDir,
  type JevAsker,
  type JevQuestions,
  type Message,
  type TrainingRecord,
} from '../src/index.js';
import { resolveHookConfig, compactSession } from '../hooks/fast-compaction.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Fix the failing test.'),
    message('assistant', '', {
      toolUses: [{ tool_use_id: 'tool-1', tool: 'Read', input: { file_path: 'src/a.ts' }, text: fileA }],
    }),
    message('user', '', { toolResults: [{ tool_use_id: 'tool-1', text: fileA }] }),
    message('assistant', 'looks fine'),
    message('user', 'go ahead'),
  ];
}

function sessionTranscript(): Parameters<typeof compactSession>[0] {
  return [
    { role: 'user', text: 'Fix the failing test.', toolUses: [] },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'tool-1', tool: 'Read', input: { file_path: 'src/a.ts' }, text: fileA }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'tool-1', text: fileA, isError: false }] },
    { role: 'assistant', text: 'looks fine', toolUses: [] },
    { role: 'user', text: 'go ahead', toolUses: [] },
  ];
}

function fakeJev(): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
        ),
      };
    },
  };
}

const dirs: string[] = [];

afterEach(async () => {
  while (dirs.length > 0) await rm(dirs.pop()!, { recursive: true, force: true });
});

async function tempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'fast-compaction-training-'));
  dirs.push(dir);
  return dir;
}

describe('compact training log', () => {
  it('records the redacted state, questions, answers and decisions once per compaction', async () => {
    const records: TrainingRecord[] = [];
    const result = await compact(transcript(), fakeJev(), {
      preserveRecentMessages: 0,
      goal: 'fix the test',
      trainingLog: (record) => void records.push(record),
    });
    expect(records).toHaveLength(1);
    const record = records[0]!;
    expect(record.schema).toBe(2);
    expect(record.source).toBe('live');
    expect(new Date(record.ts).toString()).not.toBe('Invalid Date');
    expect(record.state.goal).toBe('fix the test');
    expect(record.state.history.length).toBeGreaterThan(0);
    expect(record.stateTokens).toBe(result.stats.stateTokens);
    expect(record.stateStage).toBe(result.stats.stateStage);
    expect(Object.keys(record.questions).sort()).toEqual([
      'call_t1',
      'result_t1',
    ]);
    expect(Object.keys(record.messageQuestions ?? {}).sort()).toEqual(['msg_3', 'msg_4']);
    // keys are msg_<i>, matching the messageQuestions join
    expect(Object.keys(record.messageAnswers ?? {}).sort()).toEqual(['msg_3', 'msg_4']);
    expect(record.messageDecisions?.map((d) => [d.i, d.reason])).toEqual([
      [0, 'recent'],
      [3, 'kept'],
      [4, 'kept'],
    ]);
    expect(record.answers).toEqual({ t1: { keepCall: 0.9, keepResult: 0.9 } });
    expect(record.decisions).toEqual(result.decisions);
    expect(record.keepThreshold).toBe(0.5);
  });

  it('tags replay provenance when trainingSource is set', async () => {
    const records: TrainingRecord[] = [];
    await compact(transcript(), fakeJev(), {
      preserveRecentMessages: 0,
      trainingSource: 'replay',
      trainingLog: (record) => void records.push(record),
    });
    expect(records[0]!.source).toBe('replay');
  });

  it('logs nothing when no call was asked and never breaks compaction when the logger throws', async () => {
    const records: TrainingRecord[] = [];
    const options = {
      preserveRecentMessages: 6,
      trainingLog: () => {
        throw new Error('disk full');
      },
    };
    const asked = await compact(transcript(), fakeJev(), options);
    expect(asked.stats.requests).toBe(0);
    expect(records).toHaveLength(0);

    const withCandidates = { ...options, preserveRecentMessages: 0 };
    const result = await compact(transcript(), fakeJev(), withCandidates);
    expect(result.stats.requests).toBe(2); // one call batch + one message batch
  });
});

describe('createTrainingLogger', () => {
  it('appends one JSON line per record into a monthly file it creates', async () => {
    const dir = await tempDir();
    const logger = createTrainingLogger({ dir });
    const record: TrainingRecord = {
      schema: 1,
      ts: '2026-10-01T00:00:00.000Z',
      source: 'live',
      state: { context: 'c', goal: 'g', history: [] },
      stateTokens: 10,
      stateStage: 'full',
      questions: {},
      answers: {},
      decisions: [],
      keepThreshold: 0.5,
    };
    await logger(record);
    await logger(record);
    const month = new Date().toISOString().slice(0, 7);
    const lines = (await readFile(join(dir, `compactions-${month}.jsonl`), 'utf8')).trim().split('\n');
    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0]!)).toEqual(record);
  });

  it('defaults the directory to ~/.fast-compaction/training', () => {
    expect(defaultTrainingDir()).toMatch(/\.fast-compaction[/\\]training$/);
  });
});

describe('resolveHookConfig training options', () => {
  it('reads the boolean flag and directory, defaulting the flag to false', () => {
    expect(resolveHookConfig({}).trainingLog).toBe(false);
    expect(resolveHookConfig({ trainingLog: true, trainingDir: '/tmp/jev' })).toMatchObject({
      trainingLog: true,
      trainingDir: '/tmp/jev',
    });
    expect(resolveHookConfig({ trainingLog: 'yes' }).trainingLog).toBe(false);
  });
});

describe('compactSession training wiring', () => {
  it('writes a record to disk only when the flag is on', async () => {
    const dir = await tempDir();
    const base = resolveHookConfig({ preserveRecentMessages: 0 });
    const fetch = async (): Promise<{ status: number; ok: boolean; text: string }> => ({
      status: 200,
      ok: true,
      text: JSON.stringify({ answers: { call_t1: { noul: 0.9 }, result_t1: { noul: 0.1 } } }),
    });
    const off = { ...base, apiKey: 'k', trainingLog: false, trainingDir: dir };
    await compactSession(sessionTranscript(), off, fetch);
    await expect(readFile(join(dir, `compactions-${new Date().toISOString().slice(0, 7)}.jsonl`), 'utf8')).rejects.toThrow();

    const on = { ...base, apiKey: 'k', trainingLog: true, trainingDir: dir };
    const { result } = await compactSession(sessionTranscript(), on, fetch);
    expect(result.stats.requests).toBe(2); // call batch + message batch
    const lines = (await readFile(join(dir, `compactions-${new Date().toISOString().slice(0, 7)}.jsonl`), 'utf8')).trim().split('\n');
    expect(lines).toHaveLength(1);
    const record = JSON.parse(lines[0]!) as TrainingRecord;
    expect(record.answers).toEqual({ t1: { keepCall: 0.9, keepResult: 0.1 } });
    expect(record.keepThreshold).toBe(0.5);
  });
});
