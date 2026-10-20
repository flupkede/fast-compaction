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
import { resolveHookConfig } from '../hooks/fast-compaction.js';

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
    expect(Object.keys(record.messageQuestions ?? {}).sort()).toEqual(['msg_3']);
    // keys are msg_<i>, matching the messageQuestions join
    expect(Object.keys(record.messageAnswers ?? {}).sort()).toEqual(['msg_3']);
    expect(record.messageDecisions?.map((d) => [d.i, d.reason])).toEqual([
      [0, 'user'],
      [3, 'kept'],
      [4, 'user'],
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
  it('no longer carries training options — the hook sandbox forbids fs access', () => {
    expect('trainingLog' in resolveHookConfig({ trainingLog: true })).toBe(false);
    expect('trainingDir' in resolveHookConfig({ trainingDir: '/tmp/jev' })).toBe(false);
  });
});
