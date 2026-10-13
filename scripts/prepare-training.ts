/**
 * Converts collected training records into fine-tuning examples for a
 * System One-compatible judge. Each record yields one call-judgement example
 * and (schema 2) one message-judgement example: the student learns to map
 * {state, questions, keepThreshold} to the noul answers the teacher gave.
 *
 *   npm run prepare-training -- [--dir <training-dir>] [--out <dir>]
 *                               [--val-fraction 0.05] [--limit n]
 */
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { homedir } from 'node:os';
import type { TrainingRecord } from '../src/index.js';

export const JUDGE_SYSTEM_PROMPT = `You are a context-compaction judge. The user message holds a JSON object with "state" (the conversation history, tool results omitted, identifiers pseudonymised), "questions" (ids mapped to noul instructions) and "keepThreshold". Reply with a JSON object mapping every question id to {"type":"noul","noul":p} where p in [0,1] is the probability the item should stay in the compacted history. Output JSON only.`;

export interface FineTuneExample {
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[];
}

/** The noul target for one call: two questions per call id (keepCall + keepResult). */
export function callTarget(
  answers: TrainingRecord['answers'],
): Record<string, { type: 'noul'; noul: number }> {
  const target: Record<string, { type: 'noul'; noul: number }> = {};
  for (const [id, answer] of Object.entries(answers)) {
    target[`call_${id}`] = { type: 'noul', noul: answer.keepCall };
    target[`result_${id}`] = { type: 'noul', noul: answer.keepResult };
  }
  return target;
}

function example(
  system: string,
  state: unknown,
  questions: Record<string, unknown>,
  keepThreshold: number,
  target: Record<string, unknown>,
): FineTuneExample {
  return {
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: JSON.stringify({ state, questions, keepThreshold }) },
      { role: 'assistant', content: JSON.stringify(target) },
    ],
  };
}

/** One record becomes up to two examples: the call set and the message set. */
export function toFineTuneExamples(record: TrainingRecord): FineTuneExample[] {
  const examples: FineTuneExample[] = [];
  if (Object.keys(record.questions ?? {}).length > 0) {
    examples.push(
      example(JUDGE_SYSTEM_PROMPT, record.state, record.questions, record.keepThreshold, callTarget(record.answers)),
    );
  }
  if (record.messageQuestions && Object.keys(record.messageQuestions).length > 0) {
    const target: Record<string, { type: 'noul'; noul: number }> = {};
    for (const [id, noul] of Object.entries(record.messageAnswers ?? {})) {
      target[id] = { type: 'noul', noul };
    }
    examples.push(
      example(JUDGE_SYSTEM_PROMPT, record.state, record.messageQuestions, record.keepThreshold, target),
    );
  }
  return examples;
}

/** FNV-1a: cheap deterministic hash for the stable train/val split. */
function fnv1a(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** Stable per-state split: independent of file order and reruns. */
export function splitOf(state: unknown, valFraction: number): 'train' | 'val' {
  return fnv1a(JSON.stringify(state)) / 0xffffffff < valFraction ? 'val' : 'train';
}

async function loadRecords(dir: string, limit?: number): Promise<TrainingRecord[]> {
  const files = (await readdir(dir)).filter((name) => /^compactions-.*\.jsonl$/.test(name)).sort();
  const records: TrainingRecord[] = [];
  for (const name of files) {
    const raw = await readFile(join(dir, name), 'utf8');
    for (const line of raw.split(/\r?\n/)) {
      if (line.trim().length === 0) continue;
      try {
        records.push(JSON.parse(line) as TrainingRecord);
      } catch {
        // a torn append line from an interrupted run: skip, it is not data loss
      }
      if (limit !== undefined && records.length >= limit) return records;
    }
  }
  return records;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const stringFlag = (name: string): string | undefined => {
    const index = argv.indexOf(`--${name}`);
    return index === -1 || argv[index + 1] === undefined ? undefined : argv[index + 1];
  };
  const numberFlag = (name: string): number | undefined => {
    const value = stringFlag(name);
    return value === undefined ? undefined : Number(value);
  };
  const dir = stringFlag('dir') ?? join(homedir(), '.fast-compaction', 'training');
  const out = stringFlag('out') ?? join(dir, 'fine-tune');
  const valFraction = numberFlag('val-fraction') ?? 0.05;
  const limit = numberFlag('limit');

  const records = await loadRecords(dir, limit);
  const train: string[] = [];
  const val: string[] = [];
  let schema1 = 0;
  for (const record of records) {
    if (record.schema !== 2) schema1++;
    for (const item of toFineTuneExamples(record)) {
      (splitOf(record.state, valFraction) === 'val' ? val : train).push(JSON.stringify(item));
    }
  }
  await mkdir(out, { recursive: true });
  await writeFile(join(out, 'train.jsonl'), `${train.join('\n')}\n`, 'utf8');
  await writeFile(join(out, 'val.jsonl'), `${val.join('\n')}\n`, 'utf8');
  await writeFile(
    join(out, 'manifest.json'),
    `${JSON.stringify(
      {
        generated: new Date().toISOString(),
        format: 'systemone-judge-v1',
        valFraction,
        records: records.length,
        schema1,
        examples: { train: train.length, val: val.length },
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  console.log(
    `prepare-training: ${records.length} record(s) -> ${train.length} train / ${val.length} val example(s) in ${out}`,
  );
}

if (process.argv[1] && process.argv[1].endsWith('prepare-training.ts')) {
  await main();
}
