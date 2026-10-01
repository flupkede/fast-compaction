import { appendFile, mkdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import type { TrainingRecord } from './types.js';

export const DEFAULT_TRAINING_DIR = join(homedir(), '.fast-jev', 'training');

export interface TrainingLogOptions {
  /** Defaults to `~/.fast-jev/training`. */
  dir?: string;
}

/**
 * Appends one JSON line per Jev compaction to `<dir>/compactions-YYYY-MM.jsonl`.
 * The directory is created on first use; the file rotates monthly. Data stays
 * on this machine.
 */
export function createTrainingLogger(
  options: TrainingLogOptions = {},
): (record: TrainingRecord) => Promise<void> {
  const dir = options.dir ?? DEFAULT_TRAINING_DIR;
  let ready: Promise<unknown> | undefined;
  return async (record) => {
    ready ??= mkdir(dir, { recursive: true });
    await ready;
    const file = join(dir, `compactions-${new Date().toISOString().slice(0, 7)}.jsonl`);
    await appendFile(file, `${JSON.stringify(record)}\n`, 'utf8');
  };
}
