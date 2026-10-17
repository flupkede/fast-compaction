import type { TrainingRecord } from './types.js';

export interface TrainingLogOptions {
  /** Defaults to `~/.fast-compaction/training`. */
  dir?: string;
}

/** `~/.fast-compaction/training` without `node:*` imports — the plugin sandbox rejects those in hook-reachable modules. */
export function defaultTrainingDir(): string {
  const home = process.env.USERPROFILE ?? process.env.HOME ?? '.';
  return `${home}/.fast-compaction/training`;
}

/**
 * Appends one JSON line per Jev compaction to `<dir>/compactions-YYYY-MM.jsonl`.
 * The directory is created on first use; the file rotates monthly. Data stays
 * on this machine. Node built-ins load lazily so this module stays importable
 * inside the plugin sandbox.
 */
export function createTrainingLogger(
  options: TrainingLogOptions = {},
): (record: TrainingRecord) => Promise<void> {
  return async (record) => {
    const [{ appendFile, mkdir }, { join }] = await Promise.all([
      import('node:fs/promises'),
      import('node:path'),
    ]);
    const dir = options.dir ?? defaultTrainingDir();
    await mkdir(dir, { recursive: true });
    const file = join(dir, `compactions-${new Date().toISOString().slice(0, 7)}.jsonl`);
    await appendFile(file, `${JSON.stringify(record)}\n`, 'utf8');
  };
}
