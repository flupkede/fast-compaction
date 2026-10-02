import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { batchCalls, compact, resolveOptions } from './compact.js';
import { collectToolCalls, fitState } from './state.js';
import type { CompactOptions, CompactResult, JevAsker, Message } from './types.js';

export interface ReplayIndexEntry {
  at: number;
  records?: number;
  error?: string;
}

/** Resume ledger: which stored sessions were already replayed, and how it went. */
export type ReplayIndex = Record<string, ReplayIndexEntry>;

export const REPLAY_INDEX_FILE = 'replay-index.json';

export async function loadReplayIndex(dir: string): Promise<ReplayIndex> {
  try {
    return JSON.parse(await readFile(join(dir, REPLAY_INDEX_FILE), 'utf8')) as ReplayIndex;
  } catch (error) {
    // a missing ledger is a fresh start; a corrupt one must not silently reset it
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {};
    throw error;
  }
}

export async function saveReplayIndex(dir: string, index: ReplayIndex): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, REPLAY_INDEX_FILE), `${JSON.stringify(index, null, 2)}\n`, 'utf8');
}

/** Every Claude Code session file under `root`, recursively, oldest-path-first. */
export async function listClaudeTranscriptFiles(root: string): Promise<string[]> {
  const out: string[] = [];
  const walk = async (dir: string): Promise<void> => {
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.isFile() && entry.name.endsWith('.jsonl')) out.push(path);
    }
  };
  await walk(root);
  return out.sort();
}

export interface SessionPlan {
  messages: number;
  /** Candidate calls Jev would be asked about. */
  calls: number;
  /** Estimated Jev requests (state + question batches). */
  requests: number;
  /** Estimated state tokens. */
  tokens: number;
  /** Why the session cannot be replayed, when it cannot. */
  error?: string;
}

/** Counts what a replay of this transcript would ask, without asking it. */
export function planReplay(messages: readonly Message[]): SessionPlan {
  const resolved = resolveOptions();
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const candidates = calls.filter((call) => !call.pinned);
  if (candidates.length === 0) {
    return { messages: messages.length, calls: 0, requests: 0, tokens: 0 };
  }
  try {
    const fitted = fitState(messages, calls, resolved);
    const batches = batchCalls(candidates, fitted.tokens, resolved);
    return {
      messages: messages.length,
      calls: candidates.length,
      requests: batches.length,
      tokens: fitted.tokens,
    };
  } catch (error) {
    return {
      messages: messages.length,
      calls: candidates.length,
      requests: 0,
      tokens: 0,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/** Replays one stored transcript through the live compaction path, tagged `replay`. */
export async function replaySession(
  messages: readonly Message[],
  asker: JevAsker,
  trainingLog: (record: import('./types.js').TrainingRecord) => void | Promise<void>,
  options: CompactOptions = {},
): Promise<CompactResult> {
  return compact(messages, asker, { ...options, trainingLog, trainingSource: 'replay' });
}

export type ReplaySourceKind = 'claude' | 'opencode' | 'both';

/** Flags `replay-training` accepts; strings left unset take the script defaults. */
export interface ReplayArgs {
  dryRun: boolean;
  source: ReplaySourceKind;
  dir?: string;
  claudeRoot?: string;
  opencodeDb?: string;
  limit?: number;
  delayMs?: number;
  minMessages?: number;
}

/** Parses replay CLI flags; throws on anything unknown or malformed. */
export function parseReplayArgs(argv: readonly string[]): ReplayArgs {
  const args: ReplayArgs = { dryRun: false, source: 'both' };
  const value = (flag: string, arg: string | undefined): string => {
    if (arg === undefined || arg.startsWith('--')) throw new Error(`${flag} needs a value`);
    return arg;
  };
  const number = (flag: string, arg: string | undefined): number => {
    const parsed = Number(value(flag, arg));
    if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${flag} must be a number >= 0`);
    return parsed;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === '--dry-run') args.dryRun = true;
    else if (arg === '--source') {
      const kind = value(arg, argv[++i]);
      if (kind !== 'claude' && kind !== 'opencode' && kind !== 'both') {
        throw new Error(`--source must be claude, opencode or both, got "${kind}"`);
      }
      args.source = kind;
    } else if (arg === '--dir') args.dir = value(arg, argv[++i]);
    else if (arg === '--claude-root') args.claudeRoot = value(arg, argv[++i]);
    else if (arg === '--opencode-db') args.opencodeDb = value(arg, argv[++i]);
    else if (arg === '--limit') args.limit = number(arg, argv[++i]);
    else if (arg === '--delay-ms') args.delayMs = number(arg, argv[++i]);
    else if (arg === '--min-messages') args.minMessages = number(arg, argv[++i]);
    else throw new Error(`unknown argument "${arg}"`);
  }
  return args;
}

/** One stored session the runner can load, already keyed for the resume ledger. */
export interface ReplaySourceSpec {
  key: string;
  label: string;
  load: () => Promise<readonly Message[]>;
}

export interface ReplayRunOptions {
  /** Training dir holding the JSONL records and the resume ledger. */
  dir: string;
  /** Max sessions to replay this run; unbounded when unset. */
  limit?: number;
  /** Pause between replayed sessions, ms. */
  delayMs: number;
  sleep: (ms: number) => Promise<void>;
  now: () => number;
}

export interface ReplayRunSummary {
  sessions: number;
  replayed: number;
  records: number;
  skippedDone: number;
  noCandidates: number;
  errors: number;
}

/**
 * Replays every not-yet-ledgered source through `replaySession`, persisting
 * the resume ledger after each session so an interrupted run continues where
 * it stopped. `onSession` observes progress; throws only on ledger corruption.
 */
export async function runReplay(
  sources: readonly ReplaySourceSpec[],
  asker: JevAsker,
  trainingLog: (record: import('./types.js').TrainingRecord) => void | Promise<void>,
  options: ReplayRunOptions,
  onSession?: (event: { key: string; outcome: 'replayed' | 'done' | 'no-candidates' | 'error'; detail?: string }) => void,
): Promise<ReplayRunSummary> {
  const index = await loadReplayIndex(options.dir);
  const summary: ReplayRunSummary = {
    sessions: 0,
    replayed: 0,
    records: 0,
    skippedDone: 0,
    noCandidates: 0,
    errors: 0,
  };
  for (const source of sources) {
    if (options.limit !== undefined && summary.replayed >= options.limit) break;
    if (index[source.key]) {
      summary.skippedDone++;
      continue;
    }
    summary.sessions++;
    let messages: readonly Message[];
    try {
      messages = await source.load();
    } catch (error) {
      index[source.key] = {
        at: options.now(),
        error: error instanceof Error ? error.message : String(error),
      };
      summary.errors++;
      onSession?.({ key: source.key, outcome: 'error', detail: index[source.key]!.error });
      await saveReplayIndex(options.dir, index);
      continue;
    }
    const plan = planReplay(messages);
    if (plan.error) {
      index[source.key] = { at: options.now(), error: plan.error };
      summary.errors++;
      onSession?.({ key: source.key, outcome: 'error', detail: plan.error });
    } else if (plan.calls === 0) {
      index[source.key] = { at: options.now(), records: 0 };
      summary.noCandidates++;
      onSession?.({ key: source.key, outcome: 'no-candidates' });
    } else {
      try {
        // observe the record append: a swallowed write failure must stay retriable
        let emitted = false;
        let write: Promise<void> = Promise.resolve();
        const observing: typeof trainingLog = (record) => {
          emitted = true;
          write = (async () => {
            await trainingLog(record);
          })();
          return write;
        };
        await replaySession(messages, asker, observing);
        await write;
        if (!emitted) throw new Error('no training record emitted');
        index[source.key] = { at: options.now(), records: 1 };
        summary.replayed++;
        summary.records++;
        onSession?.({
          key: source.key,
          outcome: 'replayed',
          detail: `${plan.requests} request(s), ${plan.calls} calls`,
        });
      } catch (error) {
        // network/runtime failures stay retriable: no ledger entry
        summary.errors++;
        onSession?.({
          key: source.key,
          outcome: 'error',
          detail: error instanceof Error ? error.message : String(error),
        });
        continue;
      }
    }
    await saveReplayIndex(options.dir, index);
    if (options.delayMs > 0) await options.sleep(options.delayMs);
  }
  return summary;
}
