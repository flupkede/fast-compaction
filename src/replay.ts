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
