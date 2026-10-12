import { mkdir, readdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import {
  batchCalls,
  compact,
  messageChunkSize,
  resolveOptions,
} from './compact.js';
import { collectToolCalls, fitState, isPinned } from './state.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ResolvedCompactOptions,
} from './types.js';

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
  const target = join(dir, REPLAY_INDEX_FILE);
  const tmp = `${target}.tmp`;
  await writeFile(tmp, `${JSON.stringify(index, null, 2)}\n`, 'utf8');
  // rename over the old ledger so a crash mid-write cannot leave partial JSON behind
  await rename(tmp, target);
}

/** Every Claude Code session file under `root`, recursively, in deterministic path order. */
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
  /** Judgeable prose entries (outside the protected window). */
  prose: number;
  /** Estimated Jev requests (state + question batches). */
  requests: number;
  /** Estimated state tokens. */
  tokens: number;
  /** Why the session cannot be replayed, when it cannot. */
  error?: string;
}

/** Counts what a replay of this transcript would ask, without asking it. */
export function planReplay(messages: readonly Message[], options: CompactOptions = {}): SessionPlan {
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages, resolved.strip);
  const candidates = calls.filter((call) => !call.pinned);
  try {
    const fitted = fitState(messages, calls, resolved);
    const prose = fitted.state.history.filter((entry) => entry.text.trim().length > 0);
    const proseCandidates = prose.filter(
      (entry) => !isPinned(entry.i, messages.length, resolved.preserveRecentMessages),
    );
    const batches = batchCalls(candidates, fitted.tokens, resolved);
    const requests =
      batches.length +
      Math.ceil(proseCandidates.length / messageChunkSize(fitted.tokens, resolved.maxRequestTokens));
    if (candidates.length === 0 && proseCandidates.length === 0) {
      return { messages: messages.length, calls: 0, prose: 0, requests: 0, tokens: fitted.tokens };
    }
    return {
      messages: messages.length,
      calls: candidates.length,
      prose: proseCandidates.length,
      requests,
      tokens: fitted.tokens,
    };
  } catch (error) {
    return {
      messages: messages.length,
      calls: candidates.length,
      prose: 0,
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

export const OLLAMA_SYSTEM_ONE_URL = 'http://localhost:11434/v1/systemone';
export const OLLAMA_MODEL = 'clef-flash';

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
  /** Disable the identifier stripping replay enables by default. */
  noStrip?: boolean;
  /** File with blocklist terms, one per line. */
  blocklistPath?: string;
  /** `ollama` (default) targets a local Ollama decision model — nothing leaves the machine; `cloud` is opt-in and needs an API key. */
  asker?: 'cloud' | 'ollama';
  /** Full System One endpoint for `--asker ollama`. */
  baseUrl?: string;
  /** Model name for `--asker ollama` (e.g. `clef-flash`). */
  model?: string;
  /** Overrides the state token ceiling (default 25000; shrink for small-context models). */
  maxStateTokens?: number;
  /** Re-attempt sessions ledgered "too large" by splitting them at user prompts (default on; `--no-segment` disables). */
  segment?: boolean;
}

/** Parses replay CLI flags; throws on anything unknown or malformed. */
export function parseReplayArgs(argv: readonly string[]): ReplayArgs {
  const args: ReplayArgs = { dryRun: false, source: 'both', asker: 'ollama', segment: true };
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
    else if (arg === '--no-strip') args.noStrip = true;
    else if (arg === '--blocklist') args.blocklistPath = value(arg, argv[++i]);
    else if (arg === '--asker') {
      const kind = value(arg, argv[++i]);
      if (kind !== 'cloud' && kind !== 'ollama') {
        throw new Error(`--asker must be cloud or ollama, got "${kind}"`);
      }
      args.asker = kind;
    } else if (arg === '--base-url') args.baseUrl = value(arg, argv[++i]);
    else if (arg === '--model') args.model = value(arg, argv[++i]);
    else if (arg === '--max-state-tokens') args.maxStateTokens = number(arg, argv[++i]);
    else if (arg === '--no-segment') args.segment = false;
    else if (arg === '--segment') args.segment = true;
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
  /** Library options applied to every replayed compaction (e.g. `strip`). */
  compactOptions?: CompactOptions;
  /** Split sessions ledgered "too large" into fitting windows at user prompts. */
  segmentOversized?: boolean;
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
export interface ReplayRunSummary {
  sessions: number;
  replayed: number;
  records: number;
  skippedDone: number;
  noCandidates: number;
  errors: number;
  /** True when the run stopped early after repeated Jev failures. */
  stoppedAfterFailures: boolean;
}

/**
 * Deterministic judge failures: a fixed outcome at fixed options, safe to
 * ledger permanently. The 400 phrase comes from Ollama's systemone route
 * ("prompt N has X tokens; expected 1–Y (input is never truncated)"), the
 * 413 from its 64 KiB body cap ("text and schema must not exceed 64 KiB"),
 * and a 500 "llama-server process has terminated" is the backend
 * segfaulting (observed 0xc0000005) on specific states. Timeouts are
 * infra-bound (a slow GPU), not session-bound, so they stay transient and
 * retriable. Anything else (network, 5xx, write failures) is transient too.
 */
export function isDeterministicJevError(detail: string): boolean {
  return (
    /Jev request failed \(400\): .*input is never truncated/.test(detail) ||
    /Jev request failed \(413\): .*must not exceed 64 KiB/.test(detail) ||
    /Jev request failed \(500\): .*llama-server process has terminated/.test(detail)
  );
}

/** Observes the record append: a swallowed write failure must stay retriable. */
function makeObserving(trainingLog: (record: import('./types.js').TrainingRecord) => void | Promise<void>) {
  let emitted = false;
  let write: Promise<void> = Promise.resolve();
  const log: typeof trainingLog = (record) => {
    emitted = true;
    write = (async () => {
      await trainingLog(record);
    })();
    return write;
  };
  return {
    log,
    get emitted() {
      return emitted;
    },
    get write() {
      return write;
    },
  };
}

/** Jev calls failing back-to-back before the run gives up (likely an outage). */
const MAX_CONSECUTIVE_FAILURES = 5;

/** True when the ledger entry is an oversized-session error worth segmenting. */
export function isTooLargeError(entry: Pick<ReplayIndexEntry, 'error'> | undefined): boolean {
  // server-side caps (64 KiB body, model context) hit mid-run are the same condition as the client-side fitting failure
  return (
    !!entry?.error &&
    (/history too large for Jev/.test(entry.error) ||
      /must not exceed 64 KiB/.test(entry.error) ||
      /exceeds the model context/.test(entry.error))
  );
}

/** True when a ledgered error recovers on a re-run: too large (re-segments) or a timeout (infra-bound). */
export function isRetryableLedgerError(entry: Pick<ReplayIndexEntry, 'error'> | undefined): boolean {
  return isTooLargeError(entry) || (!!entry?.error && /Jev request timed out/.test(entry.error));
}

/** A message that starts one of the user's prompts: the natural cut point. */
function isUserPrompt(message: Message): boolean {
  return (
    message.role === 'user' &&
    message.text.trim().length > 0 &&
    !(message.toolResults ?? []).length
  );
}

function fitsWindow(
  messages: readonly Message[],
  resolved: ResolvedCompactOptions,
): boolean {
  try {
    fitState(messages, collectToolCalls(messages, resolved.preserveRecentMessages, resolved.strip), resolved);
    return true;
  } catch {
    return false;
  }
}

/**
 * Greedy segmentation: the fewest windows that each fit the state ceiling,
 * growing exponentially then binary-searching the largest fitting window,
 * with cuts snapped back to the nearest user-prompt start. Returns null when
 * even a single message cannot fit (unshrinkable oversized content).
 */
export function segmentByFit(
  messages: readonly Message[],
  resolved: ResolvedCompactOptions,
): Message[][] | null {
  const segments: Message[][] = [];
  let start = 0;
  while (start < messages.length) {
    if (!fitsWindow(messages.slice(start, start + 1), resolved)) return null;
    let hi = messages.length;
    let lo = start + 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi + 1) / 2);
      if (fitsWindow(messages.slice(start, mid), resolved)) lo = mid;
      else hi = mid - 1;
    }
    let end = lo;
    // snap the cut back so the next window starts at one of the user's prompts
    if (end < messages.length) {
      for (let c = end; c > start + 1; c--) {
        if (isUserPrompt(messages[c]!)) {
          end = c;
          break;
        }
      }
    }
    segments.push(messages.slice(start, end));
    start = end;
  }
  return segments;
}

/**
 * Replays every not-yet-ledgered source through `replaySession`, persisting
 * the resume ledger after each session so an interrupted run continues where
 * it stopped. Transient failures (session unreadable, Jev or record-write
 * failing) stay unledgered and retry on the next run; deterministic ones
 * (history unfittable) are ledgered as errors. `onSession` observes progress;
 * throws only on ledger corruption.
 */
export async function runReplay(
  sources: readonly ReplaySourceSpec[],
  asker: JevAsker,
  trainingLog: (record: import('./types.js').TrainingRecord) => void | Promise<void>,
  options: ReplayRunOptions,
  onSession?: (event: { key: string; outcome: 'replayed' | 'done' | 'no-candidates' | 'error' | 'stopped'; detail?: string }) => void,
): Promise<ReplayRunSummary> {
  const index = await loadReplayIndex(options.dir);
  const summary: ReplayRunSummary = {
    sessions: 0,
    replayed: 0,
    records: 0,
    skippedDone: 0,
    noCandidates: 0,
    errors: 0,
    stoppedAfterFailures: false,
  };
  let consecutiveFailures = 0;
  for (const source of sources) {
    if (options.limit !== undefined && summary.replayed + summary.errors >= options.limit) break;
    const prior = index[source.key];
    if (prior && !(options.segmentOversized && isRetryableLedgerError(prior))) {
      summary.skippedDone++;
      continue;
    }
    summary.sessions++;
    let ledgered = true;
    try {
      const messages = await source.load();
      const plan = planReplay(messages, options.compactOptions);
      const oversized = plan.error && isTooLargeError({ error: plan.error });
      if (oversized && options.segmentOversized) {
        // greedy windows that fit by construction. Windows smaller than the
        // recent window judge nothing, so a pass that completes with zero
        // records falls back to the minimum recent window before being
        // declared unshrinkable.
        const resolved = resolveOptions(options.compactOptions);
        const attempts: ResolvedCompactOptions[] = [resolved];
        if (resolved.preserveRecentMessages > 1) {
          attempts.push({ ...resolved, preserveRecentMessages: 1 });
        }
        let done: { records: number; segments: number } | null = null;
        let transientDetail: string | undefined;
        let errorDetail: string | undefined;
        for (let a = 0; a < attempts.length && done === null && !transientDetail && !errorDetail; a++) {
          const attempt = attempts[a]!;
          const segments = segmentByFit(messages, attempt);
          if (!segments) continue;
          const segmentOptions: CompactOptions | undefined =
            a === 0
              ? options.compactOptions
              : { ...(options.compactOptions ?? {}), preserveRecentMessages: 1 };
          let segmentRecords = 0;
          let segmentsDone = 0;
          for (const segment of segments) {
            const observing = makeObserving(trainingLog);
            try {
              await replaySession(segment, asker, observing.log, segmentOptions);
              await observing.write;
              if (!observing.emitted) {
                // a segment entirely inside its own recent window judges
                // nothing: a legitimate no-op, not a failure
                const segmentPlan = planReplay(segment, segmentOptions);
                if (segmentPlan.calls === 0 && segmentPlan.prose === 0) {
                  segmentsDone++;
                  continue;
                }
                throw new Error('no training record emitted');
              }
              segmentRecords++;
              segmentsDone++;
            } catch (error) {
              // a segment failing transiently retries the whole session next
              // run (and feeds the breaker); a deterministic judge failure
              // lands in the ledger
              const detail = error instanceof Error ? error.message : String(error);
              if (isDeterministicJevError(detail)) {
                errorDetail = `${detail} (segment ${segmentsDone + 1}/${segments.length})`;
              } else {
                transientDetail = detail;
              }
              break;
            }
          }
          if (transientDetail === undefined && errorDetail === undefined) {
            if (segmentsDone === segments.length && segmentRecords > 0) {
              done = { records: segmentRecords, segments: segments.length };
            }
            // a zero-record pass falls through to the next preserve variant
          }
        }
        if (done !== null) {
          index[source.key] = { at: options.now(), records: done.records };
          summary.replayed++;
          summary.records += done.records;
          consecutiveFailures = 0;
          onSession?.({
            key: source.key,
            outcome: 'replayed',
            detail: `${done.segments} segment(s), ${done.records} record(s) from an oversized history`,
          });
        } else if (transientDetail !== undefined) {
          ledgered = false;
          summary.errors++;
          consecutiveFailures++;
          onSession?.({ key: source.key, outcome: 'error', detail: transientDetail });
        } else {
          const detail = `${errorDetail ?? plan.error} (unshrinkable even at preserve=1)`;
          index[source.key] = { at: options.now(), error: detail };
          summary.errors++;
          onSession?.({ key: source.key, outcome: 'error', detail });
        }
      } else if (plan.error) {
        index[source.key] = { at: options.now(), error: plan.error };
        summary.errors++;
        onSession?.({ key: source.key, outcome: 'error', detail: plan.error });
      } else if (plan.calls === 0 && plan.prose === 0) {
        index[source.key] = { at: options.now(), records: 0 };
        summary.noCandidates++;
        onSession?.({ key: source.key, outcome: 'no-candidates' });
        consecutiveFailures = 0;
      } else {
        // observe the record append: a swallowed write failure must stay retriable
        const observing = makeObserving(trainingLog);
        try {
          await replaySession(messages, asker, observing.log, options.compactOptions);
          await observing.write;
          if (!observing.emitted) throw new Error('no training record emitted');
          index[source.key] = { at: options.now(), records: 1 };
          summary.replayed++;
          summary.records++;
          consecutiveFailures = 0;
          onSession?.({
            key: source.key,
            outcome: 'replayed',
            detail: `${plan.requests} request(s), ${plan.calls} calls, ${plan.prose} prose`,
          });
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          if (isDeterministicJevError(detail)) {
            // deterministic at fixed options: ledger like a plan error
            index[source.key] = { at: options.now(), error: detail };
            summary.errors++;
            onSession?.({ key: source.key, outcome: 'error', detail });
          } else {
            // network/write failures stay retriable: no ledger entry
            ledgered = false;
            summary.errors++;
            consecutiveFailures++;
            onSession?.({ key: source.key, outcome: 'error', detail });
          }
        }
      }
    } catch (error) {
      // a transient read failure (file locked, being written) must retry next run
      ledgered = false;
      summary.errors++;
      onSession?.({
        key: source.key,
        outcome: 'error',
        detail: error instanceof Error ? error.message : String(error),
      });
    }
    if (ledgered) await saveReplayIndex(options.dir, index);
    if (options.delayMs > 0) await options.sleep(options.delayMs);
    if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      summary.stoppedAfterFailures = true;
      onSession?.({
        key: '',
        outcome: 'stopped',
        detail: `${MAX_CONSECUTIVE_FAILURES} consecutive Jev failures; stopping (re-run to resume)`,
      });
      break;
    }
  }
  return summary;
}
