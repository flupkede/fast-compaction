import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import { TYPESAFE_API_KEY_ENV } from '../src/client.js';
import { createTrainingLogger } from '../src/training-log.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';
import {
  aggregate,
  droppedTokens,
  evictable,
  formatReport,
  recordCompaction,
  recordTurn,
  reviveSession,
  STORE_PREFIX,
  type CompactionOutcome,
  type CompactionRecord,
  type SessionStats,
  type TurnSample,
} from './stats.js';

export const STATS_COMMAND = 'jev-stats';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  targetFloorStartPercent: 50,
  targetFloorMinPercent: 25,
  targetFloorStepPercent: 5,
  model: DEFAULT_MODEL,
  jevTimeoutMs: 8_000,
};

/** How long after a turn the deferred compaction waits, and how often it retries. */
const DEFER_MS = 1_000;
const DEFER_ATTEMPTS = 5;

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = Omit<CompactOptions, 'trainingLog'> & {
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  targetFloorStartPercent: number;
  targetFloorMinPercent: number;
  targetFloorStepPercent: number;
  model: string;
  jevTimeoutMs: number;
  /**
   * Full System One endpoint to post to; when set, the request goes there
   * instead of the TypeSafe API and no API key is needed (e.g. a local
   * Ollama decision model at `http://host:11434/v1/systemone`).
   */
  baseUrl?: string;
  /** Log one training record per Jev compaction to `trainingDir` (or its default). */
  trainingLog: boolean;
  trainingDir?: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function optionBoolean(options: PluginOptions, key: string, fallback: boolean): boolean {
  const value = options[key];
  return typeof value === 'boolean' ? value : fallback;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal' | 'trainingLog'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'preserveStartMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    targetFloorStartPercent: optionNumber(
      options,
      'targetFloorStartPercent',
      HOOK_DEFAULTS.targetFloorStartPercent,
    ),
    targetFloorMinPercent: optionNumber(
      options,
      'targetFloorMinPercent',
      HOOK_DEFAULTS.targetFloorMinPercent,
    ),
    targetFloorStepPercent: optionNumber(
      options,
      'targetFloorStepPercent',
      HOOK_DEFAULTS.targetFloorStepPercent,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    jevTimeoutMs: optionNumber(options, 'jevTimeoutMs', HOOK_DEFAULTS.jevTimeoutMs),
    trainingLog: optionBoolean(options, 'trainingLog', false),
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  const trainingDir = optionString(options, 'trainingDir');
  if (trainingDir) config.trainingDir = trainingDir;
  const baseUrl = optionString(options, 'baseUrl');
  if (baseUrl) config.baseUrl = baseUrl;
  return config;
}

/**
 * The floor for the next compaction's token target: one step down after an
 * accepted Jev compaction, reset to the start after any built-in-summary
 * fallback.
 */
export function nextFloorPercent(
  current: number | undefined,
  outcome: CompactionOutcome,
  config: Pick<
    HookConfig,
    'targetFloorStartPercent' | 'targetFloorMinPercent' | 'targetFloorStepPercent'
  >,
): number {
  if (outcome !== 'jev') return config.targetFloorStartPercent;
  return Math.max(
    config.targetFloorMinPercent,
    (current ?? config.targetFloorStartPercent) - config.targetFloorStepPercent,
  );
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(
  fetchFn: HookFetch,
  apiKey: string,
  model: string,
  baseUrl?: string,
): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model, baseUrl }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when no judge is configured or it fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
): Promise<SessionCompaction> {
  if (!config.apiKey && !config.baseUrl) {
    throw new Error(
      `no judge configured: set ${TYPESAFE_API_KEY_ENV} for the cloud, or baseUrl for a local decision model`,
    );
  }
  const apiKey = config.apiKey ?? 'local';
  const { trainingLog, trainingDir, ...rest } = config;
  const options: CompactOptions = trainingLog
    ? { ...rest, trainingLog: createTrainingLogger(trainingDir ? { dir: trainingDir } : {}) }
    : rest;
  const result = await compact(
    messages,
    jevAsker(fetchFn, apiKey, config.model, config.baseUrl),
    options,
  );
  return { result, messages: toSessionMessages(messages, result.messages) };
}

/**
 * Rejects when `work` outlasts `ms`. `$.http.fetch` takes no signal, so the
 * requests keep running; the hook just stops waiting and falls back before
 * the engine's hook budget expires.
 */
export function withDeadline<T>(
  work: Promise<T>,
  ms: number,
  sleep: (ms: number) => Promise<void>,
): Promise<T> {
  const deadline = sleep(ms).then((): never => {
    throw new Error(`Jev did not answer within ${ms} ms`);
  });
  deadline.catch(() => {});
  return Promise.race([work, deadline]);
}

/**
 * Compacts once the turn has ended. `$.session.compact()` rejects while a turn
 * runs, and a `turn.complete` hook is still part of its turn, so it is never
 * awaited there; it retries a few times in case the next turn already started.
 */
export async function deferredCompact(
  compactFn: () => Promise<unknown>,
  sleep: (ms: number) => Promise<void>,
  log: (text: string) => void,
): Promise<void> {
  let last: unknown;
  for (let attempt = 0; attempt < DEFER_ATTEMPTS; attempt++) {
    await sleep(DEFER_MS);
    try {
      await compactFn();
      return;
    } catch (error) {
      last = error;
    }
  }
  log(`auto-compact skipped (${last instanceof Error ? last.message : String(last)})`);
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
    stats.budgetEvictions > 0 ? `${stats.budgetEvictions} budget` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s), ${stats.ms} ms`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

/** The slice of `$.store` the stats need, so they can be driven without an engine. */
export type StatsStore = {
  get: (key: string) => Promise<unknown>;
  set: (key: string, value: unknown) => Promise<void>;
  delete: (key: string) => Promise<void>;
  keys: () => Promise<string[]>;
};

/** Host-shaped holder: the plugin validator requires `$.store.*` spelled at each call site. */
export type StatsHostStore = { store: StatsStore };

export async function loadAllStats($: StatsHostStore, now: number): Promise<SessionStats[]> {
  const keys = (await $.store.keys()).filter((key) => key.startsWith(STORE_PREFIX));
  return Promise.all(
    keys.map(async (key) => reviveSession(key.slice(STORE_PREFIX.length), await $.store.get(key), now)),
  );
}

/** One key per session: concurrent sessions each write only their own. */
export async function updateSessionStats(
  $: StatsHostStore,
  id: string,
  now: number,
  update: (stats: SessionStats) => void,
): Promise<void> {
  const key = STORE_PREFIX + id;
  const stats = reviveSession(id, await $.store.get(key), now);
  update(stats);
  await $.store.set(key, stats);
}

/** The session's current compaction floor; sessions without one start at the configured start. */
async function currentFloorPercent($: StatsHost, config: HookConfig): Promise<number> {
  const [id, now] = await Promise.all([$.session.id(), $.clock.now()]);
  const stats = reviveSession(id, await $.store.get(STORE_PREFIX + id), now);
  return stats.floorPercent ?? config.targetFloorStartPercent;
}

export async function evictOldStats($: StatsHostStore, now: number): Promise<number> {
  const stale = evictable(await loadAllStats($, now));
  for (const key of stale) await $.store.delete(key);
  return stale.length;
}

/** `/jev-stats` (all sessions), `/jev-stats session` (this one), `/jev-stats reset`. */
export async function statsCommand(
  $: StatsHostStore,
  args: string,
  sessionId: string,
  now: number,
): Promise<string> {
  const arg = args.trim().toLowerCase();
  const all = await loadAllStats($, now);
  if (arg === 'reset') {
    for (const s of all) await $.store.delete(STORE_PREFIX + s.id);
    return `fast-compaction stats reset (${all.length} session(s) cleared).`;
  }
  if (arg === 'session') {
    const own = all.filter((s) => s.id === sessionId);
    return formatReport(aggregate(own), 'this session', own.length);
  }
  if (arg !== '') return `Unknown argument "${args.trim()}". Use /${STATS_COMMAND} [session|reset].`;
  return formatReport(aggregate(all), 'all sessions', all.length);
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get(TYPESAFE_API_KEY_ENV);
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)[TYPESAFE_API_KEY_ENV];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

type StatsHost = {
  store: StatsStore;
  session: { id: () => Promise<string> };
  clock: { now: () => Promise<number> };
  ui: { log: (text: string) => void };
};

// Serialises this process's read-modify-write of its session key.
let statsChain: Promise<void> = Promise.resolve();

function withStats($: StatsHost, update: (stats: SessionStats, now: number) => void): void {
  statsChain = statsChain
    .then(async () => {
      const [id, now] = await Promise.all([$.session.id(), $.clock.now()]);
      await updateSessionStats($, id, now, (stats) => update(stats, now));
    })
    .catch((error: unknown) => {
      $.ui.log(`stats not recorded (${error instanceof Error ? error.message : String(error)})`);
    });
}

function logCompaction($: StatsHost, record: Omit<CompactionRecord, 'at'>): void {
  withStats($, (stats, at) => void recordCompaction(stats, { ...record, at }));
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;

  on('session.start', async ($, event, next) => {
    try {
      await $.command.register({
        name: STATS_COMMAND,
        description: 'fast-compaction token and savings stats',
        argumentHint: '[session|reset]',
        immediate: true,
      });
      await evictOldStats($, await $.clock.now());
    } catch (error) {
      $.ui.log(`stats setup failed (${error instanceof Error ? error.message : String(error)})`);
    }
    return next(event);
  });

  on('command.run', { command: STATS_COMMAND }, async ($, event) => {
    await statsChain;
    const [id, now] = await Promise.all([$.session.id(), $.clock.now()]);
    return { text: await statsCommand($, event.args, id, now) };
  });

  on('session.compact', async ($, event, next) => {
    const agent = event.agentId !== undefined;
    let tokensBefore: number | undefined;
    let windowTokens: number | undefined;
    if (!agent) {
      try {
        const usage = await $.session.usage();
        tokensBefore = usage.context.tokens;
        windowTokens = usage.context.window;
      } catch {
        tokensBefore = undefined;
      }
    }
    // agents and usage-less sessions compact threshold-only: no ceiling
    let floor: number | undefined;
    let targetTokens: number | undefined;
    if (windowTokens !== undefined) {
      try {
        floor = await currentFloorPercent($, configured);
        targetTokens = Math.round((windowTokens * floor) / 100);
      } catch {
        floor = undefined;
        targetTokens = undefined;
      }
    }
    const persistFloor = (outcome: CompactionOutcome): void => {
      if (floor === undefined) return;
      withStats($, (stats) => {
        stats.floorPercent = nextFloorPercent(floor, outcome, configured);
      });
    };
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      if (targetTokens !== undefined) config.targetTokens = targetTokens;
      const { result, messages } = await withDeadline(
        compactSession(event.messages, config, async (url, init) => {
          const response = await $.http.fetch(url, init);
          return { status: response.status, ok: response.ok, text: response.text };
        }),
        config.jevTimeoutMs,
        (ms) => $.clock.sleep(ms, { signal: next.signal }),
      );
      for (const line of decisionLogLines(result)) $.ui.log(line);
      const ratio = reductionRatio(result);
      if (ratio < config.minReductionRatio) {
        notify(
          $,
          `fallback to built-in summary (below ${percent(config.minReductionRatio)} minimum: ${summarize(result)})`,
        );
        logCompaction($, {
          outcome: 'fallback-low',
          ratio,
          ms: result.stats.ms,
          reason: `below ${percent(config.minReductionRatio)}`,
          agent,
        });
        persistFloor('fallback-low');
        return next(event);
      }
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      const record: Omit<CompactionRecord, 'at'> = { outcome: 'jev', ratio, ms: result.stats.ms, agent };
      if (tokensBefore !== undefined) record.tokensBefore = tokensBefore;
      const dropped = droppedTokens(tokensBefore, ratio);
      if (dropped !== undefined) record.tokensDropped = dropped;
      logCompaction($, record);
      persistFloor('jev');
      return { messages };
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      notify($, `fallback to built-in summary (${reason})`);
      logCompaction($, { outcome: 'fallback-error', reason, agent });
      persistFloor('fallback-error');
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    const answer = await next(event);
    let context: { tokens?: number; percent?: number };
    try {
      const usage = await $.session.usage();
      context = usage.context;
      const sample: Omit<TurnSample, 'at'> = { isAgent: event.agentId !== undefined };
      if (event.usage) sample.usage = event.usage;
      if (context.tokens !== undefined) sample.contextTokens = context.tokens;
      if (usage.cost) sample.costUsd = usage.cost.usd;
      withStats($, (stats, at) => void recordTurn(stats, { ...sample, at }));
    } catch (error) {
      $.ui.log(`auto-compact skipped (${error instanceof Error ? error.message : String(error)})`);
      return answer;
    }
    if (compacting) return answer;
    if ((context.percent ?? 0) < configured.compactAtPercent) return answer;
    compacting = true;
    void deferredCompact(
      () => $.session.compact(),
      (ms) => $.clock.sleep(ms),
      (text) => $.ui.log(text),
    ).finally(() => {
      compacting = false;
    });
    return answer;
  });
};

export { resolveOptions };
