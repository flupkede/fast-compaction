import { noulAnswer } from './request.js';
import { createStripper } from './strip.js';
import {
  collectToolCalls,
  estimateTokens,
  fitState,
  goalFromMessages,
  isPinned,
  sliceRanges,
  stateEnvelopeTokens,
  worseStage,
} from './state.js';
import type {
  CallAnswer,
  CallDecision,
  CompactOptions,
  CompactResult,
  CompactionState,
  FittedState,
  HistoryEntry,
  JevAsker,
  JevQuestions,
  JudgeTiming,
  Message,
  MessageDecision,
  ResolvedCompactOptions,
  ToolCall,
  ToolUse,
  TrainingRecord,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  preserveStartMessages: 2,
  maxStateTokens: 25_000,
  // at least ~2k below the shim's num_ctx (32768): prompt plus generated
  // answer tokens must fit the judge's context — 26000 leaves that headroom
  maxRequestTokens: 26_000,
  maxQuestionsPerRequest: 12,
  askConcurrency: 2,
  duplicateKeepLast: 2,
  truncateHeadChars: 300,
  trainingSource: 'live',
};

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  const target = finite(options.targetTokens, Number.NaN);
  const targetTokens = Number.isFinite(target) ? Math.max(1, Math.floor(target)) : undefined;
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    maxQuestionsPerRequest: Math.max(
      1,
      Math.min(
        64,
        Math.floor(
          finite(options.maxQuestionsPerRequest, DEFAULT_OPTIONS.maxQuestionsPerRequest),
        ),
      ),
    ),
    askConcurrency: Math.max(
      1,
      Math.floor(finite(options.askConcurrency, DEFAULT_OPTIONS.askConcurrency)),
    ),
    duplicateKeepLast: Math.max(
      0,
      Math.floor(finite(options.duplicateKeepLast, DEFAULT_OPTIONS.duplicateKeepLast)),
    ),
    timeBudgetMs:
      typeof options.timeBudgetMs === 'number' && Number.isFinite(options.timeBudgetMs) && options.timeBudgetMs >= 0
        ? options.timeBudgetMs
        : undefined,
    tolerateRequestFailures: options.tolerateRequestFailures === true,
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    targetTokens,
    // the start window only matters under budget pressure; 0 keeps
    // threshold-only compaction identical to before
    preserveStartMessages:
      targetTokens === undefined
        ? 0
        : Math.max(
            0,
            Math.floor(
              finite(options.preserveStartMessages, DEFAULT_OPTIONS.preserveStartMessages),
            ),
          ),
    trainingLog: options.trainingLog,
    trainingSource: options.trainingSource ?? 'live',
    strip: options.strip ? createStripper(options.strip) : undefined,
  };
}

/** The two `noul` questions asked about one call: keep the call, keep its result. */
export function questionsFor(call: ToolCall): JevQuestions {
  return {
    [`call_${call.id}`]: {
      type: 'noul',
      instructions: `Tool call ${call.id} (${call.tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
    },
    [`result_${call.id}`]: {
      type: 'noul',
      instructions: `The full output of tool call ${call.id} (${call.tool}, ${call.resultChars} chars) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
    },
  };
}

/** The one `noul` question asked about a conversation-message history entry. */
export function questionForEntry(entry: HistoryEntry): JevQuestions {
  return {
    [`msg_${entry.i}`]: {
      type: 'noul',
      instructions: `The ${entry.role} message at history position ${entry.i} should stay in the compacted history: its text still matters for understanding or continuing the work, and later turns still refer to it`,
    },
  };
}

/**
 * Splits the candidate calls into batches whose questions, together with the
 * (always complete) state, fit one request. Each call carries two questions
 * (`keepCall` + `keepResult`), so the call cap is half
 * `maxQuestionsPerRequest`.
 */
export function batchCalls(
  calls: readonly ToolCall[],
  stateTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxRequestTokens' | 'maxQuestionsPerRequest'>,
): ToolCall[][] {
  const budget = options.maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  const callCap = Math.max(1, Math.floor(options.maxQuestionsPerRequest / 2));
  const batches: ToolCall[][] = [];
  let current: ToolCall[] = [];
  let currentTokens = 0;
  for (const call of calls) {
    const tokens = estimateTokens(JSON.stringify(questionsFor(call)));
    if (current.length > 0 && (currentTokens + tokens > budget || current.length >= callCap)) {
      batches.push(current);
      current = [];
      currentTokens = 0;
    }
    if (current.length === 0 && tokens > budget) {
      throw new Error(
        `state leaves no room for questions (~${stateTokens} of ${options.maxRequestTokens} tokens)`,
      );
    }
    current.push(call);
    currentTokens += tokens;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

/** The empty fitted state used when no request was made. */
const EMPTY_FITTED: FittedState = {
  state: { context: '', goal: '', history: [] },
  tokens: 0,
  stage: '',
};

/** Stable JSON for a tool input: object keys sorted recursively, so two
 * structurally identical inputs produce the same string. */
function stableInput(input: Record<string, unknown>): string {
  const stable = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(stable);
    if (value && typeof value === 'object') {
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(value as Record<string, unknown>).sort()) {
        out[key] = stable((value as Record<string, unknown>)[key]);
      }
      return out;
    }
    return value;
  };
  try {
    return JSON.stringify(stable(input));
  } catch {
    return String(input);
  }
}

/**
 * Ids of calls that repeat an earlier identical call (same tool, same
 * normalised input): only the newest `duplicateKeepLast` occurrences of each
 * distinct call survive, everything older is mechanically droppable — a
 * repeat adds nothing its kept newest results do not already hold (the flood
 * class: hundreds of identical paginated browser/search calls). Pinned calls
 * are never marked, but they do count as occurrences, so the kept-newest
 * window stays honest. 0 disables.
 */
export function duplicateCallIds(calls: readonly ToolCall[], keepLast: number): Set<string> {
  const duplicates = new Set<string>();
  if (keepLast <= 0) return duplicates;
  const keyOf = (call: ToolCall): string => `${call.tool}\u0000${stableInput(call.input)}`;
  const totals = new Map<string, number>();
  for (const call of calls) {
    const key = keyOf(call);
    totals.set(key, (totals.get(key) ?? 0) + 1);
  }
  const nth = new Map<string, number>();
  for (const call of calls) {
    const key = keyOf(call);
    const n = (nth.get(key) ?? 0) + 1;
    nth.set(key, n);
    if (!call.pinned && n <= (totals.get(key) ?? 0) - keepLast) duplicates.add(call.id);
  }
  return duplicates;
}

/** Question-token cost of one `msg_<i>` question (constant instructions). */
const MESSAGE_QUESTION_TOKENS = 64;

/** Message batch size bounded by the question cap and the request budget. */
export function messageChunkSize(
  stateTokens: number,
  maxRequestTokens: number,
  maxQuestionsPerRequest = 12,
): number {
  const room = maxRequestTokens - stateTokens - REQUEST_OVERHEAD_TOKENS;
  return Math.max(1, Math.min(maxQuestionsPerRequest, Math.floor(room / MESSAGE_QUESTION_TOKENS)));
}

function chunk<T>(items: readonly T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += size) chunks.push(items.slice(i, i + size));
  return chunks;
}

export function decideCall(
  call: Pick<ToolCall, 'id' | 'tool' | 'pinned'>,
  answer: CallAnswer,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): CallDecision {
  const base = { id: call.id, tool: call.tool, ...answer };
  if (call.pinned) return { ...base, action: 'keep', reason: 'pinned' };
  if (answer.keepResult >= options.keepThreshold) {
    return { ...base, action: 'keep', reason: 'kept' };
  }
  if (answer.keepCall >= options.keepThreshold) {
    return { ...base, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, action: 'drop_call', reason: 'call_dropped' };
}

/**
 * Folds one message judgement into a decision. `undefined` means the entry was
 * never asked about (protected window, or user prose — task instructions are
 * never dropped) and always stays; `null` means it was asked about but the
 * judge gave no answer — kept, conservatively.
 */
export function decideMessage(
  entry: Pick<HistoryEntry, 'i' | 'role' | 'text'>,
  keepProbability: number | null | undefined,
  options: Pick<ResolvedCompactOptions, 'keepThreshold'>,
): MessageDecision {
  const base = { i: entry.i, role: entry.role, chars: entry.text.length };
  if (keepProbability === undefined) {
    return { ...base, keep: true, reason: entry.role === 'user' ? 'user' : 'recent' };
  }
  if (entry.role === 'user') {
    // defence in depth: user prose is never judgeable, so this only fires on
    // a stale answer for an entry that left the candidate set
    return { ...base, keep: true, reason: 'user' };
  }
  if (keepProbability === null) {
    return { ...base, keep: true, reason: 'kept' };
  }
  const keep = keepProbability >= options.keepThreshold;
  return { ...base, keep, reason: keep ? 'kept' : 'dropped' };
}

function sumJudge(total: CompactResult['stats']['judge'], timing: JudgeTiming | undefined): CompactResult['stats']['judge'] {
  if (!timing) return total;
  const next = total ?? { requests: 0, loadMs: 0, promptMs: 0, promptTokens: 0, genMs: 0, genTokens: 0 };
  return {
    requests: next.requests + 1,
    loadMs: next.loadMs + (timing.loadMs ?? 0),
    promptMs: next.promptMs + (timing.promptMs ?? 0),
    promptTokens: next.promptTokens + (timing.promptTokens ?? 0),
    genMs: next.genMs + (timing.genMs ?? 0),
    genTokens: next.genTokens + (timing.genTokens ?? 0),
  };
}

/** Runs `fn` over `items` with at most `limit` in flight. Submissions happen
 * only when a slot frees, so a request never queues behind another — the
 * host's hard 30s per-request fetch abort counts wall time from submission.
 * Limit 2 matches ollama's parallel slots: higher just queues server-side. */
async function pooled<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const i = next++;
        results[i] = await fn(items[i]!);
      }
    }),
  );
  return results;
}

async function askBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly ToolCall[],
  judge: JudgeTiming[] = [],
): Promise<Map<string, CallAnswer>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionsFor));
  const response = await asker.ask(state, questions);
  judge.push(response.judge ?? {});
  const { answers } = response;
  return new Map(
    batch.map((call) => [
      call.id,
      {
        keepCall: noulAnswer(answers, `call_${call.id}`),
        keepResult: noulAnswer(answers, `result_${call.id}`),
      },
    ]),
  );
}

/** One noul answer per `msg_<i>` question in the batch. */
async function askMessageBatch(
  asker: JevAsker,
  state: CompactionState,
  batch: readonly HistoryEntry[],
  judge: JudgeTiming[] = [],
): Promise<Map<number, number>> {
  const questions: JevQuestions = Object.assign({}, ...batch.map(questionForEntry));
  const response = await asker.ask(state, questions);
  judge.push(response.judge ?? {});
  const { answers } = response;
  return new Map(
    batch.flatMap((entry) => {
      const answer = answers[`msg_${entry.i}`];
      const noul = answer && 'noul' in answer && typeof answer.noul === 'number' ? answer.noul : undefined;
      return noul === undefined ? [] : [[entry.i, noul] as const];
    }),
  );
}

function truncatedResultText(text: string, isError: boolean, headChars: number): string {
  if (text.length <= headChars + 120) return text;
  const head = headChars > 0 ? `${text.slice(0, headChars)}\n` : '';
  return `${head}[fast-compaction truncated ${text.length - headChars} chars of this tool result${
    isError ? ' (error)' : ''
  }; re-run the tool if needed]`;
}

/**
 * Rebuilds the conversation from the decisions. A dropped call disappears
 * together with its result; a dropped result keeps a bounded head and note.
 * Messages that lose all their content are removed; untouched messages are
 * returned as the same objects they came in as.
 */
export function applyDecisions(
  messages: readonly Message[],
  decisions: readonly CallDecision[],
  calls: readonly ToolCall[],
  headChars: number,
  messageDecisions: readonly MessageDecision[] = [],
): Message[] {
  const byId = new Map(calls.map((call) => [call.id, call]));
  const actions = new Map<string, CallDecision['action']>();
  for (const decision of decisions) {
    const call = byId.get(decision.id);
    if (call && decision.action !== 'keep') actions.set(call.tool_use_id, decision.action);
  }
  const droppedMessages = new Map<number, true>(
    messageDecisions.filter((d) => !d.keep).map((d) => [d.i, true]),
  );
  const kept: Message[] = [];
  for (const [index, message] of messages.entries()) {
    const proseDropped = droppedMessages.has(index);
    const text = proseDropped ? '' : message.text;
    const touched =
      proseDropped ||
      message.toolUses.some((tool) => actions.has(tool.tool_use_id)) ||
      (message.toolResults ?? []).some((result) => actions.has(result.tool_use_id));
    if (!touched) {
      kept.push(message);
      continue;
    }
    const toolUses = message.toolUses
      .filter((tool) => actions.get(tool.tool_use_id) !== 'drop_call')
      .map((tool) => {
        if (actions.get(tool.tool_use_id) !== 'drop_result') return tool;
        const resultText = truncatedResultText(
          tool.text ?? '',
          tool.isError ?? false,
          headChars,
        );
        if ((tool.text ?? '') === resultText) return tool;
        const copy: ToolUse = {
          tool_use_id: tool.tool_use_id,
          tool: tool.tool,
          input: tool.input,
          text: resultText,
        };
        if (tool.isError) copy.isError = true;
        return copy;
      });
    const toolResults = (message.toolResults ?? [])
      .filter((result) => actions.get(result.tool_use_id) !== 'drop_call')
      .map((result) => {
        if (actions.get(result.tool_use_id) !== 'drop_result') return result;
        const text = truncatedResultText(result.text, result.isError ?? false, headChars);
        return text === result.text
          ? result
          : {
              tool_use_id: result.tool_use_id,
              text,
              isError: result.isError,
            };
      });
    if (
      !proseDropped &&
      !message.toolUses.some(
        (tool) => actions.get(tool.tool_use_id) === 'drop_call',
      ) &&
      !(message.toolResults ?? []).some(
        (result) => actions.get(result.tool_use_id) === 'drop_call',
      ) &&
      toolUses.every((tool, index) => tool === message.toolUses[index]) &&
      toolResults.every(
        (result, index) => result === message.toolResults?.[index],
      )
    ) {
      kept.push(message);
      continue;
    }
    if (text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) {
      continue;
    }
    const rebuilt: Message = { role: message.role, text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  }
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

/**
 * Estimated transcript tokens: text, tool inputs and tool results, measured
 * with the same heuristic as the Jev state. The tool result carries the
 * outcome, so the mirrored tool-use text is not counted twice.
 */
export function transcriptTokens(messages: readonly Message[]): number {
  let text = '';
  for (const message of messages) {
    text += message.text;
    for (const tool of message.toolUses) {
      try {
        text += JSON.stringify(tool.input);
      } catch {
        text += ' ';
      }
    }
    for (const result of message.toolResults ?? []) text += result.text;
  }
  return estimateTokens(text);
}

/**
 * Demotes threshold-kept entries until the compacted transcript fits
 * `targetTokens`: kept results first, then whole calls, then kept prose —
 * lowest Jev score first, oldest on ties, protected windows never touched.
 * Best-effort: stops when nothing evictable remains. Returns the number of
 * demotions applied.
 */
function applyBudget(
  messages: readonly Message[],
  decisions: CallDecision[],
  messageDecisions: MessageDecision[],
  calls: readonly ToolCall[],
  messageAnswers: ReadonlyMap<number, number>,
  targetTokens: number,
  headChars: number,
): number {
  const over = (): boolean =>
    transcriptTokens(applyDecisions(messages, decisions, calls, headChars, messageDecisions)) >
    targetTokens;
  const ranked = <T>(
    items: readonly T[],
    score: (item: T) => number,
    order: (item: T) => number,
  ): T[] =>
    items
      .map((item) => ({ item, score: score(item) }))
      .sort((a, b) => a.score - b.score || order(a.item) - order(b.item))
      .map(({ item }) => item);
  const results = ranked(
    decisions.flatMap((decision, index) =>
      decision.action === 'keep' && decision.reason !== 'pinned' ? [index] : [],
    ),
    (index) => decisions[index]!.keepResult,
    (index) => index,
  );
  const callsLeft = ranked(
    decisions.flatMap((decision, index) =>
      decision.reason !== 'pinned' && decision.action !== 'drop_call' ? [index] : [],
    ),
    (index) => decisions[index]!.keepCall,
    (index) => index,
  );
  // never-answered prose was never scored: it ranks below every judged entry;
  // user prose (reason 'user') is exempt — task instructions survive the budget
  const prose = ranked(
    messageDecisions.flatMap((decision, index) =>
      decision.keep && decision.reason !== 'recent' && decision.reason !== 'user' ? [index] : [],
    ),
    (index) => messageAnswers.get(messageDecisions[index]!.i) ?? 0,
    (index) => messageDecisions[index]!.i,
  );

  let evictions = 0;
  for (const index of results) {
    if (!over()) return evictions;
    decisions[index] = { ...decisions[index]!, action: 'drop_result', reason: 'budget' };
    evictions += 1;
  }
  for (const index of callsLeft) {
    if (!over()) return evictions;
    decisions[index] = { ...decisions[index]!, action: 'drop_call', reason: 'budget' };
    evictions += 1;
  }
  for (const index of prose) {
    if (!over()) return evictions;
    messageDecisions[index] = { ...messageDecisions[index]!, keep: false, reason: 'budget' };
    evictions += 1;
  }
  return evictions;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

function count(decisions: readonly CallDecision[], reason: CallDecision['reason']): number {
  return decisions.filter((decision) => decision.reason === reason).length;
}

/** A batch of one call must always fit beside the state, whatever the caller
 * tuned: the state fit is additionally capped by the request budget minus a
 * minimal question reserve, so a fat state can never starve every batch
 * (observed live: state 19258 against maxRequestTokens 19000). */
const QUESTION_RESERVE_TOKENS = 320;

function stateFitOptions(resolved: ResolvedCompactOptions): ResolvedCompactOptions {
  const ceiling = Math.max(
    1_000,
    Math.min(
      resolved.maxStateTokens,
      resolved.maxRequestTokens - REQUEST_OVERHEAD_TOKENS - QUESTION_RESERVE_TOKENS,
    ),
  );
  return ceiling === resolved.maxStateTokens ? resolved : { ...resolved, maxStateTokens: ceiling };
}

/** Request-level judge failures are transient (network, 5xx, the engine's 30s
 * abort); parse-level failures mean the judge spoke nonsense and must throw. */
const REQUEST_FAILURE = /judge request failed|fetch failed|aborted|timed? ?out/i;

function isRequestFailure(error: unknown): boolean {
  return error instanceof Error && REQUEST_FAILURE.test(error.message);
}

/**
 * Compacts a transcript by asking Jev, for every tool call outside the pinned
 * first and newest messages, whether the call and whether its result must
 * stay. The whole history (results omitted, fitted into `maxStateTokens`) is
 * sent as state with every batch of questions. Throws when Jev fails or the
 * history cannot be fitted; the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(
    messages,
    resolved.preserveRecentMessages,
    resolved.strip,
    resolved.preserveStartMessages,
  );
  const candidates = calls.filter((call) => !call.pinned);
  // mechanical pre-filter: identical repeats never reach the judge
  const duplicates = duplicateCallIds(calls, resolved.duplicateKeepLast);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  let fitted: FittedState | undefined;
  let callJudge: JudgeTiming[] = [];
  const judgeTimings: JudgeTiming[] = [];
  let partial = false;
  let batchErrors = 0;
  let sliceCount = 0;
  // run-wide aggregates: the largest state sent and the worst collapse stage
  // any slice needed (single-slice runs report that slice, as before)
  let runStateTokens = 0;
  let runStage = '';
  const answers = new Map<string, CallAnswer>();
  const messageAnswers = new Map<number, number>();
  const allProse: HistoryEntry[] = [];
  const allMessageCandidates: HistoryEntry[] = [];
  const protectedIndex = (i: number): boolean =>
    isPinned(i, messages.length, resolved.preserveRecentMessages, resolved.preserveStartMessages);
  // piece-wise compaction: stop asking in time, keep the unanswered, let the
  // next compaction judge the remainder (a virgin fat window cannot fit the
  // host's hook budget in one pass)
  const deadline = resolved.timeBudgetMs === undefined ? undefined : started + resolved.timeBudgetMs;
  const outOfTime = (): boolean => deadline !== undefined && Date.now() >= deadline;

  // slice-wise judging: the unpinned middle is partitioned into ranges that
  // each fit the state ceiling by faithful (un-collapsed) entry size, so the
  // judge scores real content instead of `[… N chars omitted …]` placeholders
  // (a collapsed state made it score everything droppable)
  const goal = resolved.goal || goalFromMessages(messages, resolved.strip);
  const sliceStart = Math.max(1, resolved.preserveStartMessages);
  const sliceEnd = Math.max(sliceStart, messages.length - resolved.preserveRecentMessages);
  const chunkOptions = {
    ...stateFitOptions(resolved),
    goal,
    // the slice range is the protection; the fitter must be able to shrink
    // freely inside a chunk and must never see global window sizes
    preserveRecentMessages: 0,
    preserveStartMessages: 0,
  };
  const ranges = sliceRanges(
    messages,
    calls,
    // the per-slice budget must subtract the envelope (context + goal) the
    // fitter always adds on top, or slices that "fit" by entry sum still
    // collapse inside fitState
    Math.max(500, chunkOptions.maxStateTokens - stateEnvelopeTokens(goal)),
    sliceStart,
    sliceEnd,
    resolved.strip,
  );
  sliceCount = ranges.length;
  for (const range of ranges) {
    if (outOfTime()) {
      partial = true;
      break;
    }
    const chunkMessages = messages.slice(range.start, range.end);
    // calls whose result lands beyond the range stay unjudged (conservative)
    const chunkCalls = calls.filter(
      (call) => call.callIndex >= range.start && call.callIndex < range.end && call.resultIndex < range.end,
    );
    const state = fitState(chunkMessages, chunkCalls, chunkOptions);
    // fitState indexes within the chunk; shift to global message indices
    state.state.history = state.state.history.map((entry) => ({
      ...entry,
      i: entry.i + range.start,
    }));
    fitted = fitted ?? state;
    runStateTokens = Math.max(runStateTokens, state.tokens);
    runStage = worseStage(runStage, state.stage);
    const sliceCandidates = chunkCalls.filter(
      (call) => !call.pinned && !duplicates.has(call.id),
    );
    const sliceBatches = batchCalls(sliceCandidates, state.tokens, resolved);
    const answeredCall = await pooled(
      sliceBatches,
      resolved.askConcurrency,
      (batch) => {
        if (outOfTime()) {
          partial = true;
          return Promise.resolve(new Map<string, CallAnswer>());
        }
        // one dead batch must not kill the pass: its calls stay (conservative)
        return askBatch(asker, state.state, batch, judgeTimings).catch((error: unknown) => {
          if (!resolved.tolerateRequestFailures || !isRequestFailure(error)) throw error;
          partial = true;
          batchErrors += 1;
          return new Map<string, CallAnswer>();
        });
      },
    );
    for (const map of answeredCall) {
      for (const [id, answer] of map) answers.set(id as string, answer as CallAnswer);
    }
    // prose entries present in the fitted state, outside the protected window;
    // user prose carries the task instructions: never judged, never dropped
    const proseEntries = state.state.history.filter((entry) => entry.text.trim().length > 0);
    const messageCandidates = proseEntries.filter(
      (entry) => entry.role !== 'user' && !protectedIndex(entry.i),
    );
    allProse.push(...proseEntries);
    allMessageCandidates.push(...messageCandidates);
    if (outOfTime()) {
      partial = true;
      break;
    }
    const messageChunks = chunk(
      messageCandidates,
      messageChunkSize(state.tokens, resolved.maxRequestTokens, resolved.maxQuestionsPerRequest),
    );
    const answeredMessages = await pooled(
      messageChunks,
      resolved.askConcurrency,
      (batch) => {
        if (outOfTime()) {
          partial = true;
          return Promise.resolve(new Map<number, number>());
        }
        return askMessageBatch(asker, state.state, batch, judgeTimings).catch((error: unknown) => {
          if (!resolved.tolerateRequestFailures || !isRequestFailure(error)) throw error;
          partial = true;
          batchErrors += 1;
          return new Map<number, number>();
        });
      },
    );
    for (const map of answeredMessages) {
      for (const [i, noul] of map) messageAnswers.set(i, noul);
    }
  }
  const proseEntries = allProse;
  const messageCandidates = allMessageCandidates;
  callJudge = judgeTimings;

  const decisions = calls.map((call) =>
    decideCall(call, answers.get(call.id) ?? { keepCall: 1, keepResult: 1 }, resolved),
  );
  // duplicates were never asked: record the mechanical verdict (synthetic
  // 0.0 answers, like pinned carries synthetic 1.0); decisions is parallel
  // to calls (built with calls.map)
  for (const [index, call] of calls.entries()) {
    if (!duplicates.has(call.id)) continue;
    decisions[index] = {
      id: call.id,
      tool: call.tool,
      keepCall: 0,
      keepResult: 0,
      action: 'drop_call',
      reason: 'duplicate',
    };
  }
  const messageDecisions = proseEntries.map((entry) =>
    decideMessage(
      entry,
      messageCandidates.some((c) => c.i === entry.i)
        ? (messageAnswers.get(entry.i) ?? null)
        : undefined,
      resolved,
    ),
  );
  const budgetEvictions =
    resolved.targetTokens === undefined
      ? 0
      : applyBudget(
          messages,
          decisions,
          messageDecisions,
          calls,
          messageAnswers,
          resolved.targetTokens,
          resolved.truncateHeadChars,
        );
  const kept = applyDecisions(
    messages,
    decisions,
    calls,
    resolved.truncateHeadChars,
    messageDecisions,
  );

  if (resolved.trainingLog && (candidates.length > 0 || messageCandidates.length > 0)) {
    const questions: JevQuestions = Object.assign({}, ...candidates.map(questionsFor));
    const messageQuestions: JevQuestions = Object.assign(
      {},
      ...messageCandidates.map(questionForEntry),
    );
    const record: TrainingRecord = {
      schema: 2,
      ts: new Date().toISOString(),
      source: resolved.trainingSource,
      state: (fitted ?? EMPTY_FITTED).state,
      stateTokens: runStateTokens,
      stateStage: runStage,
      questions,
      answers: Object.fromEntries(answers),
      decisions,
      keepThreshold: resolved.keepThreshold,
      messageQuestions,
      messageAnswers: Object.fromEntries(
        [...messageAnswers].map(([i, noul]) => [`msg_${i}`, noul]),
      ),
      messageDecisions,
    };
    try {
      await resolved.trainingLog(record);
    } catch {
      // logging must never break compaction
    }
  }

  let judge: CompactResult['stats']['judge'];
  for (const timing of callJudge) judge = sumJudge(judge, timing);

  return {
    messages: kept,
    decisions,
    messageDecisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count(decisions, 'kept'),
      resultsDropped: count(decisions, 'result_dropped'),
      callsDropped: count(decisions, 'call_dropped'),
      pinned: count(decisions, 'pinned'),
      /** Prose entries judged droppable and removed. */
      messagesDropped: messageDecisions.filter((d) => d.reason === 'dropped').length,
      duplicates: duplicates.size,
      budgetEvictions,
      stateTokens: runStateTokens,
      stateStage: runStage,
      // actually asked, not planned: a time budget can skip planned batches
      requests: callJudge.length,
      ms: Date.now() - started,
      ...(sliceCount > 1 ? { slices: sliceCount } : {}),
      ...(partial ? { partial: true } : {}),
      ...(batchErrors > 0 ? { batchErrors } : {}),
      judge,
    },
  };
}
