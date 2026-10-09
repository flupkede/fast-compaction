import type { StripConfig, Stripper } from './strip.js';

export type Role = 'user' | 'assistant';

/**
 * A tool_use block of an assistant message. `text` and `isError` mirror the
 * outcome once the transcript holds it (Claude Code attaches them).
 */
export interface ToolUse {
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  text?: string;
  isError?: boolean;
}

/** A tool_result block of a user message. */
export interface ToolResult {
  tool_use_id: string;
  text: string;
  isError?: boolean;
}

/**
 * One transcript message. The shape is a subset of Claude Code's
 * `SessionMessage`, so a session transcript can be passed in as is.
 */
export interface Message {
  role: Role;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
}

/** A tool call paired with its result by `tool_use_id`. */
export interface ToolCall {
  /** Short id used in the Jev state and question names (`t1`, `t2`, ...). */
  id: string;
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  /** Index of the message holding the tool_use block. */
  callIndex: number;
  /** Index of the message holding the tool_result block. */
  resultIndex: number;
  resultChars: number;
  isError: boolean;
  /** In the first or the newest preserved messages; never a candidate. */
  pinned: boolean;
}

export interface CallAnswer {
  /** Jev's probability that the call itself still matters. */
  keepCall: number;
  /** Jev's probability that the full result still needs to stay verbatim. */
  keepResult: number;
}

export type CallAction = 'keep' | 'drop_result' | 'drop_call';

export interface CallDecision extends CallAnswer {
  id: string;
  tool: string;
  action: CallAction;
  reason: 'pinned' | 'kept' | 'result_dropped' | 'call_dropped';
}

export interface HistoryToolCall {
  id: string;
  tool: string;
  input: string;
  result: string;
}

export interface HistoryEntry {
  i: number;
  role: Role;
  text: string;
  /** Structured per call, or one compact line per call once the state has to shrink. */
  tool_calls?: HistoryToolCall[] | string[];
}

/** The state sent with every Jev request: the whole history, results omitted. */
export interface CompactionState {
  context: string;
  goal: string;
  history: HistoryEntry[];
}

export interface FittedState {
  state: CompactionState;
  tokens: number;
  /** Which fitting stage produced the state, for diagnostics. */
  stage: string;
}

export interface CompactOptions {
  /** Ongoing task description; defaults to the last few user prompts. */
  goal?: string;
  /** Minimum keep probability for a call or result to stay. Default 0.5. */
  keepThreshold?: number;
  /** Newest messages never touched (the first message is always kept). Default 6. */
  preserveRecentMessages?: number;
  /** Estimated token ceiling for the state. Default 25000. */
  maxStateTokens?: number;
  /** Estimated token ceiling for state plus one batch of questions. Default 30000. */
  maxRequestTokens?: number;
  /** Characters of a dropped tool result to retain. Default 300. */
  truncateHeadChars?: number;
  /**
   * Receives one record whenever Jev was asked (a compaction with no
   * candidate calls asks nothing and logs nothing): redacted state,
   * questions, Jev's answers, the applied decisions, for local
   * training-data collection. A logger that throws is ignored: logging
   * never breaks compaction.
   */
  trainingLog?: (record: TrainingRecord) => void | Promise<void>;
  /** Tag on the record's `source`; `'live'` unless a replay marks `'replay'`. */
  trainingSource?: 'live' | 'replay';
  /**
   * Identifier pseudonymisation (paths, URLs, e-mails, GUIDs, IPs, domains,
   * blocklist) applied after secret redaction, before anything is sent.
   * Off by default for live compaction; the replay CLI turns it on.
   */
  strip?: StripConfig;
}

export interface ResolvedCompactOptions {
  goal: string;
  keepThreshold: number;
  preserveRecentMessages: number;
  maxStateTokens: number;
  maxRequestTokens: number;
  truncateHeadChars: number;
  trainingLog?: (record: TrainingRecord) => void | Promise<void>;
  trainingSource: 'live' | 'replay';
  strip?: Stripper;
}

export interface CompactResult {
  /** The compacted transcript; untouched messages are the input objects. */
  messages: Message[];
  decisions: CallDecision[];
  messageDecisions: MessageDecision[];
  stats: {
    messagesBefore: number;
    messagesAfter: number;
    charsBefore: number;
    charsAfter: number;
    calls: number;
    kept: number;
    resultsDropped: number;
    callsDropped: number;
    pinned: number;
    /** Prose entries judged droppable and removed. */
    messagesDropped: number;
    stateTokens: number;
    /** Which fitting stage the state needed, '' when no request was made. */
    stateStage: string;
    requests: number;
    ms: number;
  };
}

/**
 * One labelled compaction for future model training. `state` is the exact
 * (already redacted, fitted) payload Jev saw; `answers` holds the real
 * probabilities asked for the candidate calls only; `decisions` is what was
 * applied (pinned entries carry synthetic 1.0 answers there).
 *
 * Schema 2 adds conversation-message judging: `messageQuestions`/`messageDecisions`
 * cover the prose history entries the model judged alongside the calls. Records
 * with `schema: 1` predate it and carry call decisions only.
 */
export interface TrainingRecord {
  schema: 1 | 2;
  ts: string;
  /** Where the compaction ran: live in a client hook, or replayed from a stored transcript. */
  source: 'live' | 'replay';
  state: CompactionState;
  stateTokens: number;
  stateStage: string;
  questions: JevQuestions;
  answers: Record<string, CallAnswer>;
  decisions: CallDecision[];
  keepThreshold: number;
  /** Per-entry noul questions asked about conversation messages (schema 2). */
  messageQuestions?: JevQuestions;
  /** What was applied to the conversation messages (schema 2). */
  messageDecisions?: MessageDecision[];
}

/** The judgement applied to one conversation-message prose entry. */
export interface MessageDecision {
  /** Message index the history entry was built from (`HistoryEntry.i`). */
  i: number;
  role: 'user' | 'assistant';
  /** Characters of prose the entry carried when judged. */
  chars: number;
  keep: boolean;
  reason: 'kept' | 'dropped' | 'recent';
}

/** The `state` of a Jev request: a string or any JSON-serialisable object. */
export type JevState = string | object;

export interface NoulQuestion {
  type: 'noul';
  instructions: string;
  criteria?: {
    true?: string;
    false?: string;
  };
}

export interface ChoiceQuestion {
  type: 'choice';
  instructions: string;
  criteria: Record<string, string | null>;
}

export interface ScoreQuestion {
  type: 'score';
  instructions: string;
  criteria: string[];
}

export type JevQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion;
export type JevQuestions = Record<string, JevQuestion>;

export interface NoulAnswer {
  type?: 'noul';
  noul: number;
}

export interface ChoiceAnswer {
  type?: 'choice';
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type?: 'score';
  score: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export type JevAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface JevResponse {
  model?: string;
  answers: Record<string, JevAnswer>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  [key: string]: unknown;
}

/** Anything that can answer Jev questions: `JevClient`, or a host-provided adapter. */
export interface JevAsker {
  ask(state: JevState, questions: JevQuestions): Promise<JevResponse>;
}
