export type TokenUsage = {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
};

export type UsageTotals = {
  turns: number;
  input: number;
  cacheRead: number;
  cacheWrite: number;
  output: number;
};

export type CompactionOutcome = 'judge' | 'fallback-low' | 'fallback-error';

/** Judge-reported timing sums for one compaction (from a shim in front of ollama). */
export type JudgeSplit = {
  loadMs: number;
  promptMs: number;
  promptTokens: number;
  genMs: number;
  genTokens: number;
};

export type CompactionRecord = {
  at: number;
  outcome: CompactionOutcome;
  tokensBefore?: number;
  tokensDropped?: number;
  ratio?: number;
  ms?: number;
  /** System One requests the compaction made. */
  requests?: number;
  /** Judge timing when the endpoint reported it (summed over the compaction's requests). */
  judge?: JudgeSplit;
  reason?: string;
  /** A subagent's own transcript: counted, but it never moves the main loop's credited drop. */
  agent?: boolean;
};

export type SessionStats = {
  id: string;
  startedAt: number;
  lastAt: number;
  main: UsageTotals;
  agents: UsageTotals;
  costUsd: number;
  compactions: Record<CompactionOutcome, number>;
  judgeMs: number;
  /** Judge timing summed over every judge compaction that reported it. */
  judgeRequests: number;
  judgeLoadMs: number;
  judgePromptMs: number;
  judgePromptTokens: number;
  judgeGenMs: number;
  judgeGenTokens: number;
  droppedTokens: number;
  savedTokens: number;
  /** Tokens the latest judge compaction removed; each later main-loop API call is credited with them until the next compaction. */
  activeDrop: number;
  recent: CompactionRecord[];
  /** Percentage of the context window the next compaction targets; undefined = no budget compaction yet, the configured start floor applies. */
  floorPercent?: number;
  /** Diagnostics: the context reading at the last turn.complete, and the last compaction attempt's outcome. */
  lastPercent?: number;
  lastTokens?: number;
  lastAttempt?: string;
  /** Anti-treadmill guard: ms epoch of the last compaction attempt, and the
   * lowest context reading since it (host units — system prompt + tool
   * schemas included, which the plugin never sees in its own accounting). */
  lastCompactAt?: number;
  minTokensSinceCompact?: number;
};

export const STORE_PREFIX = 'session:';
export const RECENT_PER_SESSION = 20;
export const MAX_SESSIONS = 200;
export const REASON_MAX_CHARS = 200;
/** Anthropic prompt-cache pricing relative to a base input token. */
export const CACHE_READ_PRICE = 0.1;
export const CACHE_WRITE_PRICE = 1.25;

/** Input-token equivalents at cache pricing: what the input side actually costs. */
export function weightedInput(u: UsageTotals): number {
  return u.input + u.cacheRead * CACHE_READ_PRICE + u.cacheWrite * CACHE_WRITE_PRICE;
}

/**
 * Saved tokens priced as they would have been sent: at the main loop's cache-read
 * share they were cheap cache reads, the rest full-price input.
 */
export function weightedSaved(saved: number, main: UsageTotals): number {
  const sent = main.input + main.cacheRead + main.cacheWrite;
  const readShare = sent > 0 ? main.cacheRead / sent : 0;
  return saved * (readShare * CACHE_READ_PRICE + (1 - readShare));
}

function emptyUsage(): UsageTotals {
  return { turns: 0, input: 0, cacheRead: 0, cacheWrite: 0, output: 0 };
}

export function emptySession(id: string, now: number): SessionStats {
  return {
    id,
    startedAt: now,
    lastAt: now,
    main: emptyUsage(),
    agents: emptyUsage(),
    costUsd: 0,
    compactions: { judge: 0, 'fallback-low': 0, 'fallback-error': 0 },
    judgeMs: 0,
    judgeRequests: 0,
    judgeLoadMs: 0,
    judgePromptMs: 0,
    judgePromptTokens: 0,
    judgeGenMs: 0,
    judgeGenTokens: 0,
    droppedTokens: 0,
    savedTokens: 0,
    activeDrop: 0,
    recent: [],
    floorPercent: undefined,
  };
}

/** Rebuilds a stored value, tolerating fields an older version did not write. */
export function reviveSession(id: string, value: unknown, now: number): SessionStats {
  const base = emptySession(id, now);
  if (!value || typeof value !== 'object') return base;
  const stored = value as Partial<SessionStats> & Record<string, unknown>;
  // legacy names from before the judge rename (jevMs, compactions.jev, …)
  const legacy = stored as Record<string, number | undefined>;
  const timing = {
    judgeMs: stored.judgeMs ?? legacy.jevMs ?? base.judgeMs,
    judgeRequests: stored.judgeRequests ?? legacy.jevRequests ?? base.judgeRequests,
    judgeLoadMs: stored.judgeLoadMs ?? legacy.jevLoadMs ?? base.judgeLoadMs,
    judgePromptMs: stored.judgePromptMs ?? legacy.jevPromptMs ?? base.judgePromptMs,
    judgePromptTokens: stored.judgePromptTokens ?? legacy.jevPromptTokens ?? base.judgePromptTokens,
    judgeGenMs: stored.judgeGenMs ?? legacy.jevGenMs ?? base.judgeGenMs,
    judgeGenTokens: stored.judgeGenTokens ?? legacy.jevGenTokens ?? base.judgeGenTokens,
  };
  const rawCompactions = (stored.compactions ?? {}) as Record<string, number | undefined>;
  const compactions = {
    ...base.compactions,
    ...rawCompactions,
    judge: (rawCompactions.judge ?? 0) + (rawCompactions.jev ?? 0),
  } as SessionStats['compactions'];
  return {
    ...base,
    ...stored,
    ...timing,
    id,
    main: { ...base.main, ...stored.main },
    agents: { ...base.agents, ...stored.agents },
    compactions,
    recent: Array.isArray(stored.recent)
      ? stored.recent
          .map((r) => {
            const record = r as CompactionRecord;
            // legacy outcome name from before the judge rename
            return (record as { outcome?: unknown }).outcome === 'jev'
              ? { ...record, outcome: 'judge' as const }
              : record;
          })
          .filter(
            (r): r is CompactionRecord =>
              !!r && typeof r.at === 'number' && typeof r.outcome === 'string' && Object.hasOwn(base.compactions, r.outcome),
          )
      : [],
    // the spread above copies whatever an older store wrote; normalize explicitly
    floorPercent:
      typeof stored.floorPercent === 'number' && Number.isFinite(stored.floorPercent)
        ? stored.floorPercent
        : undefined,
  };
}

function addUsage(totals: UsageTotals, usage: TokenUsage): void {
  totals.turns += 1;
  totals.input += usage.input_tokens;
  totals.cacheRead += usage.cache_read_input_tokens;
  totals.cacheWrite += usage.cache_creation_input_tokens;
  totals.output += usage.output_tokens;
}

/**
 * A turn's usage sums every API response in it; dividing its input side by the
 * last response's input gives roughly how many calls re-sent the context.
 */
export function estimateCalls(usage: TokenUsage, contextTokens: number | undefined): number {
  const sent = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens;
  if (!contextTokens || contextTokens <= 0 || sent <= 0) return 1;
  return Math.max(1, Math.round(sent / contextTokens));
}

export type TurnSample = {
  at: number;
  usage?: TokenUsage;
  isAgent: boolean;
  contextTokens?: number;
  costUsd?: number;
};

export function recordTurn(stats: SessionStats, turn: TurnSample): SessionStats {
  stats.lastAt = turn.at;
  if (turn.costUsd !== undefined && turn.costUsd > stats.costUsd) stats.costUsd = turn.costUsd;
  if (!turn.usage) return stats;
  addUsage(turn.isAgent ? stats.agents : stats.main, turn.usage);
  if (!turn.isAgent && stats.activeDrop > 0) {
    stats.savedTokens += stats.activeDrop * estimateCalls(turn.usage, turn.contextTokens);
  }
  return stats;
}

export function recordCompaction(stats: SessionStats, record: CompactionRecord): SessionStats {
  stats.lastAt = record.at;
  // any attempt (whatever the outcome) stamps the cooldown brake and opens a
  // new low-water cycle: the context it leaves behind is the new floor
  stats.lastCompactAt = record.at;
  stats.minTokensSinceCompact = undefined;
  stats.compactions[record.outcome] += 1;
  if (record.outcome === 'judge') {
    stats.judgeMs += record.ms ?? 0;
    stats.judgeRequests += record.requests ?? 0;
    stats.judgeLoadMs += record.judge?.loadMs ?? 0;
    stats.judgePromptMs += record.judge?.promptMs ?? 0;
    stats.judgePromptTokens += record.judge?.promptTokens ?? 0;
    stats.judgeGenMs += record.judge?.genMs ?? 0;
    stats.judgeGenTokens += record.judge?.genTokens ?? 0;
  }
  if (!record.agent) {
    const dropped = record.outcome === 'judge' ? (record.tokensDropped ?? 0) : 0;
    stats.droppedTokens += dropped;
    stats.activeDrop = dropped;
  }
  const reason =
    record.reason === undefined ? undefined : record.reason.replace(/\s+/g, ' ').trim().slice(0, REASON_MAX_CHARS);
  stats.recent = [...stats.recent, { ...record, reason }].slice(-RECENT_PER_SESSION);
  return stats;
}

/** Tokens a judge compaction removed: the context before times the reduction it measured on characters. */
export function droppedTokens(tokensBefore: number | undefined, ratio: number): number | undefined {
  if (tokensBefore === undefined || !Number.isFinite(ratio)) return undefined;
  return Math.max(0, Math.round(tokensBefore * ratio));
}

/** Store keys to delete so that at most `max` sessions stay, the least recently active first. */
export function evictable(sessions: readonly SessionStats[], max: number = MAX_SESSIONS): string[] {
  if (sessions.length <= max) return [];
  return [...sessions]
    .sort((a, b) => a.lastAt - b.lastAt)
    .slice(0, sessions.length - max)
    .map((s) => STORE_PREFIX + s.id);
}

function sumUsage(a: UsageTotals, b: UsageTotals): UsageTotals {
  return {
    turns: a.turns + b.turns,
    input: a.input + b.input,
    cacheRead: a.cacheRead + b.cacheRead,
    cacheWrite: a.cacheWrite + b.cacheWrite,
    output: a.output + b.output,
  };
}

export function aggregate(sessions: readonly SessionStats[]): SessionStats {
  const total = emptySession('all', sessions.reduce((m, s) => Math.min(m, s.startedAt), Infinity));
  for (const s of sessions) {
    total.lastAt = Math.max(total.lastAt, s.lastAt);
    total.main = sumUsage(total.main, s.main);
    total.agents = sumUsage(total.agents, s.agents);
    total.costUsd += s.costUsd;
    for (const key of Object.keys(total.compactions) as CompactionOutcome[]) {
      total.compactions[key] += s.compactions[key];
    }
    total.judgeMs += s.judgeMs;
    total.judgeRequests += s.judgeRequests;
    total.judgeLoadMs += s.judgeLoadMs;
    total.judgePromptMs += s.judgePromptMs;
    total.judgePromptTokens += s.judgePromptTokens;
    total.judgeGenMs += s.judgeGenMs;
    total.judgeGenTokens += s.judgeGenTokens;
    total.droppedTokens += s.droppedTokens;
    total.savedTokens += s.savedTokens;
    total.recent.push(...s.recent);
  }
  total.recent.sort((a, b) => a.at - b.at);
  return total;
}

export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}K`;
  return String(Math.round(n));
}

function stamp(ms: number): string {
  return new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
}

function recordLine(r: CompactionRecord): string {
  const size =
    r.tokensBefore !== undefined && r.tokensDropped !== undefined
      ? `${formatTokens(r.tokensBefore)} → ${formatTokens(r.tokensBefore - r.tokensDropped)}`
      : '';
  const ratio = r.ratio !== undefined ? `${Math.round(r.ratio * 100)}%` : '';
  const ms = r.ms !== undefined ? `${r.ms} ms` : '';
  const req = r.requests !== undefined ? `${r.requests} req` : '';
  return [stamp(r.at), r.outcome.padEnd(14), size, ratio, ms, req, r.reason ?? ''].filter(Boolean).join('  ');
}

/** `avg load + prefill (tokens) + gen (tokens)` when any judge timing was reported. */
function timingLine(stats: SessionStats, judgeCount: number): string | undefined {
  const hasJudge =
    stats.judgePromptMs > 0 || stats.judgeGenMs > 0 || stats.judgeLoadMs > 0 || stats.judgeRequests > 0;
  if (!hasJudge || judgeCount === 0) return undefined;
  const parts = [
    `load ${Math.round(stats.judgeLoadMs / judgeCount)} ms`,
    `prefill ${Math.round(stats.judgePromptMs / judgeCount)} ms (${formatTokens(stats.judgePromptTokens / judgeCount)} tok)`,
    `gen ${Math.round(stats.judgeGenMs / judgeCount)} ms (${formatTokens(stats.judgeGenTokens / judgeCount)} tok)`,
  ];
  if (stats.judgeRequests > 0) parts.push(`${(stats.judgeRequests / judgeCount).toFixed(1)} req/compaction`);
  return `Timing       avg per judge compaction: ${parts.join(' + ')}`;
}

export function formatReport(stats: SessionStats, scope: string, sessionCount: number, recent = 10): string {
  const c = stats.compactions;
  const u = sumUsage(stats.main, stats.agents);
  const sent = u.input + u.cacheRead + u.cacheWrite;
  const hit = sent > 0 ? `${((u.cacheRead / sent) * 100).toFixed(1)}%` : 'n/a';
  const avgMs = c.judge > 0 ? Math.round(stats.judgeMs / c.judge) : 0;
  const avgContext = u.turns > 0 ? formatTokens(sent / u.turns) : 'n/a';
  const fallbacks = c['fallback-low'] + c['fallback-error'];
  const lines = [
    `fast-compaction: ${scope}, ${sessionCount} session(s) since ${
      sessionCount > 0 ? stamp(stats.startedAt) : 'n/a'
    }`,
    `Compactions  ${c.judge} by judge${c.judge > 0 ? ` · judge avg ${avgMs} ms` : ''} · ${
      fallbacks > 0 ? `${fallbacks} fallback (${c['fallback-low']} low reduction, ${c['fallback-error']} error)` : 'no fallbacks'
    }`,
    `Saved        ~${formatTokens(weightedSaved(stats.savedTokens, stats.main))} input-eq. tokens ≈ $${(weightedSaved(stats.savedTokens, stats.main) / 1e6 * 3).toFixed(2)} at $3/M input (estimate)`,
    `Cost         $${stats.costUsd.toFixed(2)} reported across ${sessionCount} session(s) · ${u.turns} turns · avg ${avgContext} input tokens/turn (incl. cache) · cache hit ${hit} of input`,
  ];
  const timing = timingLine(stats, c.judge);
  if (timing) lines.push(timing);
  const last = stats.recent.slice(-recent);
  if (last.length > 0) lines.push('', 'Recent', ...last.map(recordLine));
  return lines.join('\n');
}
