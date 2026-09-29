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

export type CompactionOutcome = 'jev' | 'fallback-low' | 'fallback-error';

export type CompactionRecord = {
  at: number;
  outcome: CompactionOutcome;
  tokensBefore?: number;
  tokensDropped?: number;
  ratio?: number;
  ms?: number;
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
  jevMs: number;
  droppedTokens: number;
  savedTokens: number;
  /** Tokens the latest Jev compaction removed; each later main-loop API call is credited with them until the next compaction. */
  activeDrop: number;
  recent: CompactionRecord[];
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
    compactions: { jev: 0, 'fallback-low': 0, 'fallback-error': 0 },
    jevMs: 0,
    droppedTokens: 0,
    savedTokens: 0,
    activeDrop: 0,
    recent: [],
  };
}

/** Rebuilds a stored value, tolerating fields an older version did not write. */
export function reviveSession(id: string, value: unknown, now: number): SessionStats {
  const base = emptySession(id, now);
  if (!value || typeof value !== 'object') return base;
  const stored = value as Partial<SessionStats>;
  return {
    ...base,
    ...stored,
    id,
    main: { ...base.main, ...stored.main },
    agents: { ...base.agents, ...stored.agents },
    compactions: { ...base.compactions, ...stored.compactions },
    recent: Array.isArray(stored.recent)
      ? stored.recent.filter(
          (r): r is CompactionRecord =>
            !!r && typeof r.at === 'number' && typeof r.outcome === 'string' && Object.hasOwn(base.compactions, r.outcome),
        )
      : [],
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
  stats.compactions[record.outcome] += 1;
  if (record.outcome === 'jev') stats.jevMs += record.ms ?? 0;
  if (!record.agent) {
    const dropped = record.outcome === 'jev' ? (record.tokensDropped ?? 0) : 0;
    stats.droppedTokens += dropped;
    stats.activeDrop = dropped;
  }
  const reason =
    record.reason === undefined ? undefined : record.reason.replace(/\s+/g, ' ').trim().slice(0, REASON_MAX_CHARS);
  stats.recent = [...stats.recent, { ...record, reason }].slice(-RECENT_PER_SESSION);
  return stats;
}

/** Tokens a Jev compaction removed: the context before times the reduction it measured on characters. */
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
    total.jevMs += s.jevMs;
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
  return [stamp(r.at), r.outcome.padEnd(14), size, ratio, ms, r.reason ?? ''].filter(Boolean).join('  ');
}

export function formatReport(stats: SessionStats, scope: string, sessionCount: number, recent = 5): string {
  const c = stats.compactions;
  const u = sumUsage(stats.main, stats.agents);
  const sent = u.input + u.cacheRead + u.cacheWrite;
  const hit = sent > 0 ? `${((u.cacheRead / sent) * 100).toFixed(1)}%` : 'n/a';
  const avgMs = c.jev > 0 ? Math.round(stats.jevMs / c.jev) : 0;
  const avgContext = u.turns > 0 ? formatTokens(sent / u.turns) : 'n/a';
  const lines = [
    `fast-jev-compaction stats: ${scope}, ${sessionCount} session(s) since ${
      sessionCount > 0 ? stamp(stats.startedAt) : 'n/a'
    }`,
    `Compactions  ${c.jev} jev · ${c['fallback-low']} fallback (low reduction) · ${c['fallback-error']} fallback (error)${
      c.jev > 0 ? ` · jev avg ${avgMs} ms` : ''
    }`,
    `Dropped      ${formatTokens(stats.droppedTokens)} tokens removed by Jev`,
    `Saved        ~${formatTokens(stats.savedTokens)} context tokens not re-sent · ~${formatTokens(
      weightedSaved(stats.savedTokens, stats.main),
    )} input-equivalent at cache pricing (estimate)`,
    `Weighted     ${formatTokens(weightedInput(u))} input-equivalent sent (cache read ${CACHE_READ_PRICE}x, write ${CACHE_WRITE_PRICE}x)`,
    `Turns        ${stats.main.turns} main · ${stats.agents.turns} subagent · avg ${avgContext} input per turn`,
    `Tokens       input ${formatTokens(u.input)} · cache read ${formatTokens(u.cacheRead)} · cache write ${formatTokens(
      u.cacheWrite,
    )} · output ${formatTokens(u.output)} · cache hit ${hit}`,
    `Cost         $${stats.costUsd.toFixed(2)} (Claude Code cost ledger)`,
  ];
  const last = stats.recent.slice(-recent);
  if (last.length > 0) lines.push('', `Last ${last.length} compaction(s)`, ...last.map(recordLine));
  return lines.join('\n');
}
