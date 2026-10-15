# fast-compaction

Repo instruction doc. Vendor-named variants (`CLAUDE.md`) mean the same.

## Current Plan

Objective: compaction targets an explicit token ceiling — the hook derives `targetTokens = context.window × floorPercent/100`, floor starts at 50% of the window, steps down 5pp after every accepted Jev compaction, minimum 25%; after any built-in-summary fallback the floor resets to 50%. The library demotes budget-wise AFTER Jev's threshold verdicts: middle-zone tool results → tool calls → prose; start window (first 2 messages) and recent window (6) stay protected. Agent compactions without usage() get no budget mode (current threshold-only behavior).

### Library (src/)

1. `src/types.ts`: add options `targetTokens?: number` (absolute estimated-token ceiling for the compacted transcript; undefined = feature off, behavior identical to today) and `preserveStartMessages?: number` (default 2) to `CompactOptions` and `ResolvedCompactOptions`. Add reason `'budget'` to the `CallDecision.reason` union ('pinned'|'kept'|'result_dropped'|'call_dropped'|'budget') and the `MessageDecision.reason` union ('kept'|'dropped'|'recent'|'budget'). Add `budgetEvictions: number` to the inline stats object in `CompactResult`.
2. `src/state.ts`: `isPinned(index, total, preserveRecentMessages, preserveStartMessages)` — pin the start window as well: index < preserveStartMessages, or index === 0 (as now), or index >= total - preserveRecentMessages. Update ALL call sites: `collectToolCalls` (state.ts:91-92), the `pinned` closure in `fitState` (state.ts:254-255), `src/compact.ts:367` (`protectedIndex`), `src/replay.ts:91`. `fitState`'s Pick<> parameter type must gain `preserveStartMessages`. Other fitState callers pass resolved/spread option objects so they flow through — verify.
3. `src/compact.ts`: `DEFAULT_OPTIONS` gains `preserveStartMessages: 2`; `resolveOptions` clamps it (>=0, floor). New budget phase AFTER the threshold decisions are computed (currently lines ~396-407) and BEFORE `applyDecisions`, active only when `targetTokens` is set:
   - Size measurement: estimate transcript tokens with the existing `estimateTokens` heuristic (serialize each message's text + toolUse inputs + toolResult texts; reuse `messageChars`-style traversal).
   - Demotion ladder, middle zone only — never demote decisions with reason 'pinned' nor messageDecisions with reason 'recent' (the start window already becomes 'recent'/pinned through the new isPinned):
     (a) calls with action 'keep' → 'drop_result', reason 'budget' (result truncated to head, exactly like threshold-driven drop_result);
     (b) calls still 'keep' or 'drop_result' → 'drop_call', reason 'budget';
     (c) prose: messageDecisions with keep:true → keep:false, reason 'budget'. Entries that were NEVER judged because fitState omitted them from the state (messageDecisions where the keep probability is undefined but NOT inside the protected window — reason currently 'kept') drop FIRST within this step (score 0); then ascending judged noul.
     Order within steps (a) and (b): ascending Jev score (keepResult for (a); keepCall for (b)); tie-break oldest first (by call/message index).
   - Recompute estimated size after each demotion; stop as soon as size <= targetTokens or nothing evictable remains (best-effort).
   - `stats.budgetEvictions` = number of demotions applied by the budget phase (calls + prose).
   - WITHOUT `targetTokens` (undefined): decisions, reasons and stats must be byte-identical to today — existing tests must pass unmodified.
4. Implementation note: the current per-item threshold logic (`decideCall`, `decideMessage`) stays as-is; the budget phase only demotes its output. `applyDecisions` needs no change — it already acts on actions/keeps.

### Hook (hooks/)

5. `hooks/stats.ts`: add `floorPercent?: number` to `SessionStats`; default in `emptySession`; `reviveSession` currently spreads defaults over stored (top-level only) — set/normalize the new field explicitly so stale stored values without it get the default.
6. `hooks/fast-compaction.ts`: `HOOK_DEFAULTS` gains `targetFloorStartPercent: 50`, `targetFloorMinPercent: 25`, `targetFloorStepPercent: 5` (+ read via the existing numeric-keys loop in `resolveHookConfig`, plus `preserveStartMessages` pass-through). Extend `HookConfig`. Export a pure helper, e.g. `nextFloorPercent(current: number | undefined, outcome: 'jev' | 'fallback-low' | 'fallback-error', config): number` — 'jev' → max(min, current - step) (undefined current → start); any fallback → start.
7. `session.compact` handler: where `tokensBefore` is read, also read `usage.context.window` (the type exposes `window: number` directly — no tokens/percent math). Compute `targetTokens = window × floorPercent / 100` using the session's current floor (read from the session's SessionStats via the store, default = start). Pass `targetTokens` + `preserveStartMessages` into the compact options. If usage is unavailable (agent compactions or usage() throws): omit `targetTokens` entirely — threshold-only behavior. Escalation via the existing `withStats` serialization: after a 'jev' outcome persist the stepped-down floor; after 'fallback-low'/'fallback-error' persist the reset floor. Accept/reject rule unchanged: `ratio < minReductionRatio` → built-in summary fallback. Mention budget evictions in the toast/notify `summarize()` output only if trivial (e.g. append `N budget` to the parts list when > 0).
8. `.claude-plugin/plugin.json` `userConfig` block: declare the new numeric options `targetFloorStartPercent`, `targetFloorMinPercent`, `targetFloorStepPercent`, `preserveStartMessages` following the existing entries' style. Document them in `hooks/README.md` configuration table and the pass-through sentence. In root `README.md`: add `targetTokens` and `preserveStartMessages` to the library options table, and fix the Limitations bullet claiming text messages are never removed (under budget pressure they can be dropped, tool material first).

### Tests

9. `tests/fast-compaction.test.ts` — reuse the `fakeJev(answer, seen)` pattern and transcript helpers: (a) with a tight `targetTokens`, results demote before calls before prose, lowest score first, start+recent untouched; (b) never-judged prose drops before judged prose; (c) best-effort: target below what full eviction achieves → everything evictable goes, pinned stays; (d) no targetTokens → identical decisions to current expectations (existing tests already cover this — they must pass unmodified); (e) stats.budgetEvictions counts.
10. `tests/hook.test.ts` — unit tests for `nextFloorPercent` (step-down, clamp at min, reset on fallback, undefined → start).
11. Validate: `npm run typecheck` then `npm test`; run them with output redirected to a file and read the verdict + verbatim errors only.

### Constraints

- Comments only for non-obvious WHY; no history or "per review" comments. All code/comments/commits in English. No new .md files beyond the AGENTS.md plan placement. No hardcoded magic numbers in the hook — the floor constants live in HOOK_DEFAULTS; library default `preserveStartMessages` in DEFAULT_OPTIONS.
- Verified integration facts: `$.session.usage()` returns `SessionUsage` with `context.window`; `session.compact` event input carries no usage; `$.store` is JSON-only, one `session:<id>` key per session written via `updateSessionStats`/`statsChain`. Reason/action literal consumers outside src/compact.ts: hooks/fast-compaction.ts `decisionLog` filters 'pinned' (budget-reasoned decisions simply appear, that is fine), tests/fast-compaction.test.ts, tests/hook.test.ts, tests/training-log.test.ts — update only where behavior actually changes under the new options; without targetTokens nothing changes.
- Commit messages: concise subject + why; stage by logical unit (library, hook, docs/tests may be separate commits).
