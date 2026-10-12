# fast-compaction

Claude Code plugin that replaces the compaction summary with Jev decisions:
every tool call and result is scored in one fast request, stale ones are
dropped or truncated, everything kept stays verbatim. Also usable as an npm
library.

## What and why

Most context compaction asks an LLM to summarize old turns. A summary is
lossy: a file path, exact error, constraint, or command can disappear even when
it matters later. This library never rewrites anything. It only deletes tool
calls and tool results Jev says are no longer needed, and it asks Jev while
showing it the whole conversation. User and assistant text stays verbatim and
in order.

The repository is both an npm package (`src/`) and a Claude Code plugin
(`hooks/`, `.claude-plugin/`) that uses the package to replace Claude Code's
built-in compaction summary with the original messages.

## Secret redaction (fork of tamaratran/fast-jev-compaction)

Only the state sent to Jev (`src/state.ts`) is affected — the transcript
`compact()` returns to the caller is untouched, so nothing kept locally is
ever redacted. Before message text or tool input is serialised into that
state, `src/redact.ts` replaces secret-shaped substrings (API keys, tokens,
passwords, private key blocks, `name=value`/`name: value` credential
assignments) with a `[REDACTED:label]` marker. Tool results were never sent
in the first place (only a `chars` count, see `resultNote` in `state.ts`).
This is pattern-based, not a guarantee: it catches recognisable secret
shapes, not arbitrary confidential text.

## How it works

1. Every `tool_use` is paired with its `tool_result` by `tool_use_id`. Calls in
   the first message or in the newest `preserveRecentMessages` messages are
   pinned and never touched.
2. The **state** sent to Jev is the whole conversation so far, oldest first,
   with every tool result replaced by a short note (`ok, 4213 chars (omitted)`).
   Tool inputs are included, texts are included, nothing is summarized.
3. The state is fitted into `maxStateTokens` (25k by default) in stages, each
   applied only if the previous one was not enough: tool inputs truncated to
   1000, then 200, then 60 characters; long texts abridged to head + tail,
   oldest non-pinned messages first; old non-pinned messages collapsed to a
   `[… N chars omitted …]` note; old tool calls reduced to one line each
   (`t12 Read file_path=src/a.ts → ok 480ch`); old call-less messages left
   out; runs of old call-only messages folded into one entry. If it still
   does not fit, compaction throws. Tokens are estimated without a tokenizer (a
   word per six letters, half a token per digit, ~one per other symbol),
   calibrated to land a little above the counts Jev reports.
4. For every non-pinned call Jev gets two `noul` questions: should the **call**
   stay (knowing it was made, with its input, still matters), and should the
   **result** stay verbatim (its contents are still needed and re-running the
   tool would not do).
5. Questions are split into as many requests as needed so state plus questions
   stays under `maxRequestTokens` (30k by default, under Jev's 32k request
   limit). The same full state is resent with every request; requests run
   concurrently and their answers are merged.
6. Decisions per call, against `keepThreshold`:
   - `keepResult ≥ threshold` → keep call and result;
   - else `keepCall ≥ threshold` → keep the call, truncate the result to its
     first `truncateHeadChars` characters plus a one-line note;
   - else → remove the call together with its result.
7. The message list is rebuilt: a message that loses all its content is
   removed, untouched messages are returned as the same objects, and no result
   is ever left without its call.

Jev failures, malformed answers, a missing key, or a history that cannot be
fitted throw; the caller (or the Claude Code hook) decides what to fall back to.

## Install and usage

```sh
npm install fast-compaction
export TYPESAFE_API_KEY=...
```

```ts
import { compactMessages, reductionRatio, type Message } from 'fast-compaction';

const transcript: Message[] = [
  { role: 'user', text: 'Fix the failing test. Never edit src/generated.', toolUses: [] },
  {
    role: 'assistant',
    text: '',
    toolUses: [{ tool_use_id: 'toolu_1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_1', text: '…file…' }] },
  // …
];

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session transcript
can be passed in as is.

To bring your own transport, implement `JevAsker` (one `ask(state, questions)`
method) and call `compact(messages, asker, options)`; `buildJevRequest` and
`parseJevResponse` give you the HTTP request body and response validation.
The building blocks (`collectToolCalls`, `fitState`, `batchCalls`,
`decideCall`, `applyDecisions`) are exported too.

`apiKey` defaults to `process.env.TYPESAFE_API_KEY`. Never commit the key or
put it in a source file.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `apiKey` | `TYPESAFE_API_KEY` | TypeSafe API key (`compactMessages`/`JevClient`) |
| `model` | `jev-latest` | Jev model name |
| `baseUrl` | `https://api.typesafe.ai/v1/systemone` | System One endpoint |
| `fetch` | native `fetch` | Injectable fetch implementation for tests |
| `goal` | last 3 user prompts | Ongoing task description included in the state |
| `keepThreshold` | `0.5` | Minimum keep probability for a call or result to stay |
| `preserveRecentMessages` | `6` | Newest messages never touched (the first is always kept) |
| `maxStateTokens` | `25000` | Estimated token ceiling for the state |
| `maxRequestTokens` | `30000` | Estimated ceiling for state plus one batch of questions |
| `truncateHeadChars` | `300` | Characters of a dropped tool result retained before its note |
| `trainingLog` | off | Callback receiving one `TrainingRecord` whenever Jev was asked (redacted state, questions, Jev's answers, decisions; schema 2 also judges conversation messages) for local training-data collection; a throwing logger is ignored. Message judging covers the prose entries that fit the state: on very large histories the late shrink stage leaves the oldest prose out of the state entirely, and those entries are never judged and survive compaction untouched |
| `strip` | off | Identifier pseudonymisation applied after secret redaction: paths, URLs, e-mails, GUIDs, IPs, domains and a `blocklist` become stable tokens (`[PATH_1]`, `[URL_2]`, …) before anything is sent |

`createTrainingLogger({ dir })` turns that callback into an append-only
JSONL writer (`compactions-YYYY-MM.jsonl`, default dir `~/.fast-compaction/training`);
the data stays on the local machine.

## Replaying stored sessions

Historical sessions carry no labels — Jev's judgment only exists where a
compaction actually ran. `scripts/replay-training.ts` replays stored Claude
Code (`~/.claude/projects/**/*.jsonl`) and OpenCode (`opencode.db`) sessions
through the live pipeline and writes one `source: "replay"` record per session
that asks Jev, using the same redaction as live compaction:

```sh
npm run replay -- --dry-run        # count sessions, candidate calls, ~requests
npm run replay                     # local Ollama judge by default; re-run to resume
npm run replay -- --asker cloud    # TypeSafe cloud instead (needs TYPESAFE_API_KEY)
```

**Privacy.** The default judge is local Ollama — nothing leaves the machine
and no API key is needed. Identifier stripping is ON by default for replay
(`--no-strip` disables): after secret redaction, paths, URLs, e-mails, GUIDs,
IPs, domains and any `--blocklist <file>` terms (one per line, `#` comments)
are replaced by stable tokens, so client names and repository layouts never
reach the API when you opt into `--asker cloud`. TypeSafe's privacy policy
states they "will not train or fine tune any artificial intelligence or
machine learning models on Input" and will not disclose it beyond service
providers, but retention is open-ended and US-hosted — stripping is the
client-side control, and the only fully safe option is not sending data at
all.

## Local judge (Ollama)

The judge defaults to a local Ollama decision model (e.g. Cloudflare's
`clef-flash`), so nothing ever leaves your machine and no API key is needed:

```sh
ollama pull clef-flash            # needs Ollama >= 0.35.1
npm run replay                     # judges locally by default
```

Local runs default to a 12000-token state ceiling because the endpoint caps
requests at 64 KiB; override with `--max-state-tokens`.

Set `OLLAMA_SYSTEM_ONE_URL` (and optionally `OLLAMA_MODEL`) to persist a LAN
endpoint without retyping the flags; the flags still override.

Remote endpoints take over via `--asker cloud` (plus `--base-url` /
`--model` for other System One-compatible servers). The hook takes the same
choice through its `baseUrl` plugin option (plus `model: clef-flash` and a
raised `jevTimeoutMs` — local inference is slower than the cloud API and the
hook falls back to the built-in summary on timeout). The library layer is
unchanged: `new JevClient({ baseUrl, apiKey: 'ollama', model })`.

Progress is ledgered in `replay-index.json` inside the training dir, so an
interrupted run continues instead of duplicating records; sessions that failed
against the judge stay unledgered and retry. Sessions ledgered "history too
large" are split into fitting windows at user prompts and replayed per window
by default; `--no-segment` disables that. `--source claude|opencode|both`,
`--delay-ms` (default 500 between sessions), `--dir`, `--claude-root`,
`--opencode-db` and `--min-messages` tune the sweep. Reading OpenCode needs
Node >= 22.5 (`node:sqlite`). Replay labels are Jev judging a session's end
state rather than a mid-session compaction; records keep `source` so training
can tell them apart.

`result.stats` reports message and character counts before and after, the
per-reason decision counts, the state size in estimated tokens, which fitting
stage was needed, and the number of requests.

## Limitations

- Only tool calls and results are candidates; text messages are never removed
  or shortened in the output (they are only abridged in the state Jev sees).
- Token sizes are estimates from character counts, not a tokenizer.
- Calibration is at the request level; a probability is not a proof that a
  result is safe to delete. The assistant can always re-run the tool.
- The full state is repeated with every request, so a history near the state
  ceiling costs one request per handful of questions.

## Claude Code plugin

The repository root is a Claude Code function-hook plugin: `hooks/fast-compaction.ts`
is a thin adapter that feeds `session.compact` transcripts through `src/` and
falls back to Claude Code's built-in summary on errors or insufficient
reduction. See [`hooks/README.md`](hooks/README.md) for configuration and the
Claude Code 2.1.274 type reference.

### Install in Claude Code

Function hooks are an early-access Claude Code feature (2.1.274+), so the
opt-in flag must be set wherever Claude Code runs, e.g. in `~/.claude/settings.json`:

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1", "TYPESAFE_API_KEY": "<your key>" } }
```

Then add this repository as a plugin marketplace and install the plugin,
either from the shell or as slash commands inside a session:

```sh
claude plugin marketplace add tamaratran/fast-jev-compaction
claude plugin install fast-compaction@fast-compaction
```

The install prompts for the plugin options (API key, thresholds, `truncateHeadChars`,
…); leave them at their defaults to use `TYPESAFE_API_KEY` from the environment.
Restart Claude Code or run `/reload-plugins`. From then on `/compact` (and
auto-compaction) goes through Jev: the toast reads
`fast-compaction: kept N/M messages, no summary (…)` when the pruned history
replaced the built-in summary, or `fallback to built-in summary (…)` when Jev
could not remove enough (short sessions, or when it fails).

To run from a checkout without installing: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`
from the repository root. No publishing step is required; the marketplace is
just the repo's `.claude-plugin/marketplace.json`.

## Development

```sh
npm install
npm run typecheck        # library + hook
npm test
npm run build
npm run validate:plugin  # claude plugin validate
TYPESAFE_API_KEY="$(cat ~/.typesafe_key)" npm run demo
```

The unit tests use a fake Jev and never contact TypeSafe. The demo is the live
network check.

## Animated demo (macOS)

`demo/JevDemo` is a small native SwiftUI app that plays a scripted, dramatized
version of the compaction flow inside a Claude Code-style terminal: the tool
calls of a canned transcript are scored, results and calls Jev lets go turn red
and collapse away, and the rest stays verbatim. It never calls the API; it
exists to be screen recorded.

```sh
demo/JevDemo/build.sh   # builds demo/JevDemo/build/JevDemo.app and launches it
```

Press space in the app to replay from the start.
