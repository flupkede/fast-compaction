/**
 * Replays stored Claude Code and OpenCode sessions through the live Jev
 * pipeline and appends one replay-tagged training record per session that
 * asks Jev. Run `npm run replay -- --help` for options.
 */
import { access, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  createTrainingLogger,
  DEFAULT_MODEL,
  DEFAULT_TRAINING_DIR,
  JevClient,
  listClaudeTranscriptFiles,
  loadClaudeTranscript,
  OLLAMA_MODEL,
  OLLAMA_SYSTEM_ONE_URL,
  openOpencodeTranscripts,
  parseReplayArgs,
  planReplay,
  runReplay,
  SYSTEM_ONE_URL,
  TYPESAFE_API_KEY_ENV,
  type CompactOptions,
  type JevAsker,
  type Message,
  type ReplayArgs,
  type ReplaySourceSpec,
} from '../src/index.js';

const DEFAULTS = {
  claudeRoot: join(homedir(), '.claude', 'projects'),
  opencodeDb: join(homedir(), '.local', 'share', 'opencode', 'opencode.db'),
  dir: DEFAULT_TRAINING_DIR,
  delayMs: 500,
  minMessages: 8,
};

const USAGE = `replay-training: turn stored sessions into labelled training records

  npm run replay -- [options]

Options:
  --dry-run               count sessions, candidates and requests; ask nothing
  --source <which>        claude | opencode | both (default both)
  --dir <path>            training dir (default ~/.fast-compaction/training)
  --claude-root <path>    Claude Code projects root (default ~/.claude/projects)
  --opencode-db <path>    OpenCode opencode.db (default ~/.local/share/opencode/opencode.db)
  --limit <n>             replay at most n sessions this run
  --delay-ms <n>          pause between replayed sessions (default 500)
  --min-messages <n>      skip OpenCode sessions with fewer stored messages (default 8)
  --no-strip              disable identifier pseudonymisation (paths, URLs,
                          e-mails, GUIDs, IPs, domains, blocklist)
  --blocklist <path>      file with extra terms (client names), one per line
  --asker <which>         ollama (local decision model, e.g. clef-flash;
                          default) or cloud (TypeSafe, needs
                          ${TYPESAFE_API_KEY_ENV})
  --base-url <url>        full System One endpoint for --asker ollama
                          (default ${OLLAMA_SYSTEM_ONE_URL})
  --model <name>          model for --asker ollama (default ${OLLAMA_MODEL})
                          OLLAMA_SYSTEM_ONE_URL / OLLAMA_MODEL env vars persist
                          a LAN endpoint and model; flags still win
  --max-state-tokens <n>  state token ceiling (local default 8000 — the
                          decision prompt must fit a typical local num_ctx
                          of 16384 and the endpoint's 64 KiB request cap;
                          cloud default 25000; small-context models, e.g.
                          tev1:4b, need ~2000)
  --request-timeout-ms <n> per-request judge timeout (default 600000; the
                          OLLAMA_REQUEST_TIMEOUT_MS env var persists it)
  --no-segment            disable the default splitting of sessions ledgered
                          "history too large" into the fewest possible windows,
                          cut at user prompts, replayed as their own records

Identifier stripping is ON by default: identifiers become stable tokens
([PATH_1], [URL_2], ...) after secret redaction, before anything is sent.
The judge defaults to local Ollama: nothing leaves this machine and no API
key is needed. Reading OpenCode requires Node >= 22.5. Re-run to resume:
finished sessions are ledgered in <dir>/replay-index.json.
`;

async function claudeSources(root: string): Promise<ReplaySourceSpec[]> {
  return (await listClaudeTranscriptFiles(root)).map((file) => ({
    key: `claude:${file}`,
    label: file,
    load: (): Promise<readonly Message[]> => loadClaudeTranscript(file),
  }));
}

async function opencodeSources(dbPath: string, minMessages: number): Promise<ReplaySourceSpec[]> {
  const transcripts = await openOpencodeTranscripts(dbPath);
  return transcripts.sessions(minMessages).map((session) => ({
    key: `opencode:${session.id}`,
    label: `${session.title} (${session.directory})`,
    load: async (): Promise<readonly Message[]> => transcripts.messages(session.id),
  }));
}

/**
 * Hard ceiling per judge request (local inference on a LAN GPU is slow; the
 * breaker stops repeated failures anyway). A local model can still loop
 * indefinitely on degenerate content (observed: clef-flash never answering a
 * repeated-text state), so the client aborts first, with a phrase the
 * replay ledger treats as retriable.
 */
export const REQUEST_TIMEOUT_MS = 600_000;

/** Resolves the per-request judge timeout: flag > OLLAMA_REQUEST_TIMEOUT_MS env > default. */
export function resolveRequestTimeoutMs(args: Pick<ReplayArgs, 'requestTimeoutMs'>): number {
  const env = Number(process.env.OLLAMA_REQUEST_TIMEOUT_MS);
  return args.requestTimeoutMs ?? (Number.isFinite(env) && env > 0 ? env : REQUEST_TIMEOUT_MS);
}

/** Fetch that aborts after `timeoutMs` with a ledger-classifiable error. */
export function fetchWithTimeout(timeoutMs: number): typeof fetch {
  return (url, init) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { ...init, signal: controller.signal })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          throw new Error(`Jev request timed out (no answer within ${timeoutMs / 1000}s)`);
        }
        throw error;
      })
      .finally(() => clearTimeout(timer));
  };
}

/** Resolves the local judge once; buildAsker and the run banner share it so they cannot drift. */
export function resolveLocalJudge(args: Pick<ReplayArgs, 'baseUrl' | 'model'>): { baseUrl: string; model: string } {
  // --base-url / --model flag > OLLAMA_* env var > built-in default
  return {
    baseUrl: args.baseUrl ?? process.env.OLLAMA_SYSTEM_ONE_URL ?? OLLAMA_SYSTEM_ONE_URL,
    model: args.model ?? process.env.OLLAMA_MODEL ?? OLLAMA_MODEL,
  };
}

/** Picks the judge from the parsed flags: local Ollama by default, or the TypeSafe cloud. */
export function buildAsker(args: ReplayArgs): JevAsker {
  if (args.asker === 'cloud') {
    if (!process.env[TYPESAFE_API_KEY_ENV]) {
      throw new Error(`${TYPESAFE_API_KEY_ENV} is not configured for --asker cloud (--asker ollama is the local default)`);
    }
    return new JevClient();
  }
  const judge = resolveLocalJudge(args);
  return new JevClient({
    baseUrl: judge.baseUrl,
    apiKey: 'ollama',
    model: judge.model,
    fetch: fetchWithTimeout(resolveRequestTimeoutMs(args)),
  });
}

/**
 * Local judges typically run num_ctx 16384 and the endpoint rejects bodies
 * over 64 KiB; the decision prompt (state + instructions + schema) must fit
 * both, and the estimate runs 2-18% under real token usage.
 */
export const LOCAL_MAX_STATE_TOKENS = 8000;

/** Builds compactOptions from the parsed flags: token ceiling + identifier stripping. */
export function buildCompactOptions(
  args: Pick<ReplayArgs, 'maxStateTokens' | 'noStrip' | 'blocklistPath' | 'asker'>,
  blocklist: string[],
): CompactOptions {
  let options: CompactOptions = {};
  if (args.maxStateTokens !== undefined) options = { ...options, maxStateTokens: args.maxStateTokens };
  // local is the default asker and its endpoint caps request size; cloud keeps the library default
  else if (args.asker !== 'cloud') options = { ...options, maxStateTokens: LOCAL_MAX_STATE_TOKENS };
  if (!args.noStrip) {
    options = { ...options, strip: blocklist.length > 0 ? { blocklist } : {} };
  }
  return options;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log(USAGE);
    return;
  }
  const args = parseReplayArgs(argv);
  const dir = args.dir ?? DEFAULTS.dir;
  const delayMs = args.delayMs ?? DEFAULTS.delayMs;
  const minMessages = args.minMessages ?? DEFAULTS.minMessages;
  const blocklist = args.noStrip
    ? []
    : args.blocklistPath
      ? (await readFile(args.blocklistPath, 'utf8'))
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line.length > 0 && !line.startsWith('#'))
      : [];
  const compactOptions = buildCompactOptions(args, blocklist);
  // one line naming the judge this run will actually use (flag > env > default)
  if (args.asker === 'cloud') {
    console.log(`judge: ${DEFAULT_MODEL} @ ${SYSTEM_ONE_URL} (cloud)`);
  } else {
    const judge = resolveLocalJudge(args);
    console.log(`judge: ${judge.model} @ ${judge.baseUrl} (local)`);
  }
  const sources: ReplaySourceSpec[] = [];
  if (args.source !== 'opencode') {
    const claude = await claudeSources(args.claudeRoot ?? DEFAULTS.claudeRoot);
    console.log(`claude: ${claude.length} session file(s)`);
    sources.push(...claude);
  }
  if (args.source !== 'claude') {
    const dbPath = args.opencodeDb ?? DEFAULTS.opencodeDb;
    // an explicitly named database must exist; a missing default is just skipped
    if (args.opencodeDb !== undefined || (await access(dbPath).then(() => true, () => false))) {
      const opencode = await opencodeSources(dbPath, minMessages);
      console.log(`opencode: ${opencode.length} session(s)`);
      sources.push(...opencode);
    } else {
      console.log(`opencode: no database at ${dbPath}; use --opencode-db or --source claude`);
    }
  }
  if (args.dryRun) {
    let sessions = 0;
    let calls = 0;
    let requests = 0;
    let tokens = 0;
    let errors = 0;
    for (const source of sources) {
      let messages: readonly Message[];
      try {
        messages = await source.load();
      } catch (error) {
        errors++;
        console.error(`  unreadable ${source.key}: ${error instanceof Error ? error.message : error}`);
        continue;
      }
      const plan = planReplay(messages, compactOptions);
      sessions++;
      calls += plan.calls;
      requests += plan.requests;
      tokens += plan.tokens;
      if (plan.error) {
        errors++;
        console.error(`  unplannable ${source.key}: ${plan.error}`);
      }
    }
    console.log(
      `\ndry run: ${sessions} session(s), ${calls} candidate call(s), ~${requests} Jev request(s), ~${tokens} state tokens, ${errors} error(s)${args.noStrip ? '' : ', identifiers stripped'}`,
    );
    console.log(`run without --dry-run to write the records (cloud needs ${TYPESAFE_API_KEY_ENV}).`);
    return;
  }
  const summary = await runReplay(
    sources,
    buildAsker(args),
    createTrainingLogger({ dir }),
    {
      dir,
      limit: args.limit,
      delayMs,
      compactOptions,
      segmentOversized: args.segment !== false,
      sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
      now: () => Date.now(),
    },
    (event) => console.log(`${event.outcome.padEnd(14)} ${event.key}${event.detail ? ` — ${event.detail}` : ''}`),
  );
  console.log(
    `\nreplayed ${summary.replayed} session(s), ${summary.records} record(s) written to ${dir}; ` +
      `${summary.skippedDone} already done, ${summary.noCandidates} without candidates, ${summary.errors} error(s).`,
  );
}

// run only when invoked directly; importing (tests) must stay side-effect free
const invokedDirectly =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
