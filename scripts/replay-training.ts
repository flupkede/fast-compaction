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
  --dir <path>            training dir (default ~/.fast-jev/training)
  --claude-root <path>    Claude Code projects root (default ~/.claude/projects)
  --opencode-db <path>    OpenCode opencode.db (default ~/.local/share/opencode/opencode.db)
  --limit <n>             replay at most n sessions this run
  --delay-ms <n>          pause between replayed sessions (default 500)
  --min-messages <n>      skip OpenCode sessions with fewer stored messages (default 8)
  --no-strip              disable identifier pseudonymisation (paths, URLs,
                          e-mails, GUIDs, IPs, domains, blocklist)
  --blocklist <path>      file with extra terms (client names), one per line
  --asker <which>         cloud (TypeSafe, needs ${TYPESAFE_API_KEY_ENV}) or
                          ollama (local decision model, e.g. clef-flash)
  --base-url <url>        full System One endpoint for --asker ollama
                          (default ${OLLAMA_SYSTEM_ONE_URL})
  --model <name>          model for --asker ollama (default ${OLLAMA_MODEL})
  --max-state-tokens <n>  state token ceiling (default 25000; local models with
                          small context, e.g. tev1:4b, need ~2000)

Identifier stripping is ON by default: identifiers become stable tokens
([PATH_1], [URL_2], ...) after secret redaction, before anything is sent.
With --asker ollama nothing leaves this machine at all. Reading OpenCode
requires Node >= 22.5. Re-run to resume: finished sessions are ledgered in
<dir>/replay-index.json.
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

/** Picks the judge from the parsed flags: local Ollama, or the TypeSafe cloud. */
export function buildAsker(args: ReplayArgs): JevAsker {
  if (args.asker === 'ollama') {
    return new JevClient({
      baseUrl: args.baseUrl ?? OLLAMA_SYSTEM_ONE_URL,
      apiKey: 'ollama',
      model: args.model ?? OLLAMA_MODEL,
    });
  }
  if (!process.env[TYPESAFE_API_KEY_ENV]) {
    throw new Error(`${TYPESAFE_API_KEY_ENV} is not configured (or use --asker ollama)`);
  }
  return new JevClient();
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
  let compactOptions: CompactOptions | undefined;
  if (args.maxStateTokens !== undefined) {
    compactOptions = { ...compactOptions, maxStateTokens: args.maxStateTokens };
  }
  if (!args.noStrip) {
    const blocklist = args.blocklistPath
      ? (await readFile(args.blocklistPath, 'utf8'))
          .split(/\r?\n/)
          .map((line) => line.trim())
          .filter((line) => line.length > 0 && !line.startsWith('#'))
      : [];
    compactOptions = {
      ...compactOptions,
      strip: blocklist.length > 0 ? { blocklist } : {},
    };
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
