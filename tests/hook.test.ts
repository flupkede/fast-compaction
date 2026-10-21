import { describe, expect, it } from 'vitest';
import {
  COMPRESS_COMMAND,
  checkToolCall,
  compactSession,
  decisionLog,
  decisionLogLines,
  hotOptions,
  hookTimeBudget,
  nextFloorPercent,
  register,
  resolveHookConfig,
  shouldAutoCompact,
  STATS_COMMAND,
  summarize,
  toSessionMessages,
  withDeadline,
} from '../hooks/fast-compaction.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      targetFloorStartPercent: 50,
      targetFloorMinPercent: 25,
      targetFloorStepPercent: 5,
      model: 'jev-latest',
      judgeTimeoutMs: 8000,
      toolCallLimit: 30,
      toolCallWindowMs: 900_000,
      compactRegainPercent: 20,
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      targetFloorStartPercent: 50,
      targetFloorMinPercent: 25,
      targetFloorStepPercent: 5,
      judgeTimeoutMs: 8000,
      toolCallLimit: 30,
      toolCallWindowMs: 900_000,
      compactRegainPercent: 20,
    });
  });

  it('applies ollama defaults when asker is ollama', () => {
    expect(resolveHookConfig({ asker: 'ollama' })).toEqual({
      asker: 'ollama',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      targetFloorStartPercent: 50,
      targetFloorMinPercent: 25,
      targetFloorStepPercent: 5,
      baseUrl: 'http://localhost:11434/v1/systemone',
      model: 'clef-flash',
      judgeTimeoutMs: 600000,
      toolCallLimit: 30,
      toolCallWindowMs: 900_000,
      compactRegainPercent: 20,

    });
  });

  it('hotOptions prefers fresh pluginConfigs over register-time options', async () => {
    const $ = {
      settings: {
        read: async () => ({
          pluginConfigs: {
            'fast-compaction@fast-compaction': { options: { keepThreshold: 0.55, maxQuestionsPerRequest: 24 } },
          },
        }),
      },
    };
    const hot = await hotOptions($ as never, { keepThreshold: 0.5 });
    expect(hot).toEqual({ keepThreshold: 0.55, maxQuestionsPerRequest: 24 });
  });

  it('hotOptions falls back to register-time options when settings lack the entry', async () => {
    const fallback = { keepThreshold: 0.5 };
    expect(await hotOptions({ settings: { read: async () => ({}) } } as never, fallback)).toBe(fallback);
    expect(
      await hotOptions({ settings: { read: async () => ({ pluginConfigs: { other: {} } }) } } as never, fallback),
    ).toBe(fallback);
    const throwing = { settings: { read: async () => { throw new Error('filtered'); } } };
    expect(await hotOptions(throwing as never, fallback)).toBe(fallback);
  });

  it('derives the time budget from the judge deadline, engine-aware', () => {
    expect(hookTimeBudget(100_000)).toBe(60_000);
    // never past the engine's ~2min hook budget, even with a 10min timeout
    expect(hookTimeBudget(600_000)).toBe(110_000);
    // never below 30s of work
    expect(hookTimeBudget(50_000)).toBe(30_000);
  });

  it('still reads the pre-rename jevTimeoutMs option name', () => {
    expect(resolveHookConfig({ jevTimeoutMs: 123_456 }).judgeTimeoutMs).toBe(123_456);
    expect(resolveHookConfig({ judgeTimeoutMs: 999 }).judgeTimeoutMs).toBe(999);
    // the new name wins when both are present
    expect(resolveHookConfig({ judgeTimeoutMs: 999, jevTimeoutMs: 123_456 }).judgeTimeoutMs).toBe(999);
  });

  it('limits repeated calls of one tool and forgets them after the window', () => {
    const state = new Map<string, number[]>();
    // limit 3: three calls pass, the fourth denies with guidance
    expect(checkToolCall(state, 'mcp__aprimo__dam_browser', 1_000, 3, 900_000)).toBeNull();
    expect(checkToolCall(state, 'mcp__aprimo__dam_browser', 2_000, 3, 900_000)).toBeNull();
    expect(checkToolCall(state, 'mcp__aprimo__dam_browser', 3_000, 3, 900_000)).toBeNull();
    const deny = checkToolCall(state, 'mcp__aprimo__dam_browser', 4_000, 3, 900_000);
    expect(deny).toMatch(/dam_browser called 4x/);
    expect(deny).toMatch(/toolCallLimit=3/);
    // a different tool is unaffected
    expect(checkToolCall(state, 'Read', 4_000, 3, 900_000)).toBeNull();
    // outside the window the old calls are forgotten
    expect(checkToolCall(state, 'mcp__aprimo__dam_browser', 4_000 + 900_001, 3, 900_000)).toBeNull();
    // limit 0 disables the limiter entirely
    expect(checkToolCall(new Map(), 'Bash', 1_000, 0, 900_000)).toBeNull();
  });

  it('holds the auto trigger while the context has not regrown past its post-compaction low', () => {
    const base = {
      trigger: 'auto' as const,
      tokens: 168_000,
      minTokensSinceCompact: 136_000,
      effective: 200_000,
      regainPercent: 20,
    };
    // 32k growth < 20% of the window: held (the treadmill case)
    expect(shouldAutoCompact({ ...base, now: 1_000_000, lastCompactAt: 900_000 })).toBe(false);
    expect(shouldAutoCompact({ ...base, now: 1_000_000, lastCompactAt: 500_000 })).toBe(false);
    // growth >= regain: fires
    expect(
      shouldAutoCompact({ ...base, tokens: 180_000, now: 1_000_000, lastCompactAt: 500_000 }),
    ).toBe(true);
    // first cycle without a compaction yet: fires
    expect(
      shouldAutoCompact({
        trigger: 'auto',
        tokens: 130_000,
        effective: 200_000,
        regainPercent: 20,
        now: 1_000_000,
      }),
    ).toBe(true);
    // within the cooldown right after a compaction: held even without a sampled low
    expect(
      shouldAutoCompact({
        trigger: 'auto',
        tokens: 140_000,
        effective: 200_000,
        regainPercent: 20,
        now: 1_000_000,
        lastCompactAt: 990_000,
      }),
    ).toBe(false);
    // /compress force bypasses the guard entirely
    expect(shouldAutoCompact({ ...base, trigger: 'force', now: 1_000_000, lastCompactAt: 999_000 })).toBe(true);
    // regainPercent 0 disables the guard
    expect(
      shouldAutoCompact({ ...base, regainPercent: 0, now: 1_000_000, lastCompactAt: 500_000 }),
    ).toBe(true);
    // no context reading: legacy behaviour (fires)
    expect(
      shouldAutoCompact({
        trigger: 'auto',
        effective: 200_000,
        regainPercent: 20,
        now: 1_000_000,
        lastCompactAt: 500_000,
      }),
    ).toBe(true);
  });

  it('lets explicit baseUrl, model and judgeTimeoutMs win over the ollama defaults', () => {
    const config = resolveHookConfig({
      asker: 'ollama',
      baseUrl: 'http://gpu:11434/v1/systemone',
      ollamaUrl: 'http://ignored:11434/v1/systemone',
      model: 'student',
      judgeTimeoutMs: 1234,
    });
    expect(config.baseUrl).toBe('http://gpu:11434/v1/systemone');
    expect(config.model).toBe('student');
    expect(config.judgeTimeoutMs).toBe(1234);
  });

  it('falls back to ollamaUrl when baseUrl is not set for the ollama asker', () => {
    const config = resolveHookConfig({ asker: 'ollama', ollamaUrl: 'http://gpu:11434/v1/systemone' });
    expect(config.baseUrl).toBe('http://gpu:11434/v1/systemone');
  });

  it('ignores ollamaUrl when the asker is cloud', () => {
    const config = resolveHookConfig({ ollamaUrl: 'http://gpu:11434/v1/systemone' });
    expect(config.baseUrl).toBeUndefined();
    expect(config.model).toBe('jev-latest');
    expect(config.judgeTimeoutMs).toBe(8000);
  });
});

describe('nextFloorPercent', () => {
  const config = {
    targetFloorStartPercent: 50,
    targetFloorMinPercent: 25,
    targetFloorStepPercent: 5,
  };

  it('steps down from the start on a Jev compaction and clamps at the minimum', () => {
    expect(nextFloorPercent(undefined, 'judge', config)).toBe(45);
    expect(nextFloorPercent(45, 'judge', config)).toBe(40);
    expect(nextFloorPercent(27, 'judge', config)).toBe(25);
    expect(nextFloorPercent(25, 'judge', config)).toBe(25);
  });

  it('resets to the start after any fallback', () => {
    expect(nextFloorPercent(35, 'fallback-low', config)).toBe(50);
    expect(nextFloorPercent(35, 'fallback-error', config)).toBe(50);
  });
});

describe('withDeadline', () => {
  it('passes the work through when it settles first', async () => {
    await expect(withDeadline(Promise.resolve(1), 10, () => new Promise(() => {}))).resolves.toBe(1);
  });

  it('rejects when the deadline passes first', async () => {
    await expect(withDeadline(new Promise(() => {}), 10, async () => {})).rejects.toThrow(/within 10 ms/);
  });
});

describe('slash commands', () => {
  type HookEntry = {
    pattern: unknown;
    matcher?: Record<string, unknown>;
    hook: (...args: unknown[]) => unknown;
  };

  /** Drives register() without an engine: captures every on() registration. */
  function registeredHooks(): HookEntry[] {
    const entries: HookEntry[] = [];
    const on = (pattern: unknown, ...rest: unknown[]) => {
      const matcher =
        typeof rest[0] === 'object' && rest[0] !== null && typeof rest[1] === 'function'
          ? (rest[0] as Record<string, unknown>)
          : undefined;
      entries.push({
        pattern,
        matcher,
        hook: (matcher ? rest[1] : rest[0]) as (...args: unknown[]) => unknown,
      });
      return { catch: () => {} };
    };
    register(on as unknown as Parameters<typeof register>[0], {});
    return entries;
  }

  function runCompress(compact: () => Promise<unknown>): Promise<unknown> {
    const entry = registeredHooks().find(
      (e) => e.pattern === 'command.run' && e.matcher?.command === COMPRESS_COMMAND,
    );
    return entry!.hook({ session: { compact } }, { command: COMPRESS_COMMAND, args: '' }) as Promise<unknown>;
  }

  it('registers the stats and compress commands on session.start', async () => {
    const start = registeredHooks().find((entry) => entry.pattern === 'session.start');
    const specs: { name: string }[] = [];
    await start!.hook(
      {
        command: { register: async (spec: { name: string }) => void specs.push(spec) },
        clock: { now: async () => 0 },
        store: {
          keys: async () => [],
          get: async () => undefined,
          set: async () => {},
          delete: async () => {},
        },
        ui: { log: () => {} },
      },
      {},
      (event: unknown) => event,
    );
    expect(specs.map((spec) => spec.name)).toEqual([STATS_COMMAND, COMPRESS_COMMAND]);
  });

  it('compress queues the compaction; the next turn.complete runs it past the threshold', async () => {
    const hooks = registeredHooks();
    const compress = hooks.find(
      (e) => e.pattern === 'command.run' && e.matcher?.command === COMPRESS_COMMAND,
    )!;
    const turn = hooks.find((e) => e.pattern === 'turn.complete')!;
    let calls = 0;
    const $ = {
      session: {
        usage: async () => ({ context: { tokens: 1000, percent: 5 } }),
        compact: async () => {
          calls += 1;
          return { messages: [] };
        },
      },
      clock: { now: async () => 0, sleep: async () => {} },
      store: { keys: async () => [], get: async () => undefined, set: async () => {}, delete: async () => {} },
      ui: { log: () => {} },
    };
    const queued = await compress.hook(
      $,
      { command: COMPRESS_COMMAND, args: '' },
    );
    expect(queued).toEqual({ text: 'compaction queued — it runs right after this turn' });
    expect(calls).toBe(0);
    await turn.hook($, { agentId: undefined }, (event: unknown) => event);
    expect(calls).toBe(1);
  });

  it('turn.complete without the flag respects the percent threshold', async () => {
    const turn = registeredHooks().find((e) => e.pattern === 'turn.complete')!;
    let calls = 0;
    await turn.hook(
      {
        session: {
          usage: async () => ({ context: { tokens: 1000, percent: 5 } }),
          compact: async () => {
            calls += 1;
            return { messages: [] };
          },
        },
        clock: { now: async () => 0, sleep: async () => {} },
        store: { keys: async () => [], get: async () => undefined, set: async () => {}, delete: async () => {} },
        ui: { log: () => {} },
      },
      { agentId: undefined },
      (event: unknown) => event,
    );
    expect(calls).toBe(0);
  });

  it('a rejected attempt re-arms and the next turn.complete retries below the threshold', async () => {
    const turn = registeredHooks().find((e) => e.pattern === 'turn.complete')!;
    let calls = 0;
    let reject = true;
    let percent = 90;
    const logs: string[] = [];
    const $ = {
      session: {
        id: async () => 'rearm-test',
        usage: async () => ({ context: { tokens: 1000, percent } }),
        compact: async () => {
          if (reject) throw new Error('a turn is running');
          calls += 1;
          return { messages: [] };
        },
      },
      clock: { now: async () => 0, sleep: async () => {} },
      store: { keys: async () => [], get: async () => undefined, set: async () => {}, delete: async () => {} },
      ui: { log: (text: string) => void logs.push(text) },
    };
    // first turn: above threshold, but the host rejects — must re-arm, not die
    await turn.hook($, { agentId: undefined }, (event: unknown) => event);
    expect(calls).toBe(0);
    expect(logs).toEqual(['compaction deferred to the next turn (a turn is running)']);
    // second turn: below threshold, but the re-armed trigger compacts anyway
    reject = false;
    percent = 5;
    await turn.hook($, { agentId: undefined }, (event: unknown) => event);
    expect(calls).toBe(1);
  });

  it('triggers on tokens against the autoCompactWindow, not the model window', async () => {
    const turn = registeredHooks().find((e) => e.pattern === 'turn.complete')!;
    let calls = 0;
    await turn.hook(
      {
        session: {
          usage: async () => ({ context: { tokens: 150000, window: 1000000, percent: 15 } }),
          compact: async () => {
            calls += 1;
            return { messages: [] };
          },
        },
        settings: { read: async () => ({ autoCompactWindow: 200000 }) },
        clock: { now: async () => 0, sleep: async () => {} },
        store: { keys: async () => [], get: async () => undefined, set: async () => {}, delete: async () => {} },
        ui: { log: () => {} },
      },
      { agentId: undefined },
      (event: unknown) => event,
    );
    // 150k of a 200k auto-compact window = 75% — the host compacts soon; the
    // model-window percent (15% of 1M) must not keep the plugin idle.
    expect(calls).toBe(1);
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(2); // one call batch + one message batch
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    // msg prose judged droppable (0.1) disappears; seed, the kept tool call and the recent window stay
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 2 request\(s\), \d+ ms$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });

  it('posts to baseUrl when a local judge endpoint is configured, without an API key', async () => {
    const urls: string[] = [];
    const bodies: string[] = [];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1 }),
      baseUrl: 'http://pc:11434/v1/systemone',
      model: 'clef-flash',
    };
    const fetch = async (url: string, init?: { body?: string }) => {
      urls.push(url);
      return jevFetch(() => 0.9, bodies)(url, init);
    };
    const { result } = await compactSession(transcript(), config, fetch);
    expect(urls).toEqual(['http://pc:11434/v1/systemone', 'http://pc:11434/v1/systemone']);
    expect(JSON.parse(bodies[0]!).model).toBe('clef-flash');
    expect(result.stats.requests).toBe(2);
  });

  it('accepts an ollama-shaped config without an API key', async () => {
    const config = resolveHookConfig({ asker: 'ollama' });
    expect(config.baseUrl).toBeDefined();
    await expect(compactSession(transcript(), config, jevFetch(() => 0.9))).resolves.toHaveProperty('result');
  });
});
