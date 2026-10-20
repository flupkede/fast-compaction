import { describe, expect, it } from 'vitest';
import {
  applyDecisions,
  batchCalls,
  buildJevRequest,
  collectToolCalls,
  compact,
  compactMessages,
  decideCall,
  decideMessage,
  estimateTokens,
  fitState,
  JevClient,
  messageChunkSize,
  parseJevResponse,
   reductionRatio,
   resolveOptions,
   transcriptTokens,
   type HistoryToolCall,
  type JevAsker,
  type JevQuestions,
  type Message,
  type ToolCall,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const fileB = 'export const b = 2;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'a.ts looks fine; checking b.ts'),
    call('tool-2', 'Read', { file_path: 'src/b.ts' }, fileB),
    result('tool-2', fileB),
    call('tool-3', 'Bash', { command: 'npm test' }, 'FAIL b.test.ts'),
    result('tool-3', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'The failure is in b.test.ts; fixing now.'),
    message('user', 'go ahead'),
  ];
}

type Seen = { state: unknown; questions: string[] };

function fakeJev(answer: (name: string) => number, seen: Seen[] = []): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      seen.push({ state, questions: Object.keys(questions) });
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

const fit = {
  maxStateTokens: 25_000,
  preserveRecentMessages: 0,
  goal: 'fix the test',
};

describe('options', () => {
  it('fills in defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 6,
      maxStateTokens: 25_000,
      maxRequestTokens: 26_000,
      maxQuestionsPerRequest: 12,
      askConcurrency: 2,
      truncateHeadChars: 300,
    });
    expect(resolveOptions({
      keepThreshold: Number.NaN,
      preserveRecentMessages: 2.7,
      truncateHeadChars: -1.2,
    })).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 2,
      truncateHeadChars: 0,
    });
  });

  it('clamps the question cap to the route limit (64) and concurrency to at least 1', () => {
    expect(resolveOptions({ maxQuestionsPerRequest: 500 })).toMatchObject({
      maxQuestionsPerRequest: 64,
    });
    expect(resolveOptions({ maxQuestionsPerRequest: 0.9 })).toMatchObject({
      maxQuestionsPerRequest: 1,
    });
    expect(resolveOptions({ askConcurrency: 0 })).toMatchObject({ askConcurrency: 1 });
  });
});

describe('token estimate', () => {
  it('charges words, digits and symbols separately and never undercounts JSON badly', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('hello world')).toBe(2);
    expect(estimateTokens('internationalization')).toBe(4);
    expect(estimateTokens('12345678')).toBe(4);
    const json = JSON.stringify({ file_path: '/Users/x/src/a.ts', old_string: 'a = 1;', n: 42 });
    expect(estimateTokens(json)).toBeGreaterThanOrEqual(Math.ceil(json.length / 3));
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins recent ones', () => {
    const calls = collectToolCalls(transcript(), 3);
    expect(calls.map((c) => [c.id, c.tool, c.callIndex, c.resultIndex, c.pinned])).toEqual([
      ['t1', 'Read', 1, 2, false],
      ['t2', 'Read', 4, 5, false],
      ['t3', 'Bash', 6, 7, true],
    ]);
    expect(calls[2]?.isError).toBe(true);
    expect(calls[0]?.resultChars).toBe(fileA.length);
  });

  it('ignores calls without a result', () => {
    expect(collectToolCalls([message('user', 'hi'), call('x', 'Read', {}, '')], 0)).toHaveLength(0);
  });
});

describe('state fitting', () => {
  it('sends the whole history with tool results replaced by a note', () => {
    const messages = transcript();
    const { state, stage } = fitState(messages, collectToolCalls(messages, 0), fit);
    expect(stage).toBe('full');
    const json = JSON.stringify(state);
    expect(json).not.toContain('export const a = 1;');
    expect(json).toContain('Never edit anything under src/generated');
    expect(json).toContain('go ahead');
    expect(state.history.map((entry) => entry.i)).toEqual([0, 1, 3, 4, 6, 8, 9]);
    expect(state.history[1]?.tool_calls?.[0]).toMatchObject({
      id: 't1',
      tool: 'Read',
      result: `ok, ${fileA.length} chars (omitted)`,
    });
    expect((state.history[4]?.tool_calls?.[0] as HistoryToolCall).result).toMatch(/^error, /);
  });

  it('defaults the goal to the latest user prompts', () => {
    const { state } = fitState(transcript(), [], { ...fit, goal: '' });
    expect(state.goal).toContain('Fix the failing test');
    expect(state.goal).toContain('go ahead');
  });

  it('truncates tool inputs before touching message text', () => {
    const messages = [
      message('user', 'start'),
      call('w', 'Write', { file_path: 'x.ts', content: 'x'.repeat(5000) }, 'ok'),
      result('w', 'ok'),
      message('assistant', 'written'),
    ];
    const { state, stage, tokens } = fitState(messages, collectToolCalls(messages, 0), {
      ...fit,
      maxStateTokens: 300,
    });
    expect(stage).toBe('inputs<=200');
    expect(tokens).toBeLessThanOrEqual(300);
    expect(state.history[0]?.text).toBe('start');
    expect((state.history[1]?.tool_calls?.[0] as HistoryToolCall).input.length).toBeLessThanOrEqual(200);
  });

  it('shrinks old tool calls to one line each when nothing else is left to cut', () => {
    const messages = [message('user', 'start')];
    for (let i = 0; i < 40; i += 1) {
      messages.push(call(`c${i}`, 'Read', { file_path: `/repo/src/module-${i}.ts` }, 'x'), result(`c${i}`, 'x'));
    }
    messages.push(message('assistant', 'done'));
    const calls = collectToolCalls(messages, 1);
    const full = fitState(messages, calls, { ...fit, preserveRecentMessages: 1 });
    const compacted = fitState(messages, calls, {
      ...fit,
      preserveRecentMessages: 1,
      maxStateTokens: Math.floor(full.tokens * 0.8),
    });
    expect(compacted.stage).toBe('old calls compacted');
    expect(compacted.tokens).toBeLessThanOrEqual(Math.floor(full.tokens * 0.8));
    expect(compacted.tokens).toBeGreaterThanOrEqual(estimateTokens(JSON.stringify(compacted.state)));
    expect(compacted.state.history[1]?.tool_calls?.[0]).toBe(
      't1 Read file_path=/repo/src/module-0.ts → ok 1ch',
    );
    expect(compacted.state.history.at(-1)?.text).toBe('done');

    const merged = fitState(messages, calls, {
      ...fit,
      preserveRecentMessages: 1,
      maxStateTokens: Math.floor(full.tokens * 0.45),
    });
    expect(merged.stage).toBe('old calls merged');
    expect(merged.tokens).toBeLessThanOrEqual(Math.floor(full.tokens * 0.45));
    expect(merged.state.history).toHaveLength(3);
    expect(merged.state.history[1]?.tool_calls).toHaveLength(40);
    expect(merged.state.history[1]?.tool_calls?.[39]).toMatch(/^t40 Read /);
    expect(merged.state.history[0]?.text).toBe('start');
    expect(merged.state.history[2]?.text).toBe('done');
  });

  it('abridges long texts oldest-first and collapses old messages last', () => {
    const long = (n: number) => `${n} ` + 'lorem ipsum '.repeat(300);
    const messages = [
      message('user', long(0)),
      message('assistant', long(1)),
      message('user', long(2)),
      message('assistant', long(3)),
      message('user', 'latest'),
    ];
    const abridged = fitState(messages, [], { ...fit, maxStateTokens: 1800, preserveRecentMessages: 1 });
    expect(abridged.stage).toBe('texts abridged');
    expect(abridged.tokens).toBeLessThanOrEqual(1800);
    expect(abridged.state.history[1]?.text).toContain('chars omitted');
    expect(abridged.state.history[0]?.text).toBe(long(0));
    expect(abridged.state.history[4]?.text).toBe('latest');

    const collapsed = fitState(messages, [], { ...fit, maxStateTokens: 420, preserveRecentMessages: 1 });
    expect(collapsed.stage).toBe('old messages collapsed');
    expect(collapsed.tokens).toBeLessThanOrEqual(420);
    expect(collapsed.state.history[1]?.text).toMatch(/^\[… \d+ chars omitted …\]$/);
    expect(collapsed.state.history[0]?.text).toContain('lorem');
    expect(collapsed.state.history[4]?.text).toBe('latest');
  });

  it('throws when the history cannot be fitted', () => {
    const messages = [message('user', 'a'.repeat(2000)), message('assistant', 'b')];
    expect(() => fitState(messages, [], { ...fit, maxStateTokens: 50 })).toThrow(/too large/);
  });
});

describe('question batching', () => {
  const calls: ToolCall[] = Array.from({ length: 10 }, (_, i) => ({
    id: `t${i + 1}`,
    tool_use_id: `tool-${i + 1}`,
    tool: 'Read',
    input: {},
    callIndex: i * 2 + 1,
    resultIndex: i * 2 + 2,
    resultChars: 100,
    isError: false,
    pinned: false,
  }));
  const options = { maxRequestTokens: 30_000, maxQuestionsPerRequest: 12 };

  it('splits only on the call cap when tokens leave plenty of room', () => {
    const batches = batchCalls(calls, 1000, options);
    expect(batches.map((b) => b.length)).toEqual([6, 4]);
    expect(batches.flat().map((c) => c.id)).toEqual(calls.map((c) => c.id));
  });

  it('caps a batch at 6 calls so a request never exceeds 12 questions (engine 30s fetch abort)', () => {
    const many: ToolCall[] = Array.from({ length: 75 }, (_, i) => ({ ...calls[0]!, id: `t${i}` }));
    const batches = batchCalls(many, 1000, options);
    expect(batches.map((b) => b.length)).toEqual([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3]);
    expect(batches.flat().map((c) => c.id)).toEqual(many.map((c) => c.id));
  });

  it('derives the call cap from maxQuestionsPerRequest (floor of half)', () => {
    const wide = { ...options, maxQuestionsPerRequest: 24 };
    const many: ToolCall[] = Array.from({ length: 30 }, (_, i) => ({ ...calls[0]!, id: `t${i}` }));
    expect(batchCalls(many, 1000, wide).map((b) => b.length)).toEqual([12, 12, 6]);
    const odd = { ...options, maxQuestionsPerRequest: 5 };
    expect(batchCalls(calls, 1000, odd).map((b) => b.length)).toEqual([2, 2, 2, 2, 2]);
  });

  it('bounds the message chunk size by the question cap and the request budget', () => {
    expect(messageChunkSize(1000, 30_000, 12)).toBe(12);
    expect(messageChunkSize(1000, 30_000, 24)).toBe(24);
    // ~28.9k room → 452 by token budget, 12 by cap
    expect(messageChunkSize(1000, 30_000)).toBe(12);
    expect(messageChunkSize(29_600, 30_000, 12)).toBeLessThanOrEqual(12);
  });

  it('splits questions across requests when the state leaves little room', () => {
    const batches = batchCalls(calls, 29_600, options);
    expect(batches.length).toBeGreaterThan(1);
    expect(batches.flat().map((c) => c.id)).toEqual(calls.map((c) => c.id));
  });

  it('throws when a single question does not fit', () => {
    expect(() => batchCalls(calls, 29_990, options)).toThrow(/no room/);
  });
});

describe('decisions', () => {
  const options = { keepThreshold: 0.5 };
  const unpinned = { id: 't1', tool: 'Read', pinned: false };

  it('keeps, drops the result, or drops the call based on the keep probabilities', () => {
    expect(decideCall(unpinned, { keepCall: 0.9, keepResult: 0.7 }, options).action).toBe('keep');
    expect(decideCall(unpinned, { keepCall: 0.9, keepResult: 0.2 }, options).action).toBe('drop_result');
    expect(decideCall(unpinned, { keepCall: 0.1, keepResult: 0.2 }, options).action).toBe('drop_call');
    expect(decideCall({ ...unpinned, pinned: true }, { keepCall: 0, keepResult: 0 }, options)).toMatchObject({
      action: 'keep',
      reason: 'pinned',
    });
  });

  it('never drops user prose, asked or not', () => {
    const entry = { i: 2, role: 'user' as const, text: 'instructions' };
    expect(decideMessage(entry, undefined, options)).toMatchObject({ keep: true, reason: 'user' });
    expect(decideMessage(entry, 0, options)).toMatchObject({ keep: true, reason: 'user' });
    expect(decideMessage({ ...entry, role: 'assistant' }, 0, options)).toMatchObject({
      keep: false,
      reason: 'dropped',
    });
  });

  it('removes dropped calls and truncates dropped results', () => {
    const messages = transcript();
    messages[4]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[5]!.toolResults![0]!.text = 'x'.repeat(2000);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.1, keepResult: 0.1 }, options),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.1 }, options),
      decideCall(calls[2]!, { keepCall: 0.9, keepResult: 0.9 }, options),
    ];
    const kept = applyDecisions(messages, decisions, calls, 300);

    expect(kept.map((m) => m.text || m.toolUses[0]?.tool_use_id || m.toolResults?.[0]?.tool_use_id)).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      'a.ts looks fine; checking b.ts',
      'tool-2',
      'tool-2',
      'tool-3',
      'tool-3',
      'The failure is in b.test.ts; fixing now.',
      'go ahead',
    ]);
    expect(kept[0]).toBe(messages[0]);
    expect(kept[2]).not.toBe(messages[4]);
    expect(kept[2]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-compaction truncated 1700 chars`),
    );
    expect(kept[3]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-compaction truncated 1700 chars`),
    );
    expect(kept[2]).not.toBe(messages[4]);
    expect(kept[3]).not.toBe(messages[5]);
    expect(kept[4]).toBe(messages[6]);
    expect(kept[5]?.toolResults?.[0]?.text).toContain('expected 2 to be 3');

    const shortMessages = transcript();
    shortMessages[4]!.toolUses[0]!.text = 'y'.repeat(100);
    shortMessages[5]!.toolResults![0]!.text = 'y'.repeat(100);
    const shortKept = applyDecisions(shortMessages, decisions, calls, 300);
    expect(shortKept[2]).toBe(shortMessages[4]);
    expect(shortKept[3]).toBe(shortMessages[5]);
  });

  it('honours truncateHeadChars, including a zero head', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 })];
    const original = messages[2]!.toolResults![0]!.text;
    const total = original.length;

    const kept = applyDecisions(messages, decisions, calls, 50);
    expect(kept[2]?.toolResults?.[0]?.text).toBe(
      `${original.slice(0, 50)}\n[fast-compaction truncated ${total - 50} chars of this tool result; re-run the tool if needed]`,
    );
    expect(kept[1]?.toolUses[0]?.text).toBe(kept[2]?.toolResults?.[0]?.text);

    const noHead = applyDecisions(messages, decisions, calls, 0);
    expect(noHead[2]?.toolResults?.[0]?.text).toBe(
      `[fast-compaction truncated ${total} chars of this tool result; re-run the tool if needed]`,
    );
  });
});

describe('compact', () => {
  it('resends the full state with every batch and merges the answers', async () => {
    const seen: Seen[] = [];
    const messages = transcript();
    const stateTokens = fitState(messages, collectToolCalls(messages, 1), {
      ...fit,
      goal: '',
      preserveRecentMessages: 1,
    }).tokens;
    const output = await compact(
      messages,
      fakeJev((name) => (name.startsWith('call_') ? 0.9 : 0.1), seen),
      { preserveRecentMessages: 1, maxRequestTokens: stateTokens + 150 },
    );

    expect(output.stats.requests).toBe(seen.length);
    expect(seen.length).toBeGreaterThan(1);
    expect(seen.flatMap((r) => r.questions).sort()).toEqual([
      'call_t1',
      'call_t2',
      'call_t3',
      'msg_3',
      'msg_8',
      'result_t1',
      'result_t2',
      'result_t3',
    ]);
    expect(new Set(seen.map((r) => JSON.stringify(r.state))).size).toBe(1);
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_result', 'drop_result', 'drop_result']);
    // the prose entries at i=3/i=8 (0.1) lose their prose; they keep nothing
    // but prose, so they vanish entirely (i=0 is seed-protected)
    expect(output.messages).toHaveLength(messages.length - 2);
    expect(output.stats).toMatchObject({ resultsDropped: 3, kept: 0, callsDropped: 0, pinned: 0 });
    expect(reductionRatio(output)).toBeGreaterThan(0);
  });

  it('sums judge-reported timing across all requests into stats.judge', async () => {
    const messages = transcript();
    let calls = 0;
    const judgeAsker: JevAsker = {
      async ask(state, questions: JevQuestions) {
        calls += 1;
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
          judge: { loadMs: calls === 1 ? 7000 : 0, promptMs: 900, promptTokens: 3000, genMs: 3000, genTokens: 250 },
        };
      },
    };
    const output = await compact(messages, judgeAsker, { preserveRecentMessages: 1 });
    expect(calls).toBeGreaterThan(1);
    expect(output.stats.judge).toEqual({
      requests: calls,
      loadMs: 7000,
      promptMs: 900 * calls,
      promptTokens: 3000 * calls,
      genMs: 3000 * calls,
      genTokens: 250 * calls,
    });
  });

  it('judges conversation messages and drops the prose of the droppable ones', async () => {
    const seen: Seen[] = [];
    const messages = [
      message('user', 'old context nobody needs'),
      call('tool-1', 'Read', { file_path: 'src/a.ts' }, 'x'),
      result('tool-1', 'file body'),
      message('assistant', 'working on it'),
      message('user', 'recent instruction'),
      message('user', 'latest question'),
    ];
    const output = await compact(
      messages,
      fakeJev((name) => (name === 'msg_3' ? 0 : 0.9), seen),
      { preserveRecentMessages: 2 },
    );
    // msg questions cover prose entries only, and never the seed/recent window
    expect(seen.flatMap((r) => r.questions).filter((q) => q.startsWith('msg_')).sort()).toEqual([
      'msg_3',
    ]);
    expect(output.messageDecisions).toEqual([
      expect.objectContaining({ i: 0, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 3, keep: false, reason: 'dropped' }),
      expect.objectContaining({ i: 4, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 5, keep: true, reason: 'user' }),
    ]);
    // the dropped assistant note vanishes entirely (prose only); the kept array shifts
    expect(output.messages).toHaveLength(5);
    expect(output.messages[0]?.text).toBe('old context nobody needs');
    expect(output.messages[3]?.text).toBe('recent instruction');
    expect(output.stats).toMatchObject({ messagesDropped: 1 });
  });

  it('never judges user prose: task instructions survive, assistant prose is judged', async () => {
    const seen: Seen[] = [];
    const messages = [
      message('user', 'seed'),
      message('user', 'old task instructions nobody may lose'),
      call('tool-1', 'Read', { file_path: 'src/a.ts' }, 'x'),
      result('tool-1', 'file body'),
      message('assistant', 'old rambling nobody needs'),
      message('user', 'recent instruction'),
      message('user', 'latest question'),
    ];
    const output = await compact(
      messages,
      fakeJev(() => 0, seen),
      { preserveRecentMessages: 2 },
    );
    // only the assistant entry is judgeable — user prose is never asked about
    expect(seen.flatMap((r) => r.questions).filter((q) => q.startsWith('msg_'))).toEqual(['msg_4']);
    expect(output.messageDecisions).toEqual([
      expect.objectContaining({ i: 0, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 1, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 4, keep: false, reason: 'dropped' }),
      expect.objectContaining({ i: 5, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 6, keep: true, reason: 'user' }),
    ]);
    expect(output.messages.some((m) => m.text === 'old task instructions nobody may lose')).toBe(true);
    expect(output.messages.some((m) => m.text === 'old rambling nobody needs')).toBe(false);
  });

  it('keeps everything without calling Jev when nothing is judgeable', async () => {
    const seen: Seen[] = [];
    const messages = [message('user', 'hello'), message('assistant', 'hi')];
    const output = await compact(messages, fakeJev(() => 0, seen));
    // both prose entries sit inside the recent window: protected, never asked
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ requests: 0, calls: 0, messagesDropped: 0 });
    expect(output.messages).toEqual(messages);
  });

  it('caps the state fit by the request budget so a batch of one always fits', async () => {
    // regression: a state allowed up to maxStateTokens (25000) grew past a
    // tuned maxRequestTokens of 19000 and starved every question batch
    const messages = [
      message('user', 'seed'),
      call('tool-fat', 'Read', { file_path: 'src/big.ts' }, 'x'),
      result('tool-fat', 'body '.repeat(6000)),
      message('user', 'later'),
      message('user', 'latest'),
    ];
    const output = await compact(messages, fakeJev(() => 0.9), {
      maxRequestTokens: 19_000,
      preserveRecentMessages: 1,
    });
    expect(output.stats.stateTokens).toBeLessThanOrEqual(19_000 - 20 - 320);
    expect(output.stats.requests).toBeGreaterThan(0);
  });

  it('keeps everything unjudged when the time budget is already spent', async () => {
    const seen: Seen[] = [];
    const messages = [
      message('user', 'seed'),
      call('tool-1', 'Read', { file_path: 'src/a.ts' }, 'x'),
      result('tool-1', 'body'),
      message('user', 'latest'),
    ];
    const output = await compact(messages, fakeJev(() => 0.9, seen), {
      timeBudgetMs: 0,
      preserveRecentMessages: 1,
    });
    expect(seen).toHaveLength(0);
    expect(output.stats.requests).toBe(0);
    expect(output.stats.partial).toBe(true);
    // nothing dropped: unanswered candidates stay
    expect(output.messages).toHaveLength(messages.length);
  });

  it('judges normally without a time budget', async () => {
    const seen: Seen[] = [];
    const messages = [
      message('user', 'seed'),
      call('tool-1', 'Read', { file_path: 'src/a.ts' }, 'x'),
      result('tool-1', 'body'),
      message('user', 'latest'),
    ];
    const output = await compact(messages, fakeJev(() => 0.9, seen), {
      preserveRecentMessages: 1,
    });
    expect(seen.length).toBeGreaterThan(0);
    expect(output.stats.partial).toBeUndefined();
  });

  it('survives a failing batch: its calls stay, the rest is still judged', async () => {
    const messages = [
      message('user', 'seed'),
      call('tool-1', 'Read', { file_path: 'src/a.ts' }, 'x'),
      result('tool-1', 'body'),
      call('tool-2', 'Read', { file_path: 'src/b.ts' }, 'x'),
      result('tool-2', 'body'),
      call('tool-3', 'Read', { file_path: 'src/c.ts' }, 'x'),
      result('tool-3', 'body'),
      message('user', 'latest'),
    ];
    let asks = 0;
    const asker: JevAsker = {
      async ask(state, questions: JevQuestions) {
        asks += 1;
        if (asks === 2) throw new Error('judge request failed (502): boom');
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0 }]),
          ),
        };
      },
    };
    const output = await compact(messages, asker, {
      maxQuestionsPerRequest: 4, // forces multiple batches
      preserveRecentMessages: 1,
      tolerateRequestFailures: true,
    });
    expect(output.stats.batchErrors).toBe(1);
    expect(output.stats.partial).toBe(true);
    // batches 1 and 3 were judged and dropped their calls; batch 2 stayed
    const actions = output.decisions.map((d) => d.action);
    expect(actions).toContain('drop_call');
    expect(actions).toContain('keep');
  });

  it('judges pure-chat prose even with no tool calls at all', async () => {
    const seen: Seen[] = [];
    const messages = [
      message('user', 'seed context'),
      message('assistant', 'old rambling nobody needs'),
      message('user', 'latest question'),
    ];
    const output = await compact(
      messages,
      fakeJev((name) => (name === 'msg_1' ? 0 : 0.9), seen),
      { preserveRecentMessages: 1 },
    );
    expect(seen.flatMap((r) => r.questions).sort()).toEqual(['msg_1']);
    expect(output.messageDecisions).toEqual([
      expect.objectContaining({ i: 0, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 1, keep: false, reason: 'dropped' }),
      expect.objectContaining({ i: 2, keep: true, reason: 'user' }),
    ]);
    expect(output.messages).toHaveLength(2);
    expect(output.messages[1]?.text).toBe('latest question');
    expect(output.stats).toMatchObject({ messagesDropped: 1, requests: 1 });
  });

  it('keeps a message whose answer the judge left out, labeled conservatively', async () => {
    const messages = [
      message('user', 'seed context'),
      message('assistant', 'old rambling nobody needs'),
      message('user', 'latest question'),
    ];
    const output = await compact(messages, {
      async ask(_state, questions: JevQuestions) {
        // answer everything except the msg_1 question
        const answers = Object.fromEntries(
          Object.keys(questions)
            .filter((key) => key !== 'msg_1')
            .map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
        );
        return { answers };
      },
    }, { preserveRecentMessages: 1 });
    expect(output.messageDecisions).toEqual([
      expect.objectContaining({ i: 0, keep: true, reason: 'user' }),
      expect.objectContaining({ i: 1, keep: true, reason: 'kept' }),
      expect.objectContaining({ i: 2, keep: true, reason: 'user' }),
    ]);
    expect(output.messages).toHaveLength(3);
  });

  it('reports a tiny reduction when Jev wants everything kept', async () => {
    const output = await compact(transcript(), fakeJev(() => 0.95), { preserveRecentMessages: 1 });
    expect(output.decisions.every((d) => d.action === 'keep')).toBe(true);
    expect(reductionRatio(output)).toBe(0);
  });

  it('rejects malformed answers', async () => {
    const broken: JevAsker = {
      ask: async () => ({ answers: { call_t1: { noul: 0.5 } } }),
    };
    await expect(compact(transcript(), broken, { preserveRecentMessages: 1 })).rejects.toThrow(
      /Invalid judge answer/,
    );
  });
});

describe('budget compaction', () => {
  it('demotes results, then calls, then prose, lowest score first, start and recent untouched', async () => {
    const messages = [
      message('user', 'seed context nobody may drop'),
      message('assistant', 'ack'),
      call('tool-1', 'Read', { file_path: 'a' }, 'x'),
      result('tool-1', 'x'.repeat(6000)),
      call('tool-2', 'Read', { file_path: 'b' }, 'x'),
      result('tool-2', 'x'.repeat(6000)),
      call('tool-3', 'Bash', { command: 'npm test' }, 'x'),
      result('tool-3', 'x'.repeat(6000)),
      message('assistant', 'prose one'),
      message('user', 'latest question'),
    ];
    const answer = (name: string): number =>
      name === 'result_t1' ? 0.6 : name === 'result_t3' ? 0.7 : name === 'msg_8' ? 0.55 : 0.9;
    // two truncated results free ~2×900 tokens, one only ~900: exactly t1 + t3 must go
    const output = await compact(messages, fakeJev(answer), {
      preserveRecentMessages: 1,
      targetTokens: transcriptTokens(messages) - 1500,
    });
    expect(output.decisions.map((d) => [d.id, d.action, d.reason])).toEqual([
      ['t1', 'drop_result', 'budget'],
      ['t2', 'keep', 'kept'],
      ['t3', 'drop_result', 'budget'],
    ]);
    const resultText = (id: string): string | undefined =>
      output.messages.flatMap((m) => m.toolResults ?? []).find((r) => r.tool_use_id === id)?.text;
    expect(resultText('tool-1')).toContain('[fast-compaction truncated');
    expect(resultText('tool-3')).toContain('[fast-compaction truncated');
    expect(resultText('tool-2')).toBe('x'.repeat(6000));
    expect(output.messages[0]).toBe(messages[0]);
    expect(output.messageDecisions.map((d) => [d.i, d.keep, d.reason])).toEqual([
      [0, true, 'user'],
      [1, true, 'recent'],
      [8, true, 'kept'],
      [9, true, 'user'],
    ]);
    expect(output.stats.budgetEvictions).toBe(2);
  });

  it('drops never-judged prose before judged prose under budget pressure', async () => {
    const messages = [
      message('user', 'seed context for the session'),
      message('assistant', 'filler line'),
      message('assistant', 'judged prose stays because high'),
      message('assistant', 'never judged prose drops first'),
      message('user', 'latest question'),
    ];
    const output = await compact(messages, {
      async ask(_state, questions: JevQuestions) {
        const answers = Object.fromEntries(
          Object.keys(questions)
            .filter((key) => key !== 'msg_3')
            .map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
        );
        return { answers };
      },
    }, { preserveRecentMessages: 1, targetTokens: transcriptTokens(messages) - 2 });
    expect(output.messageDecisions.map((d) => [d.i, d.keep, d.reason])).toEqual([
      [0, true, 'user'],
      [1, true, 'recent'],
      [2, true, 'kept'],
      [3, false, 'budget'],
      [4, true, 'user'],
    ]);
    expect(output.messages.map((m) => m.text)).toEqual([
      'seed context for the session',
      'filler line',
      'judged prose stays because high',
      'latest question',
    ]);
    expect(output.stats.budgetEvictions).toBe(1);
  });

  it('evicts everything evictable best-effort and leaves pinned messages intact', async () => {
    const messages = [
      message('user', 'seed'),
      message('assistant', 'early note'),
      call('tool-1', 'Read', { file_path: 'a' }, 'x'),
      result('tool-1', 'x'.repeat(600)),
      message('assistant', 'middle prose'),
      message('user', 'latest'),
    ];
    const output = await compact(messages, fakeJev(() => 0.9), {
      preserveRecentMessages: 1,
      targetTokens: 1,
    });
    expect(output.decisions.map((d) => [d.id, d.action, d.reason])).toEqual([
      ['t1', 'drop_call', 'budget'],
    ]);
    expect(output.messageDecisions.map((d) => [d.i, d.keep, d.reason])).toEqual([
      [0, true, 'user'],
      [1, true, 'recent'],
      [4, false, 'budget'],
      [5, true, 'user'],
    ]);
    expect(output.messages).toEqual([messages[0], messages[1], messages[5]]);
    expect(output.messages[0]).toBe(messages[0]);
    expect(output.stats.budgetEvictions).toBe(3);
  });

  it('leaves decisions untouched when the target already fits', async () => {
    const messages = transcript();
    const plain = await compact(messages, fakeJev(() => 0.95), { preserveRecentMessages: 1 });
    // preserveStartMessages 0 aligns the pinning, which the budget would otherwise extend
    const budgeted = await compact(messages, fakeJev(() => 0.95), {
      preserveRecentMessages: 1,
      preserveStartMessages: 0,
      targetTokens: 1_000_000,
    });
    expect(budgeted.decisions).toEqual(plain.decisions);
    expect(budgeted.messageDecisions).toEqual(plain.messageDecisions);
    expect(budgeted.stats.budgetEvictions).toBe(0);
  });
});

describe('HTTP client', () => {
  it('builds a System One request', () => {
    const request = buildJevRequest({ apiKey: 'k' }, { a: 1 }, {
      q: { type: 'noul', instructions: 'x' },
    });
    expect(request.url).toBe('https://api.typesafe.ai/v1/systemone');
    expect(request.headers.authorization).toBe('Bearer k');
    expect(JSON.parse(request.body)).toEqual({
      model: 'jev-latest',
      state: { a: 1 },
      questions: { q: { type: 'noul', instructions: 'x' } },
    });
  });

  it('rejects failed and malformed responses', () => {
    expect(() => parseJevResponse(500, false, 'boom')).toThrow(/500/);
    expect(() => parseJevResponse(200, true, 'not json')).toThrow(/malformed/);
    expect(() => parseJevResponse(200, true, '{}')).toThrow(/missing answers/);
    expect(parseJevResponse(200, true, '{"answers":{}}')).toEqual({ answers: {} });
  });

  it('asks over fetch and refuses to run without a key', async () => {
    const bodies: string[] = [];
    const client = new JevClient({
      apiKey: 'k',
      model: 'jev-test',
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(String(init?.body));
        return new Response(JSON.stringify({ answers: { q: { noul: 0.4 } } }), { status: 200 });
      }) as typeof fetch,
    });
    const response = await client.ask('state', { q: { type: 'noul', instructions: 'x' } });
    expect(response.answers.q).toEqual({ noul: 0.4 });
    expect(JSON.parse(bodies[0]!).model).toBe('jev-test');

    const keyless = new JevClient({ apiKey: '' });
    await expect(keyless.ask('s', {})).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactMessages(transcript(), { apiKey: '', preserveRecentMessages: 1 }),
    ).rejects.toThrow(/TYPESAFE_API_KEY/);
  });
});
