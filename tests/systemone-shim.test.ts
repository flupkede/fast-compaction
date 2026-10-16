import { describe, expect, it } from 'vitest';
import { createShimServer, extractJson, judgeMetrics, toAnswers } from '../scripts/systemone-shim.js';
import type { AddressInfo } from 'node:net';

describe('judgeMetrics', () => {
  it('maps ollama eval metrics onto ms/token fields', () => {
    expect(judgeMetrics({
      load_duration: 7_360_000_000,
      prompt_eval_duration: 930_123_456,
      prompt_eval_count: 3102,
      eval_duration: 3_310_000_000,
      eval_count: 266,
    })).toEqual({ loadMs: 7360, promptMs: 930, promptTokens: 3102, genMs: 3310, genTokens: 266 });
  });
  it('keeps partial metrics and omits missing ones', () => {
    expect(judgeMetrics({ eval_duration: 800_000_000 })).toEqual({ genMs: 800 });
  });
  it('returns undefined when nothing is reported', () => {
    expect(judgeMetrics({})).toBeUndefined();
    expect(judgeMetrics({ total_duration: 'nonsense' })).toBeUndefined();
  });
});

describe('extractJson', () => {
  it('parses plain JSON', () => {
    expect(extractJson('{"a":1}')).toEqual({ a: 1 });
  });
  it('strips a reasoning block and prose', () => {
    expect(extractJson('<think>\nabc</think>\nSure! {"a":{"noul":0.9}} done')).toEqual({ a: { noul: 0.9 } });
  });
  it('throws without braces', () => {
    expect(() => extractJson('no json here')).toThrow();
  });
});

describe('toAnswers', () => {
  const questions = {
    call_1: { type: 'noul', instructions: 'i' },
    msg_2: { type: 'noul', instructions: 'j' },
  } as const;
  it('maps reply values and clamps to [0,1]', () => {
    const got = toAnswers(questions, { call_1: { noul: 1.7 }, msg_2: { noul: 0.2 } });
    expect(got.answers.call_1).toEqual({ type: 'noul', noul: 1 });
    expect(got.answers.msg_2).toEqual({ type: 'noul', noul: 0.2 });
  });
  it('falls back to 0.5 for missing or invalid ids', () => {
    const got = toAnswers(questions, { call_1: { noul: 'high' } });
    expect(got.answers.call_1).toEqual({ type: 'noul', noul: 0.5 });
    expect(got.answers.msg_2).toEqual({ type: 'noul', noul: 0.5 });
  });
});

describe('createShimServer', () => {
  it('translates a systemone request into an ollama chat call', async () => {
    let captured: { url: string; body: string } | undefined;
    const fetcher = (async (url: unknown, init?: { body?: string }) => {
      captured = { url: String(url), body: init?.body ?? '' };
      return {
        ok: true,
        json: async () => ({ message: { content: '{"call_1":{"noul":0.9}}' } }),
      } as Response;
    }) as typeof fetch;
    const server = createShimServer({ port: 0 }, fetcher);
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;
    try {
      const response = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ state: 's', questions: { call_1: { type: 'noul', instructions: 'i' } } }),
      });
      expect(response.status).toBe(200);
      expect(await response.json()).toEqual({ answers: { call_1: { type: 'noul', noul: 0.9 } } });
      expect(captured?.url).toBe('http://127.0.0.1:11434/api/chat');
      const sent = JSON.parse(captured!.body);
      expect(sent.model).toBe('fast-compaction');
      expect(sent.stream).toBe(false);
      expect(sent.messages[0].role).toBe('system');
      expect(JSON.parse(sent.messages[1].content)).toEqual({
        state: 's',
        questions: { call_1: { type: 'noul', instructions: 'i' } },
        keepThreshold: 0.5,
      });
    } finally {
      server.close();
    }
  });

  it('forwards judge timing when the upstream reports it', async () => {
    const fetcher = (async () => ({
      ok: true,
      json: async () => ({
        message: { content: '{"call_1":{"noul":0.9}}' },
        load_duration: 7_360_000_000,
        prompt_eval_duration: 900_000_000,
        prompt_eval_count: 3000,
        eval_duration: 3_000_000_000,
        eval_count: 250,
      }),
    })) as unknown as typeof fetch;
    const server = createShimServer({ port: 0 }, fetcher);
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;
    try {
      const response = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ state: 's', questions: { call_1: { type: 'noul', instructions: 'i' } } }),
      });
      expect(await response.json()).toEqual({
        answers: { call_1: { type: 'noul', noul: 0.9 } },
        judge: { loadMs: 7360, promptMs: 900, promptTokens: 3000, genMs: 3000, genTokens: 250 },
      });
    } finally {
      server.close();
    }
  });

  it('omits the judge field when the upstream reports no timing', async () => {
    const fetcher = (async () => ({
      ok: true,
      json: async () => ({ message: { content: '{"call_1":{"noul":0.9}}' } }),
    })) as unknown as typeof fetch;
    const server = createShimServer({ port: 0 }, fetcher);
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;
    try {
      const response = await fetch(`http://127.0.0.1:${port}/v1/systemone`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ state: 's', questions: { call_1: { type: 'noul', instructions: 'i' } } }),
      });
      expect(await response.json()).toEqual({ answers: { call_1: { type: 'noul', noul: 0.9 } } });
    } finally {
      server.close();
    }
  });
});
