/**
 * System One-compatible shim for a chat-trained student judge.
 *
 * Ollama's /v1/systemone route only serves decision-capable (clef-arch)
 * models; a fine-tuned chat student needs this translation layer:
 * POST {state, questions} -> JUDGE_SYSTEM_PROMPT + payload -> ollama
 * /api/chat -> noul JSON -> {answers}. Run it next to any ollama:
 *
 *   npx tsx scripts/systemone-shim.ts --ollama http://ollama.example:11434
 *   npm run agreement -- --base-url http://localhost:11435/v1/systemone --model student
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { JUDGE_SYSTEM_PROMPT } from './prepare-training.js';
import type { JevQuestions, JevResponse, JevState } from '../src/index.js';

export interface ShimOptions {
  port: number;
  ollamaUrl: string;
  model: string;
  numCtx: number;
}

const DEFAULTS: ShimOptions = {
  port: 11435,
  ollamaUrl: 'http://127.0.0.1:11434',
  model: 'fast-compaction',
  numCtx: 24576,
};

/** Strips Qwen3 reasoning blocks and any prose around the first/last brace. */
export function extractJson(text: string): unknown {
  const clean = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  const start = clean.indexOf('{');
  const end = clean.lastIndexOf('}');
  if (start === -1 || end <= start) throw new Error('student reply has no JSON object');
  return JSON.parse(clean.slice(start, end + 1));
}

/** Maps the student's reply onto every question id; missing/unparseable -> 0.5. */
export function toAnswers(questions: JevQuestions, reply: unknown): JevResponse {
  const parsed = typeof reply === 'object' && reply !== null ? (reply as Record<string, unknown>) : {};
  const answers: JevResponse['answers'] = {};
  for (const id of Object.keys(questions)) {
    const value = (parsed as Record<string, { noul?: unknown }>)[id]?.noul;
    const noul = typeof value === 'number' && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0.5;
    answers[id] = { type: 'noul', noul };
  }
  return { answers };
}

/** Judge timing from an ollama chat response; undefined when it reports none. */
export function judgeMetrics(data: Record<string, unknown>): JevResponse['judge'] {
  const ms = (value: unknown): number | undefined =>
    typeof value === 'number' && Number.isFinite(value) ? Math.round(value / 1e6) : undefined;
  const count = (value: unknown): number | undefined =>
    typeof value === 'number' && Number.isFinite(value) ? value : undefined;
  const timing: NonNullable<JevResponse['judge']> = {};
  const loadMs = ms(data.load_duration);
  if (loadMs !== undefined) timing.loadMs = loadMs;
  const promptMs = ms(data.prompt_eval_duration);
  if (promptMs !== undefined) timing.promptMs = promptMs;
  const promptTokens = count(data.prompt_eval_count);
  if (promptTokens !== undefined) timing.promptTokens = promptTokens;
  const genMs = ms(data.eval_duration);
  if (genMs !== undefined) timing.genMs = genMs;
  const genTokens = count(data.eval_count);
  if (genTokens !== undefined) timing.genTokens = genTokens;
  return Object.keys(timing).length > 0 ? timing : undefined;
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => (data += chunk));
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

/** One response exit: ignores double writes when a client vanished mid-request. */
function respond(res: ServerResponse, status: number, body: string): void {
  if (res.headersSent || res.destroyed) return;
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(body);
}

export function createShimServer(options: Partial<ShimOptions>, fetcher: typeof fetch = fetch): Server {
  const config = { ...DEFAULTS, ...options };
  return createServer(async (req: IncomingMessage, res: ServerResponse) => {
    res.on('error', () => undefined);
    if (req.method !== 'POST') {
      respond(res, 200, 'systemone-shim: POST {state, questions} here\n');
      return;
    }
    try {
      const body = JSON.parse(await readBody(req)) as {
        model?: string;
        state: JevState;
        questions: JevQuestions;
        keepThreshold?: number;
      };
      const model = body.model || config.model;
      const messages = [
        { role: 'system', content: JUDGE_SYSTEM_PROMPT },
        {
          role: 'user',
          content: JSON.stringify({
            state: body.state,
            questions: body.questions,
            keepThreshold: body.keepThreshold ?? 0.5,
          }),
        },
      ];
      const upstream = await fetcher(`${config.ollamaUrl}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model,
          messages,
          stream: false,
          options: { temperature: 0, num_ctx: config.numCtx },
        }),
      });
      if (!upstream.ok) {
        respond(res, 502, JSON.stringify({ error: `ollama: ${await upstream.text()}` }));
        return;
      }
      const data = (await upstream.json()) as { message?: { content?: string } } & Record<string, unknown>;
      const reply = data.message?.content ?? '';
      const judge = judgeMetrics(data);
      const payload: Record<string, unknown> = { ...toAnswers(body.questions, extractJson(reply)) };
      if (judge) payload.judge = judge;
      respond(res, 200, JSON.stringify(payload));
    } catch (error) {
      respond(res, 502, JSON.stringify({ error: String(error) }));
    }
  });
}

function stringFlag(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? undefined : process.argv[index + 1];
}

if (process.argv[1] && process.argv[1].endsWith('systemone-shim.ts')) {
  const port = Number(stringFlag('port') ?? DEFAULTS.port);
  const ollamaUrl = stringFlag('ollama') ?? DEFAULTS.ollamaUrl;
  const model = stringFlag('model') ?? process.env.OLLAMA_MODEL ?? DEFAULTS.model;
  const numCtx = Number(stringFlag('num-ctx') ?? DEFAULTS.numCtx);
  createShimServer({ port, ollamaUrl, model, numCtx }).listen(port, () => {
    console.log(`systemone-shim: listening on :${port}, ollama ${ollamaUrl}, model ${model}, num_ctx ${numCtx}`);
  });
}
