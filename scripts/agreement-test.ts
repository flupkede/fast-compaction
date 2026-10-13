/**
 * Measures a candidate judge against the teacher's recorded answers on the
 * held-out validation split produced by prepare-training. The teacher's
 * answers are already in the val examples, so clef-mix is never re-asked.
 * Reports decision agreement plus false drops (teacher keep, judge drop —
 * the expensive error: lost context).
 *
 *   npm run agreement -- --base-url <shim-or-ollama-url> [--model name]
 *                        [--val <val.jsonl>] [--limit n] [--request-timeout-ms n]
 */
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { JevClient, OLLAMA_MODEL, OLLAMA_SYSTEM_ONE_URL } from '../src/index.js';
import type { JevQuestions, JevState } from '../src/index.js';
import { fetchWithTimeout, resolveRequestTimeoutMs, serializeAsker } from './replay-training.js';

export interface AgreementScores {
  questions: number;
  /** Fraction of questions where teacher and judge land on the same keep/drop side. */
  decisionAgreement: number;
  meanAbsError: number;
  /** Teacher said keep, judge said drop: the expensive error. */
  falseDrops: number;
  /** Teacher said drop, judge said keep: only costs compression. */
  falseKeeps: number;
  /** Questions the judge did not answer at all. */
  unanswered: number;
}

/** Compares one expected/got answer set at one threshold. Pure. */
export function scoreAgreement(
  expected: Record<string, { noul: number }>,
  got: Record<string, { noul?: number } | undefined>,
  threshold: number,
): AgreementScores {
  let agree = 0;
  let absError = 0;
  let falseDrops = 0;
  let falseKeeps = 0;
  let unanswered = 0;
  const questions = Object.keys(expected).length;
  for (const [id, teacher] of Object.entries(expected)) {
    const student = got[id]?.noul;
    if (student === undefined) {
      unanswered++;
      continue;
    }
    absError += Math.abs(teacher.noul - student);
    const teacherKeep = teacher.noul >= threshold;
    const studentKeep = student >= threshold;
    if (teacherKeep === studentKeep) agree++;
    else if (teacherKeep && !studentKeep) falseDrops++;
    else falseKeeps++;
  }
  return {
    questions,
    decisionAgreement: questions === 0 ? 1 : agree / questions,
    meanAbsError: questions === 0 ? 0 : absError / questions,
    falseDrops,
    falseKeeps,
    unanswered,
  };
}

function combine(scores: AgreementScores[]): AgreementScores {
  const questions = scores.reduce((sum, s) => sum + s.questions, 0);
  if (questions === 0) {
    return { questions: 0, decisionAgreement: 1, meanAbsError: 0, falseDrops: 0, falseKeeps: 0, unanswered: 0 };
  }
  return {
    questions,
    decisionAgreement: scores.reduce((sum, s) => sum + s.decisionAgreement * s.questions, 0) / questions,
    meanAbsError: scores.reduce((sum, s) => sum + s.meanAbsError * s.questions, 0) / questions,
    falseDrops: scores.reduce((sum, s) => sum + s.falseDrops, 0),
    falseKeeps: scores.reduce((sum, s) => sum + s.falseKeeps, 0),
    unanswered: scores.reduce((sum, s) => sum + s.unanswered, 0),
  };
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const stringFlag = (name: string): string | undefined => {
    const index = argv.indexOf(`--${name}`);
    return index === -1 || argv[index + 1] === undefined ? undefined : argv[index + 1];
  };
  const numberFlag = (name: string): number | undefined => {
    const value = stringFlag(name);
    return value === undefined ? undefined : Number(value);
  };
  const trainingDir = join(homedir(), '.fast-compaction', 'training');
  const valPath = stringFlag('val') ?? join(trainingDir, 'fine-tune', 'val.jsonl');
  const limit = numberFlag('limit');
  const requestTimeoutMs = numberFlag('request-timeout-ms');

  const asker = serializeAsker(
    new JevClient({
      baseUrl: stringFlag('base-url') ?? process.env.OLLAMA_SYSTEM_ONE_URL ?? OLLAMA_SYSTEM_ONE_URL,
      apiKey: 'ollama',
      model: stringFlag('model') ?? process.env.OLLAMA_MODEL ?? OLLAMA_MODEL,
      fetch: fetchWithTimeout(resolveRequestTimeoutMs({ requestTimeoutMs })),
    }),
  );

  const lines = (await readFile(valPath, 'utf8')).split(/\r?\n/).filter((line) => line.trim().length > 0);
  const examples = (limit === undefined ? lines : lines.slice(0, limit)).map((line) => {
    const parsed = JSON.parse(line) as { messages: { role: string; content: string }[] };
    const user = JSON.parse(parsed.messages.find((m) => m.role === 'user')!.content) as {
      state: JevState;
      questions: JevQuestions;
      keepThreshold: number;
    };
    const assistant = JSON.parse(parsed.messages.find((m) => m.role === 'assistant')!.content) as Record<
      string,
      { noul: number }
    >;
    return { user, assistant };
  });

  const scores: AgreementScores[] = [];
  let failed = 0;
  for (const [index, item] of examples.entries()) {
    try {
      const response = await asker.ask(item.user.state, item.user.questions);
      // every question in the fine-tune data is a noul question
      scores.push(
        scoreAgreement(item.assistant, response.answers as Record<string, { noul?: number }>, item.user.keepThreshold ?? 0.5),
      );
    } catch (error) {
      failed++;
      console.log(`example ${index + 1}/${examples.length} failed: ${error instanceof Error ? error.message : error}`);
    }
    if ((index + 1) % 10 === 0) console.log(`judged ${index + 1}/${examples.length}`);
  }
  const total = combine(scores);
  console.log(
    `\nagreement: ${scores.length} example(s) asked (${failed} failed), ${total.questions} question(s)\n` +
      `  decision agreement: ${(total.decisionAgreement * 100).toFixed(1)}%\n` +
      `  mean |noul error|:  ${total.meanAbsError.toFixed(3)}\n` +
      `  false drops:       ${total.falseDrops} (teacher keep -> judge drop)\n` +
      `  false keeps:       ${total.falseKeeps} (teacher drop -> judge keep)\n` +
      `  unanswered:        ${total.unanswered}`,
  );
}

if (process.argv[1] && process.argv[1].endsWith('agreement-test.ts')) {
  await main();
}
