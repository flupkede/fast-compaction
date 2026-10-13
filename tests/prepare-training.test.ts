import { describe, expect, it } from 'vitest';
import {
  JUDGE_SYSTEM_PROMPT,
  callTarget,
  splitOf,
  toFineTuneExamples,
} from '../scripts/prepare-training.ts';
import { scoreAgreement } from '../scripts/agreement-test.ts';
import type { TrainingRecord } from '../src/index.js';

function record(overrides: Partial<TrainingRecord> = {}): TrainingRecord {
  return {
    schema: 2,
    ts: '2026-10-13T00:00:00.000Z',
    source: 'replay',
    state: { context: 'c', goal: 'g', history: [] },
    stateTokens: 10,
    stateStage: 'full',
    questions: {
      call_t1: { type: 'noul', instructions: 'call t1 matters' },
      result_t1: { type: 'noul', instructions: 'result t1 matters' },
    },
    answers: { t1: { keepCall: 0.9, keepResult: 0.2 } },
    decisions: [],
    keepThreshold: 0.5,
    messageQuestions: { msg_3: { type: 'noul', instructions: 'msg 3 matters' } },
    messageAnswers: { msg_3: 0.8 },
    ...overrides,
  };
}

describe('prepare-training', () => {
  it('turns one record into a call example and a message example with the teacher targets', () => {
    const examples = toFineTuneExamples(record());
    expect(examples).toHaveLength(2);
    for (const example of examples) {
      expect(example.messages[0]).toEqual({ role: 'system', content: JUDGE_SYSTEM_PROMPT });
      const user = JSON.parse(example.messages[1]!.content) as {
        state: unknown;
        questions: Record<string, unknown>;
        keepThreshold: number;
      };
      expect(user.keepThreshold).toBe(0.5);
      expect(Object.keys(user.questions).length).toBeGreaterThan(0);
    }
    const callTargetJson = JSON.parse(examples[0]!.messages[2]!.content) as Record<string, { noul: number }>;
    expect(callTargetJson.call_t1!.noul).toBeCloseTo(0.9);
    expect(callTargetJson.result_t1!.noul).toBeCloseTo(0.2);
    const msgTarget = JSON.parse(examples[1]!.messages[2]!.content) as Record<string, { noul: number }>;
    expect(msgTarget.msg_3!.noul).toBeCloseTo(0.8);
  });

  it('emits no example for an empty question set', () => {
    expect(toFineTuneExamples(record({ questions: {}, answers: {}, messageQuestions: undefined }))).toHaveLength(0);
  });

  it('splits stably per state and honours the fractions', () => {
    const state = { history: [{ i: 1, role: 'assistant', text: 'stable split' }] };
    const outcomes = new Set(Array.from({ length: 50 }, () => splitOf(state, 0.05)));
    expect(outcomes.size).toBe(1);
    expect(splitOf(state, 0)).toBe('train');
    expect(splitOf(state, 1)).toBe('val');
  });
});

describe('scoreAgreement', () => {
  it('counts agreement, false drops and false keeps at the threshold', () => {
    const scores = scoreAgreement(
      { a: { noul: 0.9 }, b: { noul: 0.1 }, c: { noul: 0.7 } },
      { a: { noul: 0.85 }, b: { noul: 0.6 }, c: undefined },
      0.5,
    );
    expect(scores.questions).toBe(3);
    expect(scores.decisionAgreement).toBeCloseTo(1 / 3);
    expect(scores.falseDrops).toBe(0);
    expect(scores.falseKeeps).toBe(1);
    expect(scores.unanswered).toBe(1);
    expect(scores.meanAbsError).toBeCloseTo((0.05 + 0.5) / 3);
  });

  it('flags the expensive error: teacher keep, judge drop', () => {
    const scores = scoreAgreement({ x: { noul: 0.8 } }, { x: { noul: 0.3 } }, 0.5);
    expect(scores.falseDrops).toBe(1);
    expect(scores.decisionAgreement).toBe(0);
  });
});

describe('callTarget', () => {
  it('maps every call to its two noul questions', () => {
    expect(callTarget({ t7: { keepCall: 0.4, keepResult: 0.6 } })).toEqual({
      call_t7: { type: 'noul', noul: 0.4 },
      result_t7: { type: 'noul', noul: 0.6 },
    });
  });
});
