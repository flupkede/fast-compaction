import { describe, expect, it } from 'vitest';
import {
  compact,
  createStripper,
  DEFAULT_STRIP,
  fitState,
  goalFromMessages,
  collectToolCalls,
  parseReplayArgs,
  replaySession,
  type JevAsker,
  type JevQuestions,
  type Message,
  type TrainingRecord,
} from '../src/index.js';

describe('createStripper', () => {
  it('tokenises each identifier class with stable per-class numbering', () => {
    const strip = createStripper();
    const once = strip.text(
      'read C:/work/repo/src/a.ts then https://api.example.com/v1 and mail me at dev@x.io guid 3f2504e0-4f89-11d3-9a0c-0305e82c3301 ip 10.0.0.42 host api.example.com',
    );
    expect(once).toBe(
      'read [PATH_1] then [URL_1] and mail me at [EMAIL_1] guid [GUID_1] ip [IP_1] host [DOMAIN_1]',
    );
    // the same value maps to the same token; a new value gets the next number
    const twice = strip.text('C:/work/repo/src/a.ts and C:/work/repo/src/b.ts');
    expect(twice).toBe('[PATH_1] and [PATH_2]');
  });

  it('applies blocklist terms as substrings, case-insensitively', () => {
    const strip = createStripper({ blocklist: ['Acme', 'ClientQA'] });
    expect(strip.text('fix the Acme connector for clientqa now')).toBe(
      'fix the [BLOCKED_1] connector for [BLOCKED_2] now',
    );
    expect(strip.text('AcmeBase repo name is caught too')).toBe('[BLOCKED_1]Base repo name is caught too');
  });

  it('leaves versions and plain words alone and can switch classes off', () => {
    const strip = createStripper();
    expect(strip.text('node v22.5.0 on 2 cores')).toBe('node v22.5.0 on 2 cores');
    const noPaths = createStripper({ paths: false });
    expect(noPaths.text('src/a.ts changed')).toBe('src/a.ts changed');
    expect(DEFAULT_STRIP.blocklist).toEqual([]);
  });

  it('keeps paths mid-sentence separate from following words and each other', () => {
    const strip = createStripper();
    expect(strip.text('edited src/auth.ts and src/config.ts today')).toBe(
      'edited [PATH_1] and [PATH_2] today',
    );
    expect(strip.text('read docs/README.md then src/a.ts')).toMatch(
      /^read \[PATH_\d+\] then \[PATH_\d+\]$/,
    );
  });

  it('strips nested values deeply', () => {
    const strip = createStripper();
    expect(
      strip.value({ file_path: 'src/a.ts', nested: { url: 'https://x.dev/y', n: 3 }, list: ['b/c.ts'] }),
    ).toEqual({ file_path: '[PATH_1]', nested: { url: '[URL_1]', n: 3 }, list: ['[PATH_2]'] });
  });
});

describe('stripping through the state pipeline', () => {
  const transcript: Message[] = [
    { role: 'user', text: 'Fix the Acme auth bug in src/auth.ts', toolUses: [] },
    {
      role: 'assistant',
      text: '',
      toolUses: [
        {
          tool_use_id: 't1',
          tool: 'Edit',
          input: { file_path: 'src/auth.ts', old_string: 'password=hunter2', new_string: 'password=x' },
        },
      ],
    },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 't1', text: 'ok'.repeat(500) }] },
    { role: 'assistant', text: 'edited the Acme auth', toolUses: [] },
    { role: 'user', text: 'done', toolUses: [] },
    { role: 'assistant', text: 'bye', toolUses: [] },
    { role: 'user', text: 'thanks', toolUses: [] },
    { role: 'assistant', text: 'end', toolUses: [] },
    { role: 'user', text: 'x', toolUses: [] },
    { role: 'assistant', text: 'y', toolUses: [] },
  ];

  const strip = createStripper({ blocklist: ['Acme'] });

  it('strips texts, tool inputs and the derived goal; secrets stay redacted', () => {
    const options = { maxStateTokens: 25_000, preserveRecentMessages: 0, goal: '', strip };
    const calls = collectToolCalls(transcript, 0, strip);
    expect(calls[0]!.input).toEqual({
      file_path: '[PATH_1]',
      old_string: 'password=[REDACTED:assigned-secret]',
      new_string: 'password=x',
    });
    const fitted = fitState(transcript, calls, options);
    const texts = fitted.state.history.map((entry) => entry.text);
    expect(texts.some((text) => text.includes('[BLOCKED_1]'))).toBe(true);
    expect(texts.some((text) => text.includes('[PATH_1]'))).toBe(true);
    expect(JSON.stringify(fitted.state)).not.toContain('Acme');
    expect(JSON.stringify(fitted.state)).not.toContain('auth.ts');
    expect(goalFromMessages(transcript, strip)).not.toContain('Acme');
  });

  it('is off by default for the library and on for replay via options', async () => {
    const records: TrainingRecord[] = [];
    const asker: JevAsker = {
      async ask(_state, questions: JevQuestions) {
        return {
          answers: Object.fromEntries(
            Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: 0.9 }]),
          ),
        };
      },
    };
    await compact(transcript.slice(0, 2).concat(transcript.slice(2)), asker, {
      preserveRecentMessages: 0,
      trainingLog: (record) => void records.push(record),
    });
    expect(JSON.stringify(records[0]!.state)).toContain('Acme');

    records.length = 0;
    await replaySession(transcript, asker, (record) => void records.push(record), {
      preserveRecentMessages: 0,
      strip: { blocklist: ['Acme'] },
    });
    expect(JSON.stringify(records[0]!.state)).not.toContain('Acme');
  });
});

describe('parseReplayArgs strip flags', () => {
  it('reads --no-strip and --blocklist', async () => {
    const { parseReplayArgs: parse } = await import('../src/index.js');
    const bare = parse([]);
    expect(bare).not.toHaveProperty('noStrip');
    expect(bare).not.toHaveProperty('blocklistPath');
    expect(parse(['--no-strip'])).toMatchObject({ noStrip: true });
    expect(parse(['--blocklist', 'C:/terms.txt'])).toMatchObject({ blocklistPath: 'C:/terms.txt' });
    expect(() => parse(['--blocklist'])).toThrow(/needs a value/);
  });
});
