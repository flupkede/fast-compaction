import { describe, expect, it } from 'vitest';
import { collectToolCalls, fitState, redactText, redactValue } from '../src/index.js';
import type { Message } from '../src/index.js';

describe('redactText', () => {
  it('replaces fixed-shape secrets', () => {
    expect(redactText('key AKIAABCDEFGHIJKLMNOP is live')).toBe(
      'key [REDACTED:aws-access-key] is live',
    );
    expect(redactText('Authorization: Bearer abc123.def456-ghi')).toBe(
      'Authorization: [REDACTED:bearer-token]',
    );
    expect(redactText('token ghp_' + 'a'.repeat(36))).toBe('token [REDACTED:github-token]');
  });

  it('replaces name=value and name: value assignments, keeping the name', () => {
    expect(redactText('password: hunter2live!')).toBe('password: [REDACTED:assigned-secret]');
    expect(redactText('API_KEY=sk-abcdef1234567890')).toBe(
      'API_KEY=[REDACTED:assigned-secret]',
    );
  });

  it('matches a credential name that is prefixed or camelCased, not just the bare word', () => {
    expect(redactText('export DB_PASSWORD=hunter2hunter')).toBe(
      'export DB_PASSWORD=[REDACTED:assigned-secret]',
    );
    expect(redactText('GITHUB_TOKEN=abcdef123456')).toBe('GITHUB_TOKEN=[REDACTED:assigned-secret]');
    expect(redactText('accessToken=abcdef123456')).toBe('accessToken=[REDACTED:assigned-secret]');
  });

  it('redacts a JSON-quoted key and a multi-word quoted value in full', () => {
    expect(redactText('{"password": "hunter2hunter"}')).toBe(
      '{"password": "[REDACTED:assigned-secret]"}',
    );
    expect(redactText('password: "correct horse battery"')).toBe(
      'password: "[REDACTED:assigned-secret]"',
    );
  });

  it('redacts a quoted value that itself contains a different quote character', () => {
    expect(redactText('password: "hunter2hunter\'s"')).toBe(
      'password: "[REDACTED:assigned-secret]"',
    );
    expect(redactText("password: 'Tr0ub4dor&3 don\\'t'")).not.toContain('Tr0ub4dor');
  });

  it('replaces a bare OpenAI/Anthropic-shaped key', () => {
    const key = 'sk-proj-abcdefghijklmnopqrstuvwxyz0123456789';
    expect(redactText(`export ANTHROPIC_API_KEY=${key}`)).toBe(
      'export ANTHROPIC_API_KEY=[REDACTED:llm-api-key]',
    );
    expect(redactText(`the key is ${key} for now`)).toBe(
      'the key is [REDACTED:llm-api-key] for now',
    );
  });

  it('leaves ordinary text untouched', () => {
    const text = 'Fix the failing test in src/generated, keep the goal in mind.';
    expect(redactText(text)).toBe(text);
  });
});

describe('redactValue', () => {
  it('redacts string leaves recursively without changing shape', () => {
    const input = {
      command: 'curl -H "Authorization: Bearer abc123.def456-ghi" https://api.example.com',
      env: { PASSWORD: 'hunter2live!' },
      files: ['a.ts', 'b.ts'],
    };
    const out = redactValue(input);
    expect(out.command).toContain('[REDACTED:bearer-token]');
    expect(out.env.PASSWORD).toBe('[REDACTED:assigned-secret]');
    expect(out.files).toEqual(['a.ts', 'b.ts']);
  });

  it('redacts a credential-shaped key regardless of casing, prefix, or numeric value', () => {
    const out = redactValue({
      env: { DB_PASSWORD: 'hunter2hunter', password: 12345678, api_token: 'zzzzzzzz' },
    });
    expect(out.env.DB_PASSWORD).toBe('[REDACTED:assigned-secret]');
    expect(out.env.password).toBe('[REDACTED:assigned-secret]');
    expect(out.env.api_token).toBe('[REDACTED:assigned-secret]');
  });
});

describe('state building never leaks secrets into the Jev state', () => {
  it('redacts message text and tool input before they reach fitState', () => {
    const messages: Message[] = [
      { role: 'user', text: 'Please deploy with token=ghp_' + 'b'.repeat(36), toolUses: [] },
      {
        role: 'assistant',
        text: '',
        toolUses: [
          {
            tool_use_id: 't1',
            tool: 'Bash',
            input: { command: 'export PASSWORD=hunter2live! && deploy.sh' },
          },
        ],
      },
      { role: 'user', text: '', toolResults: [{ tool_use_id: 't1', text: 'deployed' }], toolUses: [] },
    ];
    const calls = collectToolCalls(messages, 0);
    const fitted = fitState(messages, calls, { maxStateTokens: 25_000, preserveRecentMessages: 0, goal: '' });
    const serialised = JSON.stringify(fitted.state);
    expect(serialised).not.toContain('hunter2live!');
    expect(serialised).not.toContain('ghp_' + 'b'.repeat(36));
    expect(serialised).toContain('[REDACTED:');
  });
});
