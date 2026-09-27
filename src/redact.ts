/**
 * Secret-shaped substrings, scrubbed from everything before it reaches Jev.
 * `TYPESAFE_API_KEY` itself never enters the state (it only goes in the
 * request header, see `request.ts`); this catches secrets a user or tool
 * happens to be carrying in message text or tool input/output.
 */
export const SECRET_PATTERNS: ReadonlyArray<{ label: string; pattern: RegExp }> = [
  {
    label: 'private-key',
    pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----/g,
  },
  { label: 'jwt', pattern: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g },
  { label: 'llm-api-key', pattern: /\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b/g },
  { label: 'aws-access-key', pattern: /\bAKIA[0-9A-Z]{16}\b/g },
  { label: 'github-token', pattern: /\bgh[pousr]_[A-Za-z0-9]{36,255}\b/g },
  { label: 'npm-token', pattern: /\bnpm_[A-Za-z0-9]{36}\b/g },
  { label: 'slack-token', pattern: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g },
  { label: 'google-api-key', pattern: /\bAIza[0-9A-Za-z_-]{35}\b/g },
  { label: 'stripe-key', pattern: /\bsk_live_[0-9a-zA-Z]{20,}\b/g },
  { label: 'bearer-token', pattern: /\bBearer\s+[A-Za-z0-9\-_.=]{10,}/gi },
  { label: 'basic-auth', pattern: /\bBasic\s+[A-Za-z0-9+/=]{10,}/gi },
];

/**
 * Credential-shaped name fragment, matched as a *substring* (not an anchored
 * whole word) so `DB_PASSWORD`, `GITHUB_TOKEN`, `ANTHROPIC_API_KEY`,
 * `apiKey`, `accessToken` all match through their prefix/camelCase, at the
 * cost of also catching a harmless name like `tokenizer` — an acceptable
 * over-redaction given the alternative is a missed real secret.
 */
const SECRET_WORD =
  /(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?key|client[_-]?secret|account[_-]?key|shared[_-]?access[_-]?key|token|password|passwd|pwd|authorization)/i;

function isSecretName(name: string): boolean {
  return SECRET_WORD.test(name);
}

/**
 * `name: value` / `name=value` / JSON `"name": "value"` in free text. The key
 * may be bare or quoted (with the same quote closing it); the value may be
 * quoted (consumed up to its matching closing quote, so multi-word values are
 * fully covered) or bare (stopped at whitespace/punctuation). Whether a match
 * is redacted depends on `isSecretName` on the captured name, not on the
 * pattern alone. The bare-value class excludes `[` and `]` so a value already
 * replaced by a `SECRET_PATTERNS` marker is never re-matched and re-labelled.
 */
const ASSIGNED_SECRET = new RegExp(
  '(?<q1>["\'`]?)(?<name>[A-Za-z][A-Za-z0-9_-]{1,40})\\k<q1>\\s*[:=]\\s*' +
    // qval excludes only the matching closing quote (via lookahead), not every
    // quote character, so a value containing an apostrophe or backtick next to
    // a different delimiter (`"hunter2hunter's"`) is still consumed in full.
    '(?:(?<q2>["\'`])(?<qval>(?:(?!\\k<q2>)[\\s\\S]){1,300})\\k<q2>|(?<val>[^\\s"\'`,;)}\\]\\[]{3,300}))',
  'g',
);

type AssignedGroups = { name?: string; q2?: string; qval?: string; val?: string };

/** Replaces secret-shaped substrings in free text with a `[REDACTED:label]` marker. */
export function redactText(text: string): string {
  let result = text;
  for (const { label, pattern } of SECRET_PATTERNS) {
    result = result.replace(pattern, `[REDACTED:${label}]`);
  }
  return result.replace(ASSIGNED_SECRET, (match: string, ...args: unknown[]) => {
    const groups = args[args.length - 1] as AssignedGroups | undefined;
    if (!groups?.name || !isSecretName(groups.name)) return match;
    if (groups.qval !== undefined) {
      const keyPart = match.slice(0, match.length - groups.qval.length - (groups.q2?.length ?? 0));
      return `${keyPart}[REDACTED:assigned-secret]${groups.q2 ?? ''}`;
    }
    if (groups.val !== undefined) {
      const keyPart = match.slice(0, match.length - groups.val.length);
      return `${keyPart}[REDACTED:assigned-secret]`;
    }
    return match;
  });
}

/**
 * Deep-redacts every string leaf of a JSON-serialisable value; used on tool
 * inputs. A primitive (string, number or boolean) held under a
 * credential-shaped key is replaced outright — a numeric or unquoted-looking
 * password is still a password.
 */
export function redactValue<T>(value: T): T {
  if (typeof value === 'string') return redactText(value) as unknown as T;
  if (Array.isArray(value)) return value.map((item) => redactValue(item)) as unknown as T;
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value as Record<string, unknown>)) {
      const isPrimitive = typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean';
      out[key] = isPrimitive && isSecretName(key) ? '[REDACTED:assigned-secret]' : redactValue(val);
    }
    return out as unknown as T;
  }
  return value;
}
