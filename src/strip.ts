/**
 * Identifier pseudonymisation for states sent to Jev. Unlike `redact.ts`
 * (secret-shaped values), this replaces identifying *references* — paths,
 * URLs, e-mails, GUIDs, IPs, domains and blocklisted terms — with stable
 * tokens (`[PATH_1]`, `[URL_2]`, …), so the same value keeps the same token
 * within one state and Jev's relevance judgment stays coherent.
 */
export interface StripConfig {
  paths?: boolean;
  urls?: boolean;
  emails?: boolean;
  guids?: boolean;
  ips?: boolean;
  domains?: boolean;
  /** Literal terms (client names, project codenames), case-insensitive. */
  blocklist?: string[];
}

export const DEFAULT_STRIP: Required<StripConfig> = {
  paths: true,
  urls: true,
  emails: true,
  guids: true,
  ips: true,
  domains: true,
  blocklist: [],
};

export interface Stripper {
  text(input: string): string;
  value<T>(input: T): T;
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** A reserving replacer: every distinct value maps to one stable numbered token. */
class TokenMap {
  private readonly tokens = new Map<string, string>();

  constructor(private readonly label: string) {}

  replace(text: string, pattern: RegExp): string {
    return text.replace(pattern, (value: string) => {
      let token = this.tokens.get(value);
      if (token === undefined) {
        token = `[${this.label}_${this.tokens.size + 1}]`;
        this.tokens.set(value, token);
      }
      return token;
    });
  }
}

/**
 * Windows and POSIX paths, absolute or relative with at least one separator.
 * Space is excluded from the tail: a path mid-sentence must not swallow the
 * words after it — "Program Files" style names only match when absolute.
 * Dates like 2026/10/02 match too; over-tokenising is accepted, coherence is
 * kept.
 */
const PATH_PATTERN = /(?<![\w.])[A-Za-z]:[\\/][^\s"'`,;)\]}]+|(?<![\w.])[\w][\w.-]*(?:[\\/][\w.-]+){1,}/g;
const URL_PATTERN = /\bhttps?:\/\/[^\s"'`<>)\]}]+/gi;
const EMAIL_PATTERN = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g;
const GUID_PATTERN = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;
const IP_PATTERN = /\b(?:\d{1,3}\.){3}\d{1,3}\b/g;
/** Domain-shaped names over a short TLD list; avoids versions like 22.5.0. */
const DOMAIN_PATTERN =
  /\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|ai|dev|be|nl|eu|co|uk|de|fr|app|cloud|info|gov|edu)\b(?:\.[a-z]{2})?\b/gi;

export function createStripper(config: StripConfig = {}): Stripper {
  const resolved = { ...DEFAULT_STRIP, ...config };
  const paths = new TokenMap('PATH');
  const urls = new TokenMap('URL');
  const emails = new TokenMap('EMAIL');
  const guids = new TokenMap('GUID');
  const ips = new TokenMap('IP');
  const domains = new TokenMap('DOMAIN');
  const blocked = new TokenMap('BLOCKED');
  // substring, not word-boundary: "Acme" must also catch "AcmeBase" —
  // over-redaction is safe, a missed client name is not
  const blocklist = resolved.blocklist
    .filter((term) => term.trim().length > 0)
    .map((term) => new RegExp(escapeRegExp(term.trim()), 'gi'));

  const text = (input: string): string => {
    let out = input;
    // urls before paths and domains: a URL also path- and domain-shapes
    if (resolved.urls) out = urls.replace(out, URL_PATTERN);
    if (resolved.emails) out = emails.replace(out, EMAIL_PATTERN);
    if (resolved.guids) out = guids.replace(out, GUID_PATTERN);
    if (resolved.ips) out = ips.replace(out, IP_PATTERN);
    if (resolved.paths) out = paths.replace(out, PATH_PATTERN);
    if (resolved.domains) out = domains.replace(out, DOMAIN_PATTERN);
    for (const pattern of blocklist) out = blocked.replace(out, pattern);
    return out;
  };

  const value = <T>(input: T): T => {
    if (typeof input === 'string') return text(input) as unknown as T;
    if (Array.isArray(input)) return input.map((item) => value(item)) as unknown as T;
    if (input !== null && typeof input === 'object') {
      const out: Record<string, unknown> = {};
      for (const [key, val] of Object.entries(input as Record<string, unknown>)) {
        out[key] = value(val);
      }
      return out as unknown as T;
    }
    return input;
  };

  return { text, value };
}
