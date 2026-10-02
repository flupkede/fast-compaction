import { createRequire } from 'node:module';
import type { DatabaseSync as DatabaseSyncClass } from 'node:sqlite';
import type { Message, ToolUse } from '../types.js';

/**
 * OpenCode stores sessions in a SQLite database (`message` + `part` tables,
 * JSON `data` columns). Only `text` and `tool` parts carry conversation
 * content; step markers, reasoning, subtasks and files are skipped.
 */
export interface OpencodePart {
  type: string;
  text?: string;
  callID?: string;
  tool?: string;
  state?: {
    status?: string;
    input?: Record<string, unknown>;
    output?: unknown;
  };
}

export interface OpencodeMessageData {
  role?: string;
}

/** Caps a stored tool output before it enters a replayed transcript. */
const RESULT_CHARS = 20_000;

function resultText(output: unknown): string {
  const text = typeof output === 'string' ? output : JSON.stringify(output ?? '');
  return text.length > RESULT_CHARS ? text.slice(0, RESULT_CHARS) : text;
}

/**
 * Maps one stored OpenCode message to library messages. A tool part whose
 * state holds an output yields the tool use plus a synthetic user message
 * with the paired result — the library pairs calls with results by id.
 */
export function opencodeToMessages(role: string, parts: readonly OpencodePart[]): Message[] {
  const text = parts
    .filter((part) => part.type === 'text' && typeof part.text === 'string')
    .map((part) => part.text)
    .join('\n');
  const toolUses: ToolUse[] = parts
    .filter((part) => part.type === 'tool' && part.callID && part.tool)
    .map((part) => ({
      tool_use_id: part.callID!,
      tool: part.tool!,
      input: part.state?.input ?? {},
    }));
  const results = parts
    .filter(
      (part) =>
        part.type === 'tool' &&
        part.callID &&
        part.tool &&
        part.state &&
        part.state.output !== undefined &&
        part.state.output !== null,
    )
    .map((part) => ({
      tool_use_id: part.callID!,
      text: resultText(part.state?.output),
      ...(part.state?.status === 'error' ? { isError: true } : {}),
    }));
  const messages: Message[] = [];
  if (text.length > 0 || toolUses.length > 0) {
    messages.push({ role: role === 'assistant' ? 'assistant' : 'user', text, toolUses });
  }
  if (results.length > 0) {
    messages.push({ role: 'user', text: '', toolUses: [], toolResults: results });
  }
  return messages;
}

export interface OpencodeSessionInfo {
  id: string;
  title: string;
  directory: string;
}

export interface OpencodeTranscripts {
  /** Sessions with at least `minMessages` stored messages, newest first. */
  sessions(minMessages: number): OpencodeSessionInfo[];
  messages(id: string): Message[];
  close(): void;
}

/** Opens the OpenCode session database read-only. Requires Node >= 22.5. */
export async function openOpencodeTranscripts(dbPath: string): Promise<OpencodeTranscripts> {
  let DatabaseSync: typeof DatabaseSyncClass;
  try {
    // createRequire, not import: bundlers must not resolve node:sqlite
    ({ DatabaseSync } = createRequire(import.meta.url)('node:sqlite') as typeof import('node:sqlite'));
  } catch {
    throw new Error('reading OpenCode sessions requires Node >= 22.5 (node:sqlite)');
  }
  const db = new DatabaseSync(dbPath, { readOnly: true });
  const rows = (sql: string, ...params: (string | number)[]): Record<string, unknown>[] =>
    db.prepare(sql).all(...params) as Record<string, unknown>[];
  return {
    sessions(minMessages: number): OpencodeSessionInfo[] {
      return rows(
        `SELECT s.id, s.title, s.directory, COUNT(m.id) AS n
         FROM session s JOIN message m ON m.session_id = s.id
         GROUP BY s.id HAVING n >= ? ORDER BY s.time_created DESC`,
        minMessages,
      ).map((row) => ({
        id: String(row.id),
        title: String(row.title ?? ''),
        directory: String(row.directory ?? ''),
      }));
    },
    messages(id: string): Message[] {
      const out: Message[] = [];
      const messages = rows(
        'SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created, rowid',
        id,
      );
      for (const message of messages) {
        try {
          const data = JSON.parse(String(message.data)) as OpencodeMessageData;
          const parts = rows(
            'SELECT data FROM part WHERE message_id = ? ORDER BY time_created, rowid',
            String(message.id),
          ).map((part) => JSON.parse(String(part.data)) as OpencodePart);
          out.push(...opencodeToMessages(data.role ?? 'user', parts));
        } catch {
          // one unreadable message must not sink the session
        }
      }
      return out;
    },
    close(): void {
      db.close();
    },
  };
}
