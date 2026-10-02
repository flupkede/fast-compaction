import { readFile } from 'node:fs/promises';
import type { Message, ToolUse } from '../types.js';

/**
 * Claude Code stores each session as a JSONL transcript under
 * `~/.claude/projects/<project>/<session>.jsonl`; only the `user` and
 * `assistant` lines carry conversation content, the rest is metadata.
 */
interface TranscriptLine {
  type: string;
  message?: { content?: string | Block[] };
}

type Block = {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | Block[];
  is_error?: boolean;
};

function blockText(blocks: Block[] | undefined): string {
  return (blocks ?? [])
    .filter((block) => block.type === 'text' && typeof block.text === 'string')
    .map((block) => block.text)
    .join('\n');
}

/** The text of a message body: a plain string or its text blocks joined. */
function contentText(content: string | Block[] | undefined): string {
  return typeof content === 'string' ? content : blockText(content);
}

function toolResultText(content: string | Block[] | undefined): string {
  return contentText(content);
}

/** Parses a Claude Code JSONL transcript into client-agnostic messages. */
export function parseClaudeTranscript(text: string): Message[] {
  const messages: Message[] = [];
  for (const line of text.split('\n')) {
    if (line.trim().length === 0) continue;
    let parsed: TranscriptLine;
    try {
      parsed = JSON.parse(line) as TranscriptLine;
    } catch {
      continue;
    }
    if (parsed.type !== 'user' && parsed.type !== 'assistant') continue;
    const content = parsed.message?.content;
    const blocks = Array.isArray(content) ? content : [];
    const toolUses: ToolUse[] = blocks
      .filter((block) => block.type === 'tool_use' && block.id && block.name)
      .map((block) => ({
        tool_use_id: block.id!,
        tool: block.name!,
        input: block.input ?? {},
      }));
    const toolResults = blocks
      .filter((block) => block.type === 'tool_result' && block.tool_use_id)
      .map((block) => ({
        tool_use_id: block.tool_use_id!,
        text: toolResultText(block.content),
        ...(block.is_error ? { isError: true } : {}),
      }));
    const text2 = contentText(content);
    if (text2.length === 0 && toolUses.length === 0 && toolResults.length === 0) continue;
    const message: Message = {
      role: parsed.type === 'assistant' ? 'assistant' : 'user',
      text: text2,
      toolUses,
    };
    if (toolResults.length > 0) message.toolResults = toolResults;
    messages.push(message);
  }
  return messages;
}

/** Loads and parses one Claude Code session file. */
export async function loadClaudeTranscript(path: string): Promise<Message[]> {
  return parseClaudeTranscript(await readFile(path, 'utf8'));
}
