/**
 * Parakeet SentencePiece tokenizer — decodes token IDs → text.
 *
 * Ported from parakeet.js (MIT) `src/tokenizer.js` — adapted to TypeScript.
 *
 * The vocabulary is loaded from `vocab.txt` (8192 entries for Parakeet TDT 0.6B v3).
 * Format: `<token> <id>` per line. Token `<blk>` is the blank token.
 * SentencePiece marker `▁` (U+2581) is decoded as a space.
 */

import { readFile } from "node:fs/promises";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Tokenizer {
  /** Number of tokens in vocabulary. */
  readonly vocabSize: number;
  /** Token ID for the blank symbol `<blk>`. */
  readonly blankId: number;
  /** Decode an array of token IDs into text. */
  decode(ids: number[]): string;
  /**
   * Check whether a token has the SentencePiece word-boundary prefix `▁`.
   * Used by the segment builder to detect word starts without relying on
   * the `decode()` method (which strips leading whitespace).
   */
  hasWordPrefix(id: number): boolean;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/**
 * Load a tokenizer from a `vocab.txt` file on disk.
 *
 * @param vocabPath Absolute path to `vocab.txt`.
 */
export async function loadTokenizer(vocabPath: string): Promise<Tokenizer> {
  const text = await readFile(vocabPath, "utf-8");
  const lines = text.split(/\r?\n/).filter(Boolean);

  const id2token: string[] = [];
  for (const line of lines) {
    const parts = line.split(/\s+/);
    const tok = parts[0];
    const idStr = parts[1];
    if (!tok || !idStr) continue;
    const id = parseInt(idStr, 10);
    if (Number.isNaN(id)) continue;
    id2token[id] = tok;
  }

  // Find blank token ID
  let blankId = id2token.indexOf("<blk>");
  if (blankId === -1) {
    // Fallback — Parakeet TDT 0.6B v3 uses 1024
    blankId = 1024;
  }

  // Pre-compute sanitized tokens (▁ → space) for fast decode
  const sanitized: string[] = id2token.map((t) => (t ? t.replace(/\u2581/g, " ") : ""));

  return {
    vocabSize: id2token.length,
    blankId,
    decode(ids: number[]): string {
      const tokens: string[] = [];
      for (const id of ids) {
        if (id === blankId) continue;
        const tok = sanitized[id];
        if (tok === undefined) continue;
        tokens.push(tok);
      }

      let text = tokens.join("");
      // Match Python reference regex: strip leading whitespace, collapse spaces
      text = text.replace(/^\s+/, ""); // \A\s
      text = text.replace(/\s+(?=[^\w\s])/g, ""); // space before punctuation
      text = text.replace(/\s+/g, " "); // collapse multiple spaces
      return text.trim();
    },
    hasWordPrefix(id: number): boolean {
      const raw = id2token[id];
      return raw !== undefined && raw.startsWith("\u2581");
    },
  };
}
