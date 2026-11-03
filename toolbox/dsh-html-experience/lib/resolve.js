/**
 * Resolve which experience document a prompt refers to.
 *
 * The catalog is the machine-readable index maintained by
 * `toolbox/build_catalog.py`. Matching is deliberately simple and explainable:
 * tags/stacks/name count most, then title, description, symptoms and path. A
 * caller can always bypass matching with an explicit document path or entry
 * name.
 */

import { isAbsolute, join, normalize, relative, resolve } from 'node:path';
import { parseFrontmatter } from './frontmatter.js';

/**
 * Split a prompt into searchable latin words and CJK bigrams/trigrams.
 * @param {string} prompt
 * @returns {string[]}
 */
export function tokenizePrompt(prompt) {
  const text = String(prompt ?? '').toLowerCase();
  /** @type {Set<string>} */
  const tokens = new Set();
  for (const word of text.match(/[a-z0-9][a-z0-9_.+-]{1,}/g) ?? []) {
    tokens.add(word);
  }
  for (const run of text.match(/[\p{Script=Han}]+/gu) ?? []) {
    const chars = [...run];
    if (chars.length === 1) tokens.add(chars[0]);
    for (let index = 0; index < chars.length - 1; index += 1) {
      tokens.add(chars[index] + chars[index + 1]);
    }
    for (let index = 0; index < chars.length - 2; index += 1) {
      tokens.add(chars[index] + chars[index + 1] + chars[index + 2]);
    }
  }
  return [...tokens];
}

/** @param {unknown} value */
function textOf(value) {
  if (Array.isArray(value)) return value.join(' ').toLowerCase();
  return String(value ?? '').toLowerCase();
}

/**
 * Score one catalog entry against prompt tokens. Each field contributes at
 * most once per token, so a repeated word cannot dominate the ranking.
 * @param {Record<string, unknown>} entry
 * @param {string[]} tokens
 */
export function scoreEntry(entry, tokens) {
  const fields = {
    name: { text: textOf(entry.name), weight: 7 },
    title: { text: textOf(entry.title), weight: 6 },
    description: { text: textOf(entry.description), weight: 4 },
    symptoms: { text: textOf(entry.symptoms), weight: 4 },
    tags: { text: textOf(entry.tags), weight: 9 },
    stacks: { text: textOf(entry.stacks), weight: 5 },
    path: { text: textOf(entry.path), weight: 3 },
    category: { text: textOf(entry.category), weight: 2 },
  };
  let score = 0;
  for (const token of tokens) {
    for (const field of Object.values(fields)) {
      if (field.text.includes(token)) score += field.weight;
    }
  }
  return score;
}

/**
 * Whether the prompt matched an identifying field (name, title, tags, stacks,
 * path) rather than only prose. A weak prose-only match must not silently pick
 * an unrelated entry.
 * @param {Record<string, unknown>} entry
 * @param {string[]} tokens
 */
export function hasStrongMatch(entry, tokens) {
  const strong = [
    textOf(entry.name),
    textOf(entry.title),
    textOf(entry.tags),
    textOf(entry.stacks),
    textOf(entry.path),
  ].join(' ');
  return tokens.some((token) => strong.includes(token));
}

/**
 * Rank catalog entries for a prompt, best first.
 * @param {Array<Record<string, unknown>>} entries
 * @param {string} prompt
 */
export function rankEntries(entries, prompt) {
  const tokens = tokenizePrompt(prompt);
  return entries
    .map((entry) => ({ entry, score: scoreEntry(entry, tokens) }))
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      return String(right.entry.updated ?? '').localeCompare(String(left.entry.updated ?? ''));
    });
}

/**
 * Walk up from `start` until a catalog.json is found.
 * @param {string} start
 * @param {{ isFile(path: string): Promise<boolean>, readText(path: string): Promise<string> }} io
 * @returns {Promise<{ directory: string, catalog: { entries: Array<Record<string, unknown>> } } | null>}
 */
export async function loadCatalog(start, io) {
  let directory = resolve(start);
  for (let depth = 0; depth < 12; depth += 1) {
    const candidate = join(directory, 'catalog.json');
    if (await io.isFile(candidate)) {
      const parsed = JSON.parse(await io.readText(candidate));
      if (!Array.isArray(parsed.entries)) throw new Error(`${candidate} has no entries array`);
      return { directory, catalog: parsed };
    }
    const parent = resolve(directory, '..');
    if (parent === directory) break;
    directory = parent;
  }
  return null;
}

/**
 * @param {string} path
 * @param {string} cwd
 */
function displayPath(path, cwd, catalogDirectory) {
  const resolved = resolve(path);
  for (const root of [catalogDirectory, cwd]) {
    if (!root) continue;
    const rel = relative(root, resolved);
    if (rel && !rel.startsWith('..') && !isAbsolute(rel)) return normalize(rel);
  }
  return resolved;
}

/**
 * @param {Array<Record<string, unknown>>} entries
 * @param {number} [limit]
 */
export function suggestEntries(entries, limit = 6) {
  return entries.slice(0, limit).map((entry) => `${entry.name} (${entry.title})`);
}

/**
 * Resolve the experience document for a generation request.
 *
 * `doc` accepts an absolute/relative Markdown path, an entry name from the
 * catalog, or a path without the `.md` suffix. Without `doc`, the prompt is
 * matched against the catalog; a weak match is an error rather than a guess.
 *
 * @param {{
 *   doc?: string,
 *   prompt: string,
 *   cwd: string,
 *   io: { isFile(path: string): Promise<boolean>, readText(path: string): Promise<string> }
 * }} options
 * @returns {Promise<{ source: string, path: string, absolutePath: string, entry: Record<string, unknown>|null, catalogDirectory: string|null, reason: 'explicit'|'prompt' }>}
 */
export async function resolveExperience(options) {
  const { doc, prompt, cwd, io } = options;
  const catalogResult = await loadCatalog(cwd, io);
  const entries = /** @type {Array<Record<string, unknown>>} */ (catalogResult?.catalog.entries ?? []);

  if (doc && doc.trim()) {
    const requested = doc.trim();
    const byName = entries.find((entry) => entry.name === requested);
    if (byName && catalogResult) {
      const absolutePath = resolve(catalogResult.directory, String(byName.path));
      return {
        source: await io.readText(absolutePath),
        path: String(byName.path),
        absolutePath,
        entry: byName,
        catalogDirectory: catalogResult.directory,
        reason: 'explicit',
      };
    }

    const candidates = [];
    if (isAbsolute(requested)) {
      candidates.push(requested);
    } else {
      candidates.push(resolve(cwd, requested));
      if (!requested.endsWith('.md')) candidates.push(resolve(cwd, `${requested}.md`));
    }
    for (const absolutePath of candidates) {
      if (await io.isFile(absolutePath)) {
        const source = await io.readText(absolutePath);
        const { data } = parseFrontmatter(source);
        return {
          source,
          path: displayPath(absolutePath, cwd, catalogResult?.directory),
          absolutePath,
          entry: null,
          catalogDirectory: catalogResult?.directory ?? null,
          reason: 'explicit',
        };
      }
    }
    throw new Error(
      `experience document not found: ${requested}` +
        (byName === undefined && entries.length > 0
          ? `. Known entries: ${suggestEntries(entries, 4).join('; ')}`
          : ''),
    );
  }

  if (!catalogResult) {
    throw new Error(
      `no catalog.json found from ${cwd} upward; pass an explicit document path (doc) instead`,
    );
  }
  if (entries.length === 0) throw new Error('catalog.json contains no entries');

  const tokens = tokenizePrompt(prompt);
  const ranked = rankEntries(entries, prompt);
  const best = ranked[0];
  if (!best || best.score < 6 || !hasStrongMatch(best.entry, tokens)) {
    throw new Error(
      `the prompt does not clearly match an experience entry; pass an explicit document path (doc). Closest entries: ${suggestEntries(
        entries,
        4,
      ).join('; ')}`,
    );
  }

  const absolutePath = resolve(catalogResult.directory, String(best.entry.path));
  if (!(await io.isFile(absolutePath))) {
    throw new Error(`catalog entry ${String(best.entry.name)} points to a missing file: ${absolutePath}`);
  }
  return {
    source: await io.readText(absolutePath),
    path: String(best.entry.path),
    absolutePath,
    entry: best.entry,
    catalogDirectory: catalogResult.directory,
    reason: 'prompt',
  };
}
