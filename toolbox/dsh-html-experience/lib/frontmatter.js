/**
 * Parse the small YAML subset used by experience-entry frontmatter.
 *
 * Supported: `key: value`, quoted values, inline arrays (`[a, b]`), block
 * arrays (`key:` followed by `  - item`), and comments on their own line.
 * Nested objects, multi-line strings and anchors are intentionally left to the
 * literal string value; the experience template does not use them.
 */

/**
 * @param {string} value
 * @returns {string|string[]}
 */
function parseValue(value) {
  const text = value.trim();
  if (text.startsWith('[') && text.endsWith(']')) {
    return splitInlineList(text.slice(1, -1));
  }
  return unquote(text);
}

/**
 * @param {string} text
 * @returns {string}
 */
function unquote(text) {
  if (text.length >= 2 && text.startsWith('"') && text.endsWith('"')) {
    return text
      .slice(1, -1)
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\');
  }
  if (text.length >= 2 && text.startsWith("'") && text.endsWith("'")) {
    return text.slice(1, -1).replace(/''/g, "'");
  }
  return text;
}

/**
 * Split a comma-separated inline array without breaking quoted commas.
 * @param {string} text
 * @returns {string[]}
 */
function splitInlineList(text) {
  /** @type {string[]} */
  const items = [];
  let current = '';
  let quote = '';
  for (const char of text) {
    if (quote) {
      current += char;
      if (char === quote) quote = '';
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      current += char;
      continue;
    }
    if (char === ',') {
      items.push(unquote(current.trim()));
      current = '';
      continue;
    }
    current += char;
  }
  if (current.trim()) items.push(unquote(current.trim()));
  return items.filter((item) => item.length > 0);
}

/**
 * Parse frontmatter and return the remaining Markdown body.
 * @param {string} source
 * @returns {{ data: Record<string, string|string[]>, body: string, raw: string }}
 */
export function parseFrontmatter(source) {
  const normalized = String(source ?? '').replace(/\r\n?/g, '\n');
  if (!normalized.startsWith('---\n')) {
    return { data: {}, body: normalized, raw: '' };
  }
  const end = normalized.indexOf('\n---\n', 4);
  if (end === -1) {
    return { data: {}, body: normalized, raw: '' };
  }
  const raw = normalized.slice(4, end);
  const body = normalized.slice(end + 5).replace(/^\n/, '');

  /** @type {Record<string, string|string[]>} */
  const data = {};
  const lines = raw.split('\n');
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line.trim() || /^\s*#/.test(line)) continue;
    const match = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
    if (!match) continue;
    const [, key, rest] = match;
    if (rest.trim().length > 0) {
      data[key] = parseValue(rest);
      continue;
    }
    /** @type {string[]} */
    const items = [];
    let cursor = index + 1;
    while (cursor < lines.length) {
      const item = /^\s+-\s+(.*)$/.exec(lines[cursor]);
      if (!item) break;
      items.push(unquote(item[1].trim()));
      cursor += 1;
    }
    data[key] = items;
    index = cursor - 1;
  }
  return { data, body, raw };
}
