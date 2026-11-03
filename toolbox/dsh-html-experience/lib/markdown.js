/**
 * Small, dependency-free Markdown renderer for experience entries.
 *
 * It intentionally supports only the constructs the library uses (headings,
 * paragraphs, lists, tables, blockquotes, fenced code, rules, links, images,
 * emphasis and inline code) so that every construct can be escaped and checked
 * locally. Raw HTML in the source is escaped, never passed through.
 */

const ESCAPE = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** @param {string} text */
export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) => ESCAPE[char]);
}

/**
 * Link and image targets are limited to explicit, safe schemes and relative
 * paths. Unknown schemes are dropped rather than emitted.
 * @param {string} value
 * @param {{ image?: boolean }} [options]
 * @returns {string}
 */
export function safeUrl(value, options = {}) {
  const url = String(value ?? '').trim();
  if (!url) return '';
  if (url.startsWith('//')) return '';
  if (/^(https?:|mailto:)/i.test(url)) return url;
  if (options.image && /^data:image\/(?:png|jpe?g|gif|webp|avif|svg\+xml);base64,/i.test(url)) return url;
  if (url.startsWith('#') || url.startsWith('/') || url.startsWith('./') || url.startsWith('../')) return url;
  if (!/^[A-Za-z][A-Za-z0-9+.-]*:/.test(url)) return url;
  return '';
}

/**
 * Find every image target in the source so the caller can inline local files.
 * @param {string} source
 * @returns {string[]}
 */
export function collectImageUrls(source) {
  const urls = [];
  const pattern = /!\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+["'][^"']*["'])?\s*\)/g;
  let match;
  while ((match = pattern.exec(String(source ?? ''))) !== null) {
    if (!urls.includes(match[1])) urls.push(match[1]);
  }
  return urls;
}

/** @param {string} text */
function slugify(text) {
  const plain = text
    .replace(/<[^>]*>/g, '')
    .replace(/[`*_~]/g, '')
    .trim()
    .toLowerCase();
  const slug = plain
    .normalize('NFKD')
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'section';
}

/** @param {string} text */
function splitTableRow(text) {
  let line = text.trim();
  if (line.startsWith('|')) line = line.slice(1);
  if (line.endsWith('|') && !line.endsWith('\\|')) line = line.slice(0, -1);
  /** @type {string[]} */
  const cells = [];
  let current = '';
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '\\' && line[index + 1] === '|') {
      current += '|';
      index += 1;
      continue;
    }
    if (char === '|') {
      cells.push(current.trim());
      current = '';
      continue;
    }
    current += char;
  }
  cells.push(current.trim());
  return cells;
}

/** @param {string} line */
function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

/** @param {string} line */
function listMarker(line) {
  const match = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/.exec(line);
  if (!match) return null;
  return {
    indent: match[1].replace(/\t/g, '    ').length,
    ordered: /^\d/.test(match[2]),
    text: match[3],
  };
}

/**
 * Render inline Markdown, escaping everything first.
 * @param {string} text
 * @param {{ imageMap?: Map<string, string> }} [options]
 */
function renderInline(text, options = {}) {
  const placeholders = [];
  const stash = (html) => {
    const token = `\u0000PH${placeholders.length}\u0000`;
    placeholders.push(html);
    return token;
  };

  let value = String(text);

  // Inline code first, so its content is never interpreted as emphasis.
  value = value.replace(/(`+)([\s\S]*?)\1/g, (_, _ticks, code) => stash(`<code>${escapeHtml(code.trim())}</code>`));

  // Images before links.
  value = value.replace(
    /!\[([^\]]*)\]\(\s*<?([^\s)>]+)>?(?:\s+["']([^"']*)["'])?\s*\)/g,
    (_match, alt, rawUrl, title) => {
      const mapped = options.imageMap?.get(rawUrl) ?? rawUrl;
      const url = safeUrl(mapped, { image: true });
      if (!url) return escapeHtml(alt || '');
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
      return stash(
        `<img src="${escapeHtml(url)}" alt="${escapeHtml(alt)}"${titleAttr} loading="lazy" decoding="async">`,
      );
    },
  );

  value = value.replace(
    /\[([^\]]+)\]\(\s*<?([^\s)>]+)>?(?:\s+["']([^"']*)["'])?\s*\)/g,
    (_match, label, rawUrl, title) => {
      const url = safeUrl(rawUrl);
      const renderedLabel = renderInline(label, options);
      if (!url) return renderedLabel;
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
      const external = /^https?:/i.test(url);
      return stash(
        `<a href="${escapeHtml(url)}"${titleAttr}${external ? ' target="_blank" rel="noopener noreferrer"' : ''}>${renderedLabel}</a>`,
      );
    },
  );

  value = escapeHtml(value);

  // Autolinks written as <https://…> survive escaping as &lt;…&gt;.
  value = value.replace(/&lt;(https?:\/\/[^\s&]+)&gt;/g, (_match, url) =>
    stash(`<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>`),
  );

  value = value.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
  value = value.replace(/~~([^~\n]+?)~~/g, '<del>$1</del>');
  value = value.replace(/(^|[^\w*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  value = value.replace(/ {2,}\n/g, '<br>\n');

  return value.replace(/\u0000PH(\d+)\u0000/g, (_match, index) => placeholders[Number(index)]);
}

/**
 * Render the blocks of a Markdown document.
 * @param {string[]} lines
 * @param {{ headings: Array<{level:number,text:string,id:string}>, imageMap?: Map<string,string>, seenIds: Map<string,number> }} state
 * @returns {string}
 */
function renderBlocks(lines, state) {
  /** @type {string[]} */
  const out = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    // Code fence placeholder emitted during extraction.
    const placeholder = /^\u0000CODE(\d+)\u0000$/.exec(line.trim());
    if (placeholder) {
      out.push(line.trim());
      index += 1;
      continue;
    }

    const heading = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2];
      const id = uniqueId(slugify(text), state.seenIds);
      state.headings.push({ level, text: stripInline(text), id });
      out.push(`<h${level} id="${escapeHtml(id)}">${renderInline(text, state)}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)) {
      out.push('<hr>');
      index += 1;
      continue;
    }

    if (line.trimStart().startsWith('```')) {
      // An unclosed fence is treated as a paragraph rather than swallowing the document.
      out.push(`<p>${renderInline(line, state)}</p>`);
      index += 1;
      continue;
    }

    if (line.trimStart().startsWith('>')) {
      /** @type {string[]} */
      const inner = [];
      while (index < lines.length && lines[index].trimStart().startsWith('>')) {
        inner.push(lines[index].replace(/^\s*>\s?/, ''));
        index += 1;
      }
      out.push(`<blockquote>\n${renderBlocks(inner, state)}\n</blockquote>`);
      continue;
    }

    const list = listMarker(line);
    if (list) {
      const [html, next] = parseList(lines, index, list.indent, state);
      out.push(html);
      index = next;
      continue;
    }

    if (line.includes('|') && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      const [html, next] = parseTable(lines, index, state);
      out.push(html);
      index = next;
      continue;
    }

    // Paragraph: stop at a blank line or at a line that starts another block.
    /** @type {string[]} */
    const paragraph = [];
    while (index < lines.length && lines[index].trim()) {
      const current = lines[index];
      if (paragraph.length > 0 && startsBlock(current, lines[index + 1])) break;
      paragraph.push(current.trim());
      index += 1;
    }
    out.push(`<p>${renderInline(paragraph.join('\n'), state)}</p>`);
  }

  return out.join('\n');
}

/**
 * Whether a line starts a block that must interrupt a running paragraph.
 * @param {string} line
 * @param {string|undefined} next
 */
function startsBlock(line, next) {
  if (/^\s*\u0000CODE\d+\u0000\s*$/.test(line)) return true;
  if (/^(#{1,6})\s+/.test(line)) return true;
  if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(line)) return true;
  if (line.trimStart().startsWith('>')) return true;
  if (line.trimStart().startsWith('```')) return true;
  if (listMarker(line)) return true;
  if (line.includes('|') && next !== undefined && isTableSeparator(next)) return true;
  return false;
}

/** @param {string} text */
function stripInline(text) {
  return String(text)
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_~]/g, '')
    .trim();
}

/** @param {string} base @param {Map<string, number>} seen */
function uniqueId(base, seen) {
  const count = seen.get(base) ?? 0;
  seen.set(base, count + 1);
  return count === 0 ? base : `${base}-${count + 1}`;
}

/**
 * Parse a list recursively, returning serialized HTML and the next line index.
 * @param {string[]} lines
 * @param {number} start
 * @param {number} indent
 * @param {{ headings: unknown[], imageMap?: Map<string,string>, seenIds: Map<string,number> }} state
 * @returns {[string, number]}
 */
function parseList(lines, start, indent, state) {
  const first = /** @type {{ordered:boolean}} */ (listMarker(lines[start]));
  const tag = first.ordered ? 'ol' : 'ul';
  /** @type {Array<{ text: string, children: string[] }>} */
  const items = [];
  let index = start;

  while (index < lines.length) {
    const marker = listMarker(lines[index]);
    if (!marker || marker.indent !== indent) break;
    if (marker.ordered !== first.ordered) break;

    let text = marker.text;
    index += 1;
    /** @type {string[]} */
    const children = [];

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        // A blank line continues the item only when the next line is more indented.
        const next = lines[index + 1];
        const nextMarker = next === undefined ? null : listMarker(next);
        if (nextMarker && nextMarker.indent > indent) {
          index += 1;
          continue;
        }
        break;
      }
      const nextMarker = listMarker(line);
      if (nextMarker && nextMarker.indent > indent) {
        const [html, next] = parseList(lines, index, nextMarker.indent, state);
        children.push(html);
        index = next;
        continue;
      }
      if (nextMarker && nextMarker.indent <= indent) break;
      if (line.replace(/\t/g, '    ').match(/^\s*/)[0].length > indent) {
        text += `\n${line.trim()}`;
        index += 1;
        continue;
      }
      break;
    }

    items.push({ text, children });
  }

  const html = [
    `<${tag}>`,
    ...items.map((item) => {
      const checkbox = /^\[( |x|X)\]\s+/.exec(item.text);
      if (checkbox) {
        const checked = checkbox[1].toLowerCase() === 'x' ? ' checked' : '';
        const body = item.text.slice(checkbox[0].length);
        return `<li class="task"><input type="checkbox" disabled${checked}> ${renderInline(body, state)}${item.children.join('\n')}</li>`;
      }
      return `<li>${renderInline(item.text, state)}${item.children.join('\n')}</li>`;
    }),
    `</${tag}>`,
  ].join('\n');

  return [html, index];
}

/**
 * @param {string[]} lines
 * @param {number} start
 * @param {{ headings: unknown[], imageMap?: Map<string,string>, seenIds: Map<string,number> }} state
 * @returns {[string, number]}
 */
function parseTable(lines, start, state) {
  const header = splitTableRow(lines[start]);
  const align = splitTableRow(lines[start + 1]).map((cell) => {
    const left = cell.startsWith(':');
    const right = cell.endsWith(':');
    if (left && right) return 'center';
    if (right) return 'right';
    if (left) return 'left';
    return '';
  });
  /** @type {string[][]} */
  const rows = [];
  let index = start + 2;
  while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
    rows.push(splitTableRow(lines[index]));
    index += 1;
  }
  const style = (column) => (align[column] ? ` style="text-align:${align[column]}"` : '');
  const html = [
    '<div class="table-wrap"><table>',
    '<thead><tr>',
    ...header.map((cell, column) => `<th${style(column)}>${renderInline(cell, state)}</th>`),
    '</tr></thead>',
    '<tbody>',
    ...rows.map(
      (row) =>
        `<tr>${header
          .map((_, column) => `<td${style(column)}>${renderInline(row[column] ?? '', state)}</td>`)
          .join('')}</tr>`,
    ),
    '</tbody></table></div>',
  ].join('\n');
  return [html, index];
}

/**
 * Extract fenced code blocks before block parsing so their contents stay literal.
 * @param {string[]} lines
 * @param {{ lang: string, code: string }[]} blocks
 * @returns {string[]}
 */
function extractCodeFences(lines, blocks) {
  /** @type {string[]} */
  const output = [];
  let index = 0;
  while (index < lines.length) {
    const open = /^\s*```([^\s`]*)\s*$/.exec(lines[index]);
    if (!open) {
      output.push(lines[index]);
      index += 1;
      continue;
    }
    const lang = open[1] || '';
    /** @type {string[]} */
    const code = [];
    index += 1;
    while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
      code.push(lines[index]);
      index += 1;
    }
    if (index < lines.length) index += 1; // closing fence
    const id = blocks.length;
    blocks.push({ lang, code: code.join('\n') });
    output.push(`\u0000CODE${id}\u0000`);
  }
  return output;
}

/**
 * Render a Markdown document.
 * @param {string} source
 * @param {{ imageMap?: Map<string, string> }} [options]
 * @returns {{ html: string, headings: { level: number, text: string, id: string }[], text: string }}
 */
export function renderMarkdown(source, options = {}) {
  const normalized = String(source ?? '').replace(/\u0000/g, '').replace(/\r\n?/g, '\n');
  /** @type {{ lang: string, code: string }[]} */
  const blocks = [];
  const lines = extractCodeFences(normalized.split('\n'), blocks);

  // Render code blocks as placeholder stand-ins in document order.
  const codeHtml = blocks.map((block, id) => {
    const label = block.lang ? escapeHtml(block.lang) : 'text';
    const code = escapeHtml(block.code);
    return `<figure class="code-block" data-lang="${label}" id="code-${id}">
<figcaption><span class="code-lang">${label}</span><button type="button" class="copy-button" data-copy>复制</button></figcaption>
<pre><code>${code}</code></pre>
</figure>`;
  });

  const state = { headings: [], imageMap: options.imageMap, seenIds: new Map() };
  let html = renderBlocks(lines, state);
  html = html.replace(/\u0000CODE(\d+)\u0000/g, (_match, id) => codeHtml[Number(id)] ?? '');
  html = html.replace(/\u0000CODE(\d+)\u0000/g, '');

  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();

  return { html, headings: state.headings, text };
}
