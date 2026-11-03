import test from 'node:test';
import assert from 'node:assert/strict';
import { collectImageUrls, escapeHtml, renderMarkdown, safeUrl } from '../lib/markdown.js';

test('renders headings with stable, duplicate-safe ids', () => {
  const { html, headings } = renderMarkdown('# 标题\n\n## 同一节\n\n## 同一节\n');
  assert.match(html, /<h1 id="标题">标题<\/h1>/);
  assert.match(html, /<h2 id="同一节">同一节<\/h2>/);
  assert.match(html, /<h2 id="同一节-2">同一节<\/h2>/);
  assert.deepEqual(
    headings.map((heading) => [heading.level, heading.id]),
    [
      [1, '标题'],
      [2, '同一节'],
      [2, '同一节-2'],
    ],
  );
});

test('escapes raw HTML and drops unsafe link schemes', () => {
  const { html } = renderMarkdown(
    '<img src=x onerror="alert(1)">\n\n[click](javascript:alert(1))\n\n[ok](https://example.com)\n',
  );
  assert.ok(!html.includes('<img src=x'));
  assert.ok(html.includes('&lt;img src=x onerror=&quot;alert(1)&quot;&gt;'));
  assert.ok(!html.includes('href="javascript:'));
  assert.ok(html.includes('href="https://example.com"'));
  assert.ok(html.includes('target="_blank" rel="noopener noreferrer"'));
});

test('renders inline code, emphasis and strikethrough without interpreting code contents', () => {
  const { html } = renderMarkdown('**粗体** 和 *斜体*，`a < b`，~~删除~~');
  assert.match(html, /<strong>粗体<\/strong>/);
  assert.match(html, /<em>斜体<\/em>/);
  assert.match(html, /<code>a &lt; b<\/code>/);
  assert.match(html, /<del>删除<\/del>/);
});

test('renders a fenced code block with escaped content and a copy control', () => {
  const source = '```json\n{"a": "<script>alert(1)</script>"}\n```\n';
  const { html } = renderMarkdown(source);
  assert.match(html, /class="code-block"/);
  assert.match(html, /data-lang="json"/);
  assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
  assert.ok(!html.includes('<script>alert(1)</script>'));
  assert.match(html, /data-copy/);
});

test('renders GFM tables, alignment and escaped pipes', () => {
  const source = `| 名称 | 值 | 备注 |
| :--- | :---: | ---: |
| a | b\\|c | d |
`;
  const { html } = renderMarkdown(source);
  assert.match(html, /<table>/);
  assert.match(html, /style="text-align:left"/);
  assert.match(html, /style="text-align:center"/);
  assert.match(html, /style="text-align:right"/);
  assert.ok(html.includes('b|c'));
});

test('renders nested lists and blockquotes', () => {
  const source = `- 顶层
  - 子项
- 顶层二

> 引用中的 **重点**
`;
  const { html } = renderMarkdown(source);
  assert.match(html, /<ul>[\s\S]*<li>顶层[\s\S]*<ul>[\s\S]*<li>子项/);
  assert.match(html, /<blockquote>[\s\S]*<strong>重点<\/strong>/);
});

test('collects image urls and maps them through imageMap', () => {
  const source = '![图](assets/a.png) ![外链](https://example.com/b.png)';
  const urls = collectImageUrls(source);
  assert.deepEqual(urls, ['assets/a.png', 'https://example.com/b.png']);
  const { html } = renderMarkdown(source, {
    imageMap: new Map([['assets/a.png', 'data:image/png;base64,AAAA']]),
  });
  assert.match(html, /src="data:image\/png;base64,AAAA"/);
  assert.match(html, /src="https:\/\/example.com\/b.png"/);
});

test('safeUrl allows only explicit schemes and relative paths', () => {
  assert.equal(safeUrl('https://example.com'), 'https://example.com');
  assert.equal(safeUrl('mailto:a@b.c'), 'mailto:a@b.c');
  assert.equal(safeUrl('../x.md'), '../x.md');
  assert.equal(safeUrl('javascript:alert(1)'), '');
  assert.equal(safeUrl('data:text/html,x'), '');
  assert.equal(safeUrl('data:image/png;base64,AAAA', { image: true }), 'data:image/png;base64,AAAA');
});

test('escapeHtml handles all five characters', () => {
  assert.equal(escapeHtml(`&<>"'`), '&amp;&lt;&gt;&quot;&#39;');
});

test('returns plain text for reading-time estimation', () => {
  const { text } = renderMarkdown('# 标题\n\n这是 **正文**，包含 `code`。\n');
  assert.match(text, /标题/);
  assert.match(text, /这是 正文/);
  assert.ok(!text.includes('<'));
});
