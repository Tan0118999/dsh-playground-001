import test from 'node:test';
import assert from 'node:assert/strict';
import { renderReadingPage } from '../lib/render.js';

const SOURCE = `---
name: demo-entry
title: 演示条目标题
description: 一条用于测试的说明。
category: methodology
tags: [dsh, plugin]
stacks: [cordis]
updated: 2026-11-03
---

# 演示条目标题

## 第一节

这是 **正文**，带一个 [链接](https://example.com) 和 \`code\`。

| 列 | 值 |
| --- | --- |
| a | b |

\`\`\`json
{"ok": true}
\`\`\`

## 第二节

![示意图](assets/pixel.png)
`;

const PIXEL = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=',
  'base64',
);

function baseOptions(overrides = {}) {
  return {
    source: SOURCE,
    sourcePath: 'experiences/demo/demo-entry.md',
    sourceAbsolutePath: '/repo/experiences/demo/demo-entry.md',
    prompt: '读者提示：先看第一节。',
    style: 'paper',
    theme: 'auto',
    generatedAt: '2026-11-03T00:00:00.000Z',
    io: {
      readBytes: async () => PIXEL,
    },
    ...overrides,
  };
}

test('renders a complete self-contained document', async () => {
  const page = await renderReadingPage(baseOptions());
  assert.equal(page.title, '演示条目标题');
  assert.equal(page.slug, 'demo-entry');
  assert.equal(page.sections, 2);
  assert.match(page.html, /^<!doctype html>/);
  assert.match(page.html, /<html lang="zh-CN" data-preset="paper" data-theme="auto">/);
  assert.match(page.html, /<meta http-equiv="Content-Security-Policy"/);
  assert.match(page.html, /<title>演示条目标题<\/title>/);
  assert.match(page.html, /读者提示：先看第一节。/);
  assert.match(page.html, /<h2 id="第一节">第一节<\/h2>/);
  assert.match(page.html, /<h2 id="第二节">第二节<\/h2>/);
  assert.match(page.html, /class="preset-button" data-preset="slate"/);
  assert.match(page.html, /id="style-toggle"/);
  assert.match(page.html, /id="reader-sidebar"/);
  assert.match(page.html, /src="data:image\/png;base64,/);
  assert.equal((page.html.match(/<h1/g) ?? []).length, 1);
  assert.ok(!/<link[^>]+stylesheet/i.test(page.html));
  assert.ok(!/<script[^>]+src=/i.test(page.html));
  assert.deepEqual(page.warnings, []);
});

test('renders every preset token block and the active preset', async () => {
  const page = await renderReadingPage(baseOptions({ style: 'midnight', theme: 'dark' }));
  for (const preset of ['paper', 'slate', 'contrast', 'midnight']) {
    assert.match(page.html, new RegExp(`html\\[data-preset="${preset}"\\]`));
  }
  assert.match(page.html, /data-preset="midnight" data-theme="dark"/);
  assert.match(page.html, /aria-pressed="true"[^>]*data-preset="midnight"|data-preset="midnight"[^>]*aria-pressed="true"/);
});

test('applies accent, width and font-size overrides', async () => {
  const page = await renderReadingPage(
    baseOptions({ accent: '#123456', width: 52, fontSize: 18 }),
  );
  assert.match(page.html, /style="--accent:#123456;--measure:52;--reader-scale:1\.059"/);
  assert.match(page.html, /--measure:52/);
});

test('escapes prompt HTML rather than executing it', async () => {
  const page = await renderReadingPage(
    baseOptions({ prompt: '<script>alert("x")</script>\n\n**粗体**' }),
  );
  assert.ok(!page.html.includes('<script>alert("x")</script>'));
  assert.match(page.html, /&lt;script&gt;alert\(&quot;x&quot;\)&lt;\/script&gt;/);
  assert.match(page.html, /<strong>粗体<\/strong>/);
});

test('leaves local images alone when embedding is disabled or the file is unreadable', async () => {
  const disabled = await renderReadingPage(baseOptions({ embedImages: false }));
  assert.match(disabled.html, /src="assets\/pixel.png"/);

  const missing = await renderReadingPage(
    baseOptions({ io: { readBytes: async () => { throw new Error('nope'); } } }),
  );
  assert.match(missing.html, /src="assets\/pixel.png"/);
  assert.equal(missing.warnings.length, 1);
  assert.match(missing.warnings[0], /image not embedded/);
});

test('uses an explicit title override and renders metadata', async () => {
  const page = await renderReadingPage(baseOptions({ title: '自定义标题' }));
  assert.match(page.html, /<title>自定义标题<\/title>/);
  assert.match(page.html, /<h1>自定义标题<\/h1>/);
  assert.match(page.html, /#dsh/);
  assert.match(page.html, /#plugin/);
  assert.match(page.html, /cordis/);
  assert.match(page.html, /2026-11-03/);
});

test('rejects invalid style, theme and numeric options', async () => {
  await assert.rejects(() => renderReadingPage(baseOptions({ style: 'neon' })), /unknown style preset/);
  await assert.rejects(() => renderReadingPage(baseOptions({ theme: 'sepia' })), /unknown theme/);
  await assert.rejects(() => renderReadingPage(baseOptions({ accent: 'red' })), /accent must be/);
  await assert.rejects(() => renderReadingPage(baseOptions({ width: 999 })), /width must be/);
  await assert.rejects(() => renderReadingPage(baseOptions({ fontSize: 99 })), /font-size must be/);
});

test('works without frontmatter and derives a title from the first heading', async () => {
  const page = await renderReadingPage(
    baseOptions({
      source: '# 无 frontmatter 的标题\n\n## 小节\n\n内容。\n',
      sourcePath: 'plain.md',
      sourceAbsolutePath: '/repo/plain.md',
    }),
  );
  assert.equal(page.title, '无 frontmatter 的标题');
  assert.equal(page.slug, '无-frontmatter-的标题');
  assert.equal((page.html.match(/<h1/g) ?? []).length, 1);
});
