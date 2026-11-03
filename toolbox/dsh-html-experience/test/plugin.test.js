import test from 'node:test';
import assert from 'node:assert/strict';
import { isAbsolute, resolve } from 'node:path';
import { apply, parseCommandInput } from '../lib/index.js';

const DOC = `---
name: demo-entry
title: GPU 时间线阅读
description: 第一次读 Nsight Systems 时间线时如何看行与 gap。
category: performance
tags: [gpu, timeline]
stacks: [nsys]
updated: 2026-11-03
---

# GPU 时间线阅读

## 看行

时间线正文。
`;

function makeEnv() {
  const cwd = '/work';
  const files = new Map([
    [
      '/work/catalog.json',
      JSON.stringify({
        schema_version: 1,
        entries: [
          {
            name: 'demo-entry',
            title: 'GPU 时间线阅读',
            description: '第一次读 Nsight Systems 时间线时如何看行与 gap。',
            category: 'performance',
            tags: ['gpu', 'timeline'],
            symptoms: ['时间线看不懂'],
            stacks: ['nsys'],
            updated: '2026-11-03',
            path: 'experiences/demo.md',
          },
        ],
      }),
    ],
    ['/work/experiences/demo.md', DOC],
  ]);
  const writes = [];
  const tools = [];
  const commands = [];
  const ctx = {
    fs: {
      async resolve(path, options = {}) {
        const absolute = isAbsolute(path) ? path : resolve(options.cwd ?? cwd, path);
        return { targetKey: absolute, displayPath: absolute };
      },
      async stat(target) {
        return files.has(target.targetKey) ? { type: 'file', version: 'v1' } : undefined;
      },
      async readText(target) {
        if (!files.has(target.targetKey)) throw new Error(`missing ${target.targetKey}`);
        return files.get(target.targetKey);
      },
      async readBytes() {
        throw new Error('no images in this fixture');
      },
      async writeText(target, content) {
        files.set(target.targetKey, content);
        writes.push(target.targetKey);
      },
      processPath(target) {
        return target.targetKey;
      },
    },
    workingDirectory: { ensure: async () => cwd },
    tools: { register: (definition) => tools.push(definition) },
    commands: { register: (definition) => commands.push(definition) },
  };
  return { ctx, files, writes, tools, commands, cwd };
}

test('registers one tool and one command with the documented interface', () => {
  const { ctx, tools, commands } = makeEnv();
  apply(ctx);
  assert.equal(tools.length, 1);
  assert.equal(commands.length, 1);
  assert.equal(tools[0].name, 'html_experience');
  assert.deepEqual(tools[0].parameters.required, ['prompt']);
  assert.deepEqual(tools[0].parameters.properties.style.enum, ['paper', 'slate', 'contrast', 'midnight']);
  assert.deepEqual(tools[0].parameters.properties.theme.enum, ['auto', 'light', 'dark']);
  assert.equal(commands[0].name, 'html');
});

test('the tool writes a self-contained page and never overwrites by default', async () => {
  const { ctx, files, tools, writes } = makeEnv();
  apply(ctx);
  const tool = tools[0];
  const exec = { agent: {}, signal: new AbortController().signal };

  const first = await tool.execute({ prompt: '先看 GPU 时间线的 gap' }, exec);
  assert.equal(first.path, '/work/reading-demo-entry.html');
  assert.equal(first.source, 'experiences/demo.md');
  assert.equal(first.style, 'paper');
  assert.match(files.get(first.path), /<title>GPU 时间线阅读<\/title>/);
  assert.match(files.get(first.path), /先看 GPU 时间线的 gap/);

  const second = await tool.execute({ prompt: '再看一次', doc: 'demo-entry' }, exec);
  assert.equal(second.path, '/work/reading-demo-entry-2.html');
  assert.deepEqual(writes, [first.path, second.path]);

  const overwritten = await tool.execute(
    { prompt: '覆盖', doc: 'demo-entry', out: 'custom/page.html', overwrite: true },
    exec,
  );
  assert.equal(overwritten.path, '/work/custom/page.html');
  assert.equal(files.get('/work/custom/page.html'), files.get(overwritten.path));
});

test('tool result render contains the path and delivery hint', async () => {
  const { ctx, tools } = makeEnv();
  apply(ctx);
  const tool = tools[0];
  const value = await tool.execute(
    { prompt: '读一下', doc: 'experiences/demo.md', style: 'midnight', theme: 'dark' },
    { agent: {}, signal: new AbortController().signal },
  );
  const content = tool.output.render({ prompt: '读一下' }, value);
  assert.equal(content[0].type, 'text');
  assert.match(content[0].text, /HTML reading page generated/);
  assert.match(content[0].text, /midnight/);
  assert.match(content[0].text, /present the file/);
});

test('the tool reports expected failures as errors', async () => {
  const { ctx, tools } = makeEnv();
  apply(ctx);
  const tool = tools[0];
  const exec = { agent: {}, signal: new AbortController().signal };
  await assert.rejects(() => tool.execute({ prompt: 'x', doc: 'missing.md' }, exec), /not found/);
  await assert.rejects(() => tool.execute({ prompt: '' }, exec), /prompt is required/);
  await assert.rejects(() => tool.execute({ prompt: 'x', doc: 'demo-entry', style: 'neon' }, exec), /unknown style preset/);
});

test('parseCommandInput handles flags, equals syntax and the -- delimiter', () => {
  const parsed = parseCommandInput(
    ' --style=midnight --theme dark --width 60 -- 把“--force”当成正文，不当选项',
  );
  assert.equal(parsed.prompt, '把“--force”当成正文，不当选项');
  assert.equal(parsed.style, 'midnight');
  assert.equal(parsed.theme, 'dark');
  assert.equal(parsed.width, '60');
  assert.equal(parsed.force, undefined);
});

test('parseCommandInput rejects unknown options and missing values', () => {
  assert.throws(() => parseCommandInput('--neon x'), /unknown option --neon/);
  assert.throws(() => parseCommandInput('--style'), /needs a value/);
});

test('the /html command renders a page and returns its display path', async () => {
  const { ctx, commands, files } = makeEnv();
  apply(ctx);
  const command = commands[0];
  const invocation = {
    rawInput: ' --style slate --out from-command.html -- GPU 时间线怎么看 gap',
    agent: {},
    signal: new AbortController().signal,
  };
  const result = await command.handler(invocation);
  assert.equal(result.kind, 'success');
  assert.match(result.text, /HTML 阅读页已生成/);
  assert.match(result.text, /from-command\.html/);
  assert.match(files.get('/work/from-command.html'), /data-preset="slate"/);
});

test('the /html command reports usage and expected errors without throwing', async () => {
  const { ctx, commands } = makeEnv();
  apply(ctx);
  const command = commands[0];
  const base = { agent: {}, signal: new AbortController().signal };

  const usage = await command.handler({ ...base, rawInput: '' });
  assert.equal(usage.kind, 'error');
  assert.match(usage.text, /Usage: \/html/);

  const unknown = await command.handler({ ...base, rawInput: '--neon x' });
  assert.equal(unknown.kind, 'error');
  assert.match(unknown.text, /unknown option/);

  const missing = await command.handler({ ...base, rawInput: ' --doc missing.md -- prompt' });
  assert.equal(missing.kind, 'error');
  assert.match(missing.text, /not found/);

  const presets = await command.handler({ ...base, rawInput: '--list-presets' });
  assert.equal(presets.kind, 'success');
  assert.match(presets.text, /paper/);
  assert.match(presets.text, /midnight/);
});
