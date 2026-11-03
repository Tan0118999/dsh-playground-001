import test from 'node:test';
import assert from 'node:assert/strict';
import { parseFrontmatter } from '../lib/frontmatter.js';

test('parses scalars, inline arrays and block arrays', () => {
  const source = `---
name: demo-entry
title: "带引号的标题: 与冒号"
description: '单引号描述'
category: methodology
tags: [dsh, "creator-mode", plugin]
symptoms:
  - 第一条症状
  - 第二条症状
stacks:
  - cordis
updated: 2026-11-03
---

# 正文
`;
  const { data, body } = parseFrontmatter(source);
  assert.equal(data.name, 'demo-entry');
  assert.equal(data.title, '带引号的标题: 与冒号');
  assert.equal(data.description, '单引号描述');
  assert.deepEqual(data.tags, ['dsh', 'creator-mode', 'plugin']);
  assert.deepEqual(data.symptoms, ['第一条症状', '第二条症状']);
  assert.deepEqual(data.stacks, ['cordis']);
  assert.equal(data.updated, '2026-11-03');
  assert.match(body, /^# 正文/);
});

test('returns the whole source when no frontmatter exists', () => {
  const { data, body, raw } = parseFrontmatter('# 只有正文\n');
  assert.deepEqual(data, {});
  assert.equal(raw, '');
  assert.equal(body, '# 只有正文\n');
});

test('leaves an unterminated fence as body content', () => {
  const source = '---\ntitle: x\n\n# no closing fence\n';
  const { data, body } = parseFrontmatter(source);
  assert.deepEqual(data, {});
  assert.equal(body, source);
});

test('normalizes CRLF before parsing', () => {
  const source = '---\r\ntitle: x\r\n---\r\n\r\nbody\r\n';
  const { data, body } = parseFrontmatter(source);
  assert.equal(data.title, 'x');
  assert.equal(body.trim(), 'body');
});
