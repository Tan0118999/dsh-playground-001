import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { loadCatalog, rankEntries, resolveExperience, scoreEntry, tokenizePrompt } from '../lib/resolve.js';

const ENTRY_DOC = `---
name: demo-entry
title: GPU 时间线阅读
description: 第一次读 Nsight Systems 时间线时如何看行与 gap。
category: performance
tags: [gpu, timeline, nsys]
symptoms: [时间线看不懂, GPU 行有空档]
stacks: [nsys]
updated: 2026-11-03
---

# GPU 时间线阅读

## 看行
`;

async function makeWorkspace() {
  const root = await mkdtemp(join(tmpdir(), 'dsh-html-experience-'));
  await mkdir(join(root, 'experiences', 'demo'), { recursive: true });
  await writeFile(join(root, 'experiences', 'demo', 'demo-entry.md'), ENTRY_DOC, 'utf8');
  await writeFile(
    join(root, 'catalog.json'),
    JSON.stringify({
      schema_version: 1,
      entries: [
        {
          name: 'demo-entry',
          title: 'GPU 时间线阅读',
          description: '第一次读 Nsight Systems 时间线时如何看行与 gap。',
          category: 'performance',
          tags: ['gpu', 'timeline', 'nsys'],
          symptoms: ['时间线看不懂', 'GPU 行有空档'],
          stacks: ['nsys'],
          updated: '2026-11-03',
          path: 'experiences/demo/demo-entry.md',
        },
        {
          name: 'other-entry',
          title: '另一个条目',
          description: '与提示词无关。',
          category: 'engineering',
          tags: ['other'],
          symptoms: ['其他问题'],
          stacks: ['node'],
          updated: '2026-01-01',
          path: 'experiences/demo/other.md',
        },
      ],
    }),
    'utf8',
  );
  return root;
}

function nodeIo() {
  return {
    async isFile(path) {
      try {
        return (await stat(path)).isFile();
      } catch {
        return false;
      }
    },
    readText: (path) => readFile(path, 'utf8'),
  };
}

test('tokenizePrompt keeps latin words and CJK bigrams/trigrams', () => {
  const tokens = tokenizePrompt('GPU 时间线怎么读');
  assert.ok(tokens.includes('gpu'));
  assert.ok(tokens.includes('时间'));
  assert.ok(tokens.includes('间线'));
  assert.ok(tokens.includes('时间线'));
});

test('scoreEntry weighs tags and title above description', () => {
  const entry = {
    name: 'demo',
    title: 'GPU 时间线阅读',
    description: '时间线',
    tags: ['gpu'],
    symptoms: [],
    stacks: [],
    path: 'x',
    category: 'performance',
  };
  assert.ok(scoreEntry(entry, ['gpu']) >= 9);
  assert.ok(scoreEntry(entry, ['时间线']) >= 6);
  assert.ok(scoreEntry(entry, ['nothing']) === 0);
});

test('rankEntries puts the strong tag match first', () => {
  const entries = [
    { name: 'a', title: '其他', tags: ['other'], updated: '2026-01-01' },
    { name: 'b', title: 'GPU 时间线', tags: ['gpu'], updated: '2026-01-01' },
  ];
  assert.equal(rankEntries(entries, 'gpu')[0].entry.name, 'b');
});

test('resolves the best catalog entry from a prompt', async () => {
  const root = await makeWorkspace();
  try {
    const resolved = await resolveExperience({ prompt: 'GPU 时间线怎么看 gap', cwd: root, io: nodeIo() });
    assert.equal(resolved.reason, 'prompt');
    assert.equal(resolved.entry.name, 'demo-entry');
    assert.equal(resolved.path, 'experiences/demo/demo-entry.md');
    assert.match(resolved.source, /# GPU 时间线阅读/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('resolves an explicit entry name, relative path and suffixless path', async () => {
  const root = await makeWorkspace();
  try {
    const io = nodeIo();
    const byName = await resolveExperience({ doc: 'demo-entry', prompt: 'x', cwd: root, io });
    assert.equal(byName.reason, 'explicit');
    assert.equal(byName.path, 'experiences/demo/demo-entry.md');

    const byPath = await resolveExperience({ doc: 'experiences/demo/demo-entry.md', prompt: 'x', cwd: root, io });
    assert.equal(byPath.reason, 'explicit');
    assert.match(byPath.source, /# GPU 时间线阅读/);

    const bySuffixless = await resolveExperience({ doc: 'experiences/demo/demo-entry', prompt: 'x', cwd: root, io });
    assert.equal(bySuffixless.reason, 'explicit');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('fails loudly on a weak prompt, a missing document and a missing catalog', async () => {
  const root = await makeWorkspace();
  try {
    const io = nodeIo();
    await assert.rejects(
      () => resolveExperience({ prompt: '完全无关的内容 zzzz', cwd: root, io }),
      /does not clearly match an experience entry/,
    );
    await assert.rejects(
      () => resolveExperience({ doc: 'missing.md', prompt: 'x', cwd: root, io }),
      /experience document not found/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }

  const empty = await mkdtemp(join(tmpdir(), 'dsh-html-experience-empty-'));
  try {
    await assert.rejects(
      () => resolveExperience({ prompt: 'gpu', cwd: empty, io: nodeIo() }),
      /no catalog.json found/,
    );
  } finally {
    await rm(empty, { recursive: true, force: true });
  }
});

test('loadCatalog walks up from a nested directory', async () => {
  const root = await makeWorkspace();
  try {
    const nested = join(root, 'experiences', 'demo');
    const result = await loadCatalog(nested, nodeIo());
    assert.equal(result.directory, root);
    assert.equal(result.catalog.entries.length, 2);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
