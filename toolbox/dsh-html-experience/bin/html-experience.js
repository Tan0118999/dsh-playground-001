#!/usr/bin/env node
/**
 * html-experience — render one experience entry as a standalone HTML reading page.
 *
 * Usage:
 *   node bin/html-experience.js --doc <path|entry-name> --prompt "<text>" [options]
 *   node bin/html-experience.js --prompt "<text>" [--doc <path>]     # match catalog.json
 *   node bin/html-experience.js --list-presets
 *
 * Inputs:
 *   --doc, -d          Experience .md path (workspace-relative or absolute) or catalog name.
 *                      Omit to match --prompt against catalog.json.
 *   --prompt, -p       Reader-facing note/request; free-form, may be empty for the CLI.
 *   --prompt-file      Read the prompt from a UTF-8 file (overrides --prompt).
 *   --style, -s        paper | slate | contrast | midnight (default paper).
 *   --theme            auto | light | dark (default auto).
 *   --accent           Hex accent colour, e.g. #a34d2a.
 *   --width            Body measure in characters, 40-120.
 *   --font-size        Base font size in pixels, 12-24.
 *   --out, -o          Output HTML path (default reading-<slug>.html in cwd).
 *   --title            Page title override.
 *   --no-embed-images  Leave local image URLs untouched instead of inlining them.
 *   --force, -f        Overwrite an existing output file.
 *   --json             Print a machine-readable result object.
 *   --help, -h         Show this help.
 *
 * Outputs:
 *   A self-contained UTF-8 HTML file and a one-line summary on stdout (or only
 *   the summary under --dry-run). Exit code 0 on success; 2 for usage errors;
 *   1 when the document cannot be resolved or rendered, or the output cannot
 *   be written.
 *
 * Failure behavior:
 *   Unknown presets, a prompt that does not match any catalog entry, a missing
 *   document, invalid numeric options and unwritable output paths fail loudly;
 *   the tool never invents a document or silently overwrites an existing file.
 */

import { readFile, stat, writeFile } from 'node:fs/promises';
import { extname, isAbsolute, resolve } from 'node:path';
import { renderReadingPage } from '../lib/render.js';
import { resolveExperience } from '../lib/resolve.js';
import { PRESET_NAMES, presetSummaries } from '../lib/presets.js';

const HELP = `Usage: node bin/html-experience.js [options]

把一条经验文档渲染成独立的 HTML 阅读页。

  --doc, -d <path|name>     经验文档路径或 catalog 条目名
  --prompt, -p <text>       读者提示/需求，内容不限
  --prompt-file <path>      从文件读取提示词
  --style, -s <preset>      ${PRESET_NAMES.join(' | ')}（默认 paper）
  --theme <auto|light|dark> 明暗（默认 auto）
  --accent <#rrggbb>        覆盖强调色
  --width <40-120>          正文行宽（字符）
  --font-size <12-24>       正文字号（px）
  --out, -o <path>          输出路径（默认 reading-<slug>.html）
  --title <text>            覆盖标题
  --no-embed-images         不内嵌本地图片
  --force, -f               覆盖已存在的输出文件
  --dry-run                 只生成并检查，不写文件
  --list-presets            列出风格预设
  --json                    输出机器可读结果
  --help, -h                显示本说明

示例：
  node bin/html-experience.js \\
    --doc experiences/dsh-creator-mode/dsh-model-architecture-visualization.md \\
    --prompt "读者第一次用创造模式做插件，先看清单契约和验收" \\
    --style slate --out /tmp/reading.html
`;

/**
 * @param {string[]} argv
 */
function parseArgs(argv) {
  /** @type {Record<string, string|boolean>} */
  const values = {};
  /** @type {string[]} */
  const positional = [];
  const stringOptions = new Set([
    'doc',
    'd',
    'prompt',
    'p',
    'prompt-file',
    'style',
    's',
    'theme',
    'accent',
    'width',
    'font-size',
    'out',
    'o',
    'title',
  ]);
  const booleanOptions = new Set([
    'no-embed-images',
    'force',
    'f',
    'json',
    'dry-run',
    'list-presets',
    'help',
    'h',
  ]);

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--') {
      positional.push(...argv.slice(index + 1));
      break;
    }
    if (!argument.startsWith('--')) {
      positional.push(argument);
      continue;
    }
    const equals = argument.indexOf('=');
    const key = equals === -1 ? argument.slice(2) : argument.slice(2, equals);
    const inline = equals === -1 ? undefined : argument.slice(equals + 1);
    if (booleanOptions.has(key)) {
      values[key] = true;
      continue;
    }
    if (!stringOptions.has(key)) throw new Error(`unknown option --${key}\n\n${HELP}`);
    if (inline !== undefined) {
      values[key] = inline;
      continue;
    }
    const next = argv[index + 1];
    if (next === undefined) throw new Error(`option --${key} needs a value`);
    values[key] = next;
    index += 1;
  }
  return { values, positional };
}

/**
 * @param {string} path
 */
async function exists(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

async function main() {
  let parsed;
  try {
    parsed = parseArgs(process.argv.slice(2));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(2);
  }
  const { values, positional } = parsed;

  if (values.help || values.h) {
    process.stdout.write(HELP);
    return;
  }
  if (values['list-presets']) {
    if (values.json) {
      process.stdout.write(JSON.stringify(presetSummaries(), null, 2) + '\n');
    } else {
      for (const preset of presetSummaries()) {
        process.stdout.write(
          `${preset.name.padEnd(10)} ${preset.label.padEnd(6)} 默认${preset.appearance.padEnd(5)} ${preset.tagline}\n`,
        );
      }
    }
    return;
  }

  const cwd = process.cwd();
  const io = {
    isFile: exists,
    readText: (path) => readFile(path, 'utf8'),
    async readBytes(path) {
      const info = await stat(path);
      if (info.size > 8 * 1024 * 1024) throw new Error(`image is larger than 8 MB: ${path}`);
      return readFile(path);
    },
  };

  let prompt = String(values.prompt ?? values.p ?? '').trim();
  if (values['prompt-file']) {
    prompt = (await readFile(String(values['prompt-file']), 'utf8')).trim();
  }
  if (!prompt && positional.length > 0) prompt = positional.join(' ').trim();
  if (!prompt) throw new Error('a prompt is required (--prompt or --prompt-file)');

  const resolved = await resolveExperience({
    doc: /** @type {string|undefined} */ (values.doc ?? values.d),
    prompt,
    cwd,
    io,
  });

  const page = await renderReadingPage({
    source: resolved.source,
    sourcePath: resolved.path,
    sourceAbsolutePath: resolved.absolutePath,
    prompt,
    style: /** @type {string|undefined} */ (values.style ?? values.s),
    theme: /** @type {string|undefined} */ (values.theme),
    accent: /** @type {string|undefined} */ (values.accent),
    width: values.width === undefined ? undefined : Number(values.width),
    fontSize: values['font-size'] === undefined ? undefined : Number(values['font-size']),
    title: /** @type {string|undefined} */ (values.title),
    embedImages: values['no-embed-images'] !== true,
    io,
  });

  const requested = String(values.out ?? values.o ?? '');
  let outputPath = requested
    ? isAbsolute(requested)
      ? requested
      : resolve(cwd, requested)
    : resolve(cwd, `reading-${page.slug}.html`);

  if (!values.force && values.f !== true && (await exists(outputPath))) {
    const extension = extname(outputPath);
    const base = outputPath.slice(0, outputPath.length - extension.length);
    for (let attempt = 2; attempt < 100; attempt += 1) {
      const candidate = `${base}-${attempt}${extension}`;
      if (!(await exists(candidate))) {
        outputPath = candidate;
        break;
      }
    }
  }

  if (values['dry-run'] !== true) await writeFile(outputPath, page.html, 'utf8');

  const result = {
    path: outputPath,
    dry_run: values['dry-run'] === true,
    source: resolved.path,
    title: page.title,
    style: page.style,
    theme: page.theme,
    sections: page.sections,
    minutes: page.minutes,
    bytes: page.bytes,
    warnings: page.warnings,
  };
  if (values.json) {
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
    return;
  }
  process.stdout.write(
    `${values['dry-run'] === true ? 'dry-run：将生成' : 'HTML 阅读页已生成'}：${outputPath}\n源文档：${resolved.path} · 风格：${page.style}/${page.theme} · ${page.sections} 小节 · 约 ${page.minutes} 分钟 · ${page.bytes} 字节\n`,
  );
  for (const warning of page.warnings) process.stdout.write(`提示：${warning}\n`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
