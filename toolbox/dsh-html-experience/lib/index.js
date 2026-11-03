/**
 * DSH (DeepSeek Harness) plugin entry.
 *
 * The plugin exposes one model tool (`html_experience`) and one human command
 * (`/html`). Both accept an unrestricted prompt, resolve a specific experience
 * document from the library, render it as a self-contained HTML reading page,
 * and write that page into the workspace.
 *
 * It deliberately imports no DSH packages: the tool definition is a plain
 * compiled JSON-schema definition, so the plugin can be installed into any
 * profile that provides `tools`, `commands`, `fs` and `workingDirectory`.
 */

import { extname, isAbsolute, join, resolve } from 'node:path';
import { renderReadingPage } from './render.js';
import { resolveExperience } from './resolve.js';
import { PRESET_NAMES, PRESETS, THEMES, presetSummaries } from './presets.js';

export const name = 'html-experience';
export const inject = ['tools', 'commands', 'fs', 'workingDirectory'];

const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const DEFAULT_OUTPUT_PREFIX = 'reading';

const USAGE = `Usage: /html <提示词> [options]

把一条经验文档渲染成独立的 HTML 阅读页。提示词会显示为“读者提示”，
省略 --doc 时也会用来匹配 catalog.json 中的条目。

选项：
  --doc, -d <path|name>     经验文档路径或条目名（默认按提示词匹配）
  --style, -s <preset>      风格预设：${PRESET_NAMES.join(' | ')}（默认 paper）
  --theme <auto|light|dark> 明暗（默认 auto，跟随系统）
  --accent <#rrggbb>        覆盖强调色
  --width <40-120>          正文行宽（字符）
  --font-size <12-24>       正文字号（px）
  --out, -o <path>          输出路径（默认 reading-<slug>.html）
  --title <text>            覆盖页面标题
  --no-embed-images         不内嵌本地图片
  --force, -f               覆盖已存在的输出文件
  --list-presets            只列出风格预设
  --help                    显示本说明

提示词与选项冲突时，用“ -- ”分隔，例如：
  /html --style midnight -- 把验收步骤整理成读者速查`;

/**
 * Shell-like tokenizer for command options (quotes supported; no expansion).
 * @param {string} text
 * @returns {string[]}
 */
function tokenize(text) {
  const tokens = [];
  const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    tokens.push(match[1] ?? match[2] ?? match[3]);
  }
  return tokens;
}

/**
 * Parse the free-form text following `/html`.
 * Everything after the first ` -- ` is kept verbatim as the prompt.
 * @param {string} rawInput
 */
export function parseCommandInput(rawInput) {
  const raw = String(rawInput ?? '').replace(/^\s+/, '');
  const delimiter = raw.indexOf(' -- ');
  const optionText = delimiter === -1 ? raw : raw.slice(0, delimiter);
  let prompt = delimiter === -1 ? '' : raw.slice(delimiter + 4).trim();

  const tokens = tokenize(optionText);
  /** @type {Record<string, string|boolean>} */
  const options = {};
  /** @type {string[]} */
  const loose = [];

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!token.startsWith('--')) {
      loose.push(token);
      continue;
    }
    const equals = token.indexOf('=');
    const key = equals === -1 ? token.slice(2) : token.slice(2, equals);
    const inlineValue = equals === -1 ? undefined : token.slice(equals + 1);
    const takeValue = () => {
      if (inlineValue !== undefined) return inlineValue;
      const next = tokens[index + 1];
      if (next === undefined || next.startsWith('--')) throw new Error(`option --${key} needs a value`);
      index += 1;
      return next;
    };
    switch (key) {
      case 'doc':
      case 'd':
        options.doc = takeValue();
        break;
      case 'style':
      case 's':
        options.style = takeValue();
        break;
      case 'theme':
        options.theme = takeValue();
        break;
      case 'accent':
        options.accent = takeValue();
        break;
      case 'width':
        options.width = takeValue();
        break;
      case 'font-size':
        options.fontSize = takeValue();
        break;
      case 'out':
      case 'o':
        options.out = takeValue();
        break;
      case 'title':
        options.title = takeValue();
        break;
      case 'no-embed-images':
        options.embedImages = false;
        break;
      case 'force':
      case 'f':
        options.overwrite = true;
        break;
      case 'list-presets':
        options.listPresets = true;
        break;
      case 'help':
      case 'h':
        options.help = true;
        break;
      default:
        throw new Error(`unknown option --${key}\n\n${USAGE}`);
    }
  }

  if (!prompt) prompt = loose.join(' ').trim();
  return { prompt, ...options };
}

/**
 * @param {string} text
 * @param {string} cwd
 */
function defaultOutput(text, cwd) {
  return join(cwd, `${DEFAULT_OUTPUT_PREFIX}-${text}.html`);
}

/** @param {unknown} value */
function numeric(value) {
  if (value === undefined || value === '') return undefined;
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`expected a number, got "${String(value)}"`);
  return number;
}

/**
 * @param {import('@deepseek-ai/cordis').Context} ctx
 */
export function apply(ctx) {
  const io = {
    async isFile(path) {
      const target = await ctx.fs.resolve(path);
      const info = await ctx.fs.stat(target);
      return info?.type === 'file';
    },
    async readText(path) {
      const target = await ctx.fs.resolve(path);
      return ctx.fs.readText(target);
    },
    async readBytes(path) {
      const target = await ctx.fs.resolve(path);
      return ctx.fs.readBytes(target, undefined, MAX_IMAGE_BYTES);
    },
  };

  /**
   * @param {{
   *   prompt: string, doc?: string, style?: string, theme?: string, accent?: string,
   *   width?: number|string, fontSize?: number|string, out?: string, title?: string,
   *   embedImages?: boolean, overwrite?: boolean
   * }} request
   * @param {{ agent: import('@deepseek-ai/dsh-agent').Agent|undefined, signal?: AbortSignal }} context
   */
  async function generate(request, context) {
    const prompt = String(request.prompt ?? '').trim();
    if (!prompt) throw new Error('prompt is required and may contain any text');
    const signal = context.signal;
    const cwd = context.agent
      ? await ctx.workingDirectory.ensure(context.agent, signal)
      : process.cwd();

    const resolved = await resolveExperience({ doc: request.doc, prompt, cwd, io });
    const width = numeric(request.width);
    const fontSize = numeric(request.fontSize);

    const page = await renderReadingPage({
      source: resolved.source,
      sourcePath: resolved.path,
      sourceAbsolutePath: resolved.absolutePath,
      prompt,
      style: request.style,
      theme: request.theme,
      accent: request.accent,
      width,
      fontSize,
      title: request.title,
      embedImages: request.embedImages !== false,
      io,
    });

    const outputPath = request.out
      ? isAbsolute(String(request.out))
        ? String(request.out)
        : resolve(cwd, String(request.out))
      : defaultOutput(page.slug, cwd);

    let target;
    if (request.overwrite) {
      target = await ctx.fs.resolve(outputPath, { cwd, signal });
    } else {
      const extension = extname(outputPath);
      const base = outputPath.slice(0, outputPath.length - extension.length);
      for (let attempt = 0; attempt < 100; attempt += 1) {
        const candidate = attempt === 0 ? outputPath : `${base}-${attempt + 1}${extension}`;
        target = await ctx.fs.resolve(candidate, { cwd, signal });
        if (!(await ctx.fs.stat(target, signal))) break;
        target = undefined;
      }
      if (!target) throw new Error(`could not find a free output name near ${outputPath}`);
    }

    const info = await ctx.fs.stat(target, signal);
    const actor = context.actor;
    let intent;
    if (info) {
      ctx.emit?.('fs/observed', target, { kind: 'present', version: info.version }, actor);
      intent =
        typeof ctx.waterfall === 'function'
          ? await ctx.waterfall('fs/write-intent', target, actor, () => ({
              kind: 'replaceIfVersion',
              version: info.version,
            }))
          : { kind: 'replaceIfVersion', version: info.version };
    } else {
      ctx.emit?.('fs/observed', target, { kind: 'absent' }, actor);
      intent =
        typeof ctx.waterfall === 'function'
          ? await ctx.waterfall('fs/write-intent', target, actor, () => ({ kind: 'createIfAbsent' }))
          : { kind: 'createIfAbsent' };
    }
    await ctx.fs.writeText(target, page.html, intent, signal);

    return {
      path: ctx.fs.processPath(target),
      display_path: target.displayPath,
      source: resolved.path,
      title: page.title,
      style: page.style,
      theme: page.theme,
      sections: page.sections,
      minutes: page.minutes,
      bytes: page.bytes,
      presets: PRESET_NAMES,
      warnings: page.warnings,
    };
  }

  ctx.tools.register({
    name: 'html_experience',
    description:
      'Render a specific experience-library Markdown document as a self-contained, reader-facing HTML page. Use it when a person wants a styled reading page instead of raw Markdown. The prompt is free-form and unrestricted: it becomes the page reader note and, when doc is omitted, selects the experience entry from catalog.json. The generated page carries reader controls for style preset, light/dark, type scale, measure and accent.',
    parameters: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description:
            "Reader's request, note or instruction. Free-form and unrestricted. Displayed as the page's reader note and used to match an entry when doc is omitted.",
        },
        doc: {
          type: 'string',
          description:
            'Experience document: workspace-relative or absolute .md path, or a catalog entry name. Omit to match the prompt against catalog.json.',
        },
        style: {
          type: 'string',
          enum: PRESET_NAMES,
          description: `Visual preset (default paper). ${PRESET_NAMES.map(
            (preset) => `${preset}: ${PRESETS[preset].tagline}`,
          ).join('; ')}.`,
        },
        theme: {
          type: 'string',
          enum: THEMES,
          description: 'Initial appearance (default auto, follows the reader OS).',
        },
        accent: {
          type: 'string',
          description: 'Optional hex accent color such as #a34d2a, overriding the preset accent.',
        },
        width: {
          type: 'integer',
          description: 'Body measure in characters, 40-120. Defaults to the preset value.',
        },
        font_size: {
          type: 'integer',
          description: 'Base body font size in pixels, 12-24. Defaults to 17 (scaled by preset).',
        },
        out: {
          type: 'string',
          description:
            'Output HTML path relative to the workspace, or absolute. Defaults to reading-<slug>.html.',
        },
        title: {
          type: 'string',
          description: 'Optional page title override; defaults to the document frontmatter title.',
        },
        embed_images: {
          type: 'boolean',
          description: 'Embed local images as data URIs (default true) so the page works offline.',
        },
        overwrite: {
          type: 'boolean',
          description:
            'Overwrite an existing output file instead of choosing a numbered suffix (default false).',
        },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          path: { type: 'string', description: 'Absolute process path.' },
          display_path: { type: 'string' },
          source: { type: 'string' },
          title: { type: 'string' },
          style: { type: 'string' },
          theme: { type: 'string' },
          sections: { type: 'integer' },
          minutes: { type: 'integer' },
          bytes: { type: 'integer' },
          presets: { type: 'array', items: { type: 'string' } },
          warnings: { type: 'array', items: { type: 'string' } },
        },
        required: [
          'path',
          'display_path',
          'source',
          'title',
          'style',
          'theme',
          'sections',
          'minutes',
          'bytes',
          'presets',
          'warnings',
        ],
      },
      render(_args, value) {
        const lines = [
          `HTML reading page generated: ${value.path}`,
          `Source: ${value.source}; style: ${value.style}; theme: ${value.theme}; sections: ${value.sections}; about ${value.minutes} min; ${value.bytes} bytes.`,
        ];
        if (value.warnings.length > 0) lines.push(`Warnings: ${value.warnings.join('; ')}`);
        lines.push(
          'The page is self-contained and reader-facing. If the reader should receive it as a deliverable, present the file.',
        );
        return [{ type: 'text', text: lines.join('\n') }];
      },
    },
    async execute(args, exec) {
      const value = await generate(
        {
          prompt: args.prompt,
          doc: args.doc,
          style: args.style,
          theme: args.theme,
          accent: args.accent,
          width: args.width,
          fontSize: args.font_size,
          out: args.out,
          title: args.title,
          embedImages: args.embed_images,
          overwrite: args.overwrite,
        },
        { agent: exec.agent, signal: exec.signal, actor: exec },
      );
      return value;
    },
  });

  ctx.commands.register({
    name: 'html',
    description: '把一条经验文档渲染成带风格预设的 HTML 阅读页',
    input: {
      hint: '<提示词> [--doc <path>] [--style paper|slate|contrast|midnight] [--out <path>]',
    },
    async handler(invocation) {
      try {
        const parsed = parseCommandInput(invocation.rawInput);
        if (parsed.help) return { kind: 'success', text: USAGE };
        if (parsed.listPresets) {
          return {
            kind: 'success',
            text: presetSummaries()
              .map((preset) => `${preset.name}（${preset.label}，默认 ${preset.appearance}）：${preset.tagline}`)
              .join('\n'),
          };
        }
        if (!parsed.prompt) return { kind: 'error', text: USAGE };

        const value = await generate(
          {
            prompt: parsed.prompt,
            doc: /** @type {string|undefined} */ (parsed.doc),
            style: /** @type {string|undefined} */ (parsed.style),
            theme: /** @type {string|undefined} */ (parsed.theme),
            accent: /** @type {string|undefined} */ (parsed.accent),
            width: /** @type {string|undefined} */ (parsed.width),
            fontSize: /** @type {string|undefined} */ (parsed.fontSize),
            out: /** @type {string|undefined} */ (parsed.out),
            title: /** @type {string|undefined} */ (parsed.title),
            embedImages: parsed.embedImages !== false,
            overwrite: parsed.overwrite === true,
          },
          { agent: invocation.agent, signal: invocation.signal, actor: invocation },
        );
        return {
          kind: 'success',
          text: `HTML 阅读页已生成：${value.display_path}\n源文档：${value.source} · 风格：${value.style}/${value.theme} · 约 ${value.minutes} 分钟`,
        };
      } catch (error) {
        return { kind: 'error', text: error instanceof Error ? error.message : String(error) };
      }
    },
  });
}
