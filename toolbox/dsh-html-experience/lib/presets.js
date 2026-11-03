/**
 * Style presets for the generated reading page.
 *
 * Every preset owns a typographic voice (font stacks, radius, measure) and a
 * light/dark token palette. The generated HTML exposes these presets in its
 * in-page settings panel, so a reader can switch voice and appearance without
 * regenerating the file. All values are plain strings that end up in CSS
 * custom properties; do not put untrusted input here.
 */

const SERIF = '"Noto Serif SC", "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC", STSong, Georgia, serif';
const SANS = '"Noto Sans SC", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif';
const MONO = '"JetBrains Mono", "Cascadia Code", "SFMono-Regular", Menlo, Consolas, monospace';

/**
 * @typedef {{ bg: string, surface: string, text: string, soft: string, mute: string,
 *   border: string, accent: string, accentContrast: string, accentSoft: string,
 *   codeBg: string, codeText: string, mark: string, shadow: string }} Palette
 * @typedef {{ label: string, tagline: string, appearance: 'light'|'dark',
 *   body: string, heading: string, mono: string, radius: string, measure: number,
 *   scale: number, swatch: string, light: Palette, dark: Palette }} Preset
 */

/** @type {Record<string, Preset>} */
export const PRESETS = {
  paper: {
    label: '纸张',
    tagline: '暖白纸感与衬线正文，适合长时间阅读',
    appearance: 'light',
    body: SERIF,
    heading: SERIF,
    mono: MONO,
    radius: '12px',
    measure: 68,
    scale: 1,
    swatch: '#a34d2a',
    light: {
      bg: '#faf7f2',
      surface: '#fffdf9',
      text: '#2b2622',
      soft: '#554c45',
      mute: '#756a60',
      border: '#e5ddd2',
      accent: '#a34d2a',
      accentContrast: '#fffdf9',
      accentSoft: '#f3e4da',
      codeBg: '#f2ece3',
      codeText: '#3a3129',
      mark: '#f6e3a1',
      shadow: 'rgba(60, 40, 20, 0.09)',
    },
    dark: {
      bg: '#1b1815',
      surface: '#24201c',
      text: '#ede5da',
      soft: '#c9bfb2',
      mute: '#9b9083',
      border: '#3a342e',
      accent: '#e8a07a',
      accentContrast: '#2b1c12',
      accentSoft: '#3a2a22',
      codeBg: '#2c2722',
      codeText: '#e8dfd3',
      mark: '#5c4a1f',
      shadow: 'rgba(0, 0, 0, 0.42)',
    },
  },
  slate: {
    label: '蓝灰',
    tagline: '清晰的无衬线技术文档风，信息密度高',
    appearance: 'light',
    body: SANS,
    heading: SANS,
    mono: MONO,
    radius: '10px',
    measure: 70,
    scale: 1,
    swatch: '#2563eb',
    light: {
      bg: '#f6f8fb',
      surface: '#ffffff',
      text: '#17212f',
      soft: '#475569',
      mute: '#5b6b80',
      border: '#dbe3ec',
      accent: '#2563eb',
      accentContrast: '#ffffff',
      accentSoft: '#e3edff',
      codeBg: '#eef3f9',
      codeText: '#1e293b',
      mark: '#fde68a',
      shadow: 'rgba(23, 33, 47, 0.08)',
    },
    dark: {
      bg: '#0b1220',
      surface: '#111a2b',
      text: '#e2e8f0',
      soft: '#b6c2d1',
      mute: '#8494a8',
      border: '#243146',
      accent: '#60a5fa',
      accentContrast: '#06111f',
      accentSoft: '#1a2c4a',
      codeBg: '#0f172a',
      codeText: '#e2e8f0',
      mark: '#5b4a12',
      shadow: 'rgba(0, 0, 0, 0.5)',
    },
  },
  contrast: {
    label: '高对比',
    tagline: '更大字号与纯黑白对比，优先可读性',
    appearance: 'light',
    body: SANS,
    heading: SANS,
    mono: MONO,
    radius: '4px',
    measure: 62,
    scale: 1.08,
    swatch: '#9a3412',
    light: {
      bg: '#ffffff',
      surface: '#ffffff',
      text: '#000000',
      soft: '#1a1a1a',
      mute: '#3d3d3d',
      border: '#000000',
      accent: '#9a3412',
      accentContrast: '#ffffff',
      accentSoft: '#ffe8d6',
      codeBg: '#f4f4f4',
      codeText: '#000000',
      mark: '#ffe066',
      shadow: 'rgba(0, 0, 0, 0.14)',
    },
    dark: {
      bg: '#000000',
      surface: '#0a0a0a',
      text: '#ffffff',
      soft: '#eeeeee',
      mute: '#cfcfcf',
      border: '#ffffff',
      accent: '#fdba74',
      accentContrast: '#1a0a02',
      accentSoft: '#3a1d0b',
      codeBg: '#111111',
      codeText: '#ffffff',
      mark: '#6b5a00',
      shadow: 'rgba(255, 255, 255, 0.18)',
    },
  },
  midnight: {
    label: '午夜',
    tagline: '低亮度青色调，适合夜间与投屏',
    appearance: 'dark',
    body: SANS,
    heading: SANS,
    mono: MONO,
    radius: '14px',
    measure: 72,
    scale: 1,
    swatch: '#67e8f9',
    light: {
      bg: '#eef2f7',
      surface: '#ffffff',
      text: '#101827',
      soft: '#334155',
      mute: '#5b6b80',
      border: '#d3dde9',
      accent: '#0e7490',
      accentContrast: '#ffffff',
      accentSoft: '#d9f2f7',
      codeBg: '#e8eef5',
      codeText: '#12303c',
      mark: '#a5f3fc',
      shadow: 'rgba(16, 24, 39, 0.09)',
    },
    dark: {
      bg: '#0a1014',
      surface: '#101a20',
      text: '#d9f2f2',
      soft: '#a9c7c9',
      mute: '#7d9a9d',
      border: '#1f3339',
      accent: '#67e8f9',
      accentContrast: '#04222a',
      accentSoft: '#12313a',
      codeBg: '#0d171c',
      codeText: '#d9f2f2',
      mark: '#164e63',
      shadow: 'rgba(0, 0, 0, 0.55)',
    },
  },
};

/** Ordered preset names, used for CLI validation and the settings panel. */
export const PRESET_NAMES = Object.keys(PRESETS);

/** Themes accepted by the renderer. `auto` follows prefers-color-scheme. */
export const THEMES = ['auto', 'light', 'dark'];

/** Category labels used by the page header and chips. */
export const CATEGORY_LABELS = {
  methodology: '方法',
  compile: '编译',
  crash: '崩溃',
  correctness: '正确性',
  performance: '性能',
  engineering: '工程',
  other: '其他',
};

/**
 * Resolve a preset name, throwing a message that lists valid names.
 * @param {string|undefined} name
 * @returns {Preset}
 */
export function getPreset(name) {
  const preset = PRESETS[name ?? ''];
  if (!preset) {
    throw new Error(`unknown style preset "${name}". Valid presets: ${PRESET_NAMES.join(', ')}`);
  }
  return preset;
}

/**
 * Short preset summaries for the CLI and documentation.
 * @returns {{ name: string, label: string, appearance: string, tagline: string }[]}
 */
export function presetSummaries() {
  return PRESET_NAMES.map((name) => ({
    name,
    label: PRESETS[name].label,
    appearance: PRESETS[name].appearance,
    tagline: PRESETS[name].tagline,
  }));
}
