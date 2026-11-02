/**
 * Render one experience document as a self-contained HTML reading page.
 *
 * The output has no external CSS, JS or fonts. Local images referenced by the
 * Markdown are embedded as data URIs by default, so the file can be opened
 * offline or attached to a review. Reader-facing style controls (preset,
 * appearance, type scale, measure, accent) are part of the page and persist in
 * localStorage.
 */

import { basename, dirname, extname, resolve } from 'node:path';
import { parseFrontmatter } from './frontmatter.js';
import { collectImageUrls, escapeHtml, renderMarkdown } from './markdown.js';
import {
  CATEGORY_LABELS,
  PRESETS,
  PRESET_NAMES,
  THEMES,
  getPreset,
} from './presets.js';

export const GENERATOR = 'dsh-html-experience/0.1.0';

const IMAGE_MIME = {
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.avif': 'image/avif',
};

/**
 * @param {string} text
 * @returns {string}
 */
function stripMarkdownTitle(text) {
  return String(text)
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_~]/g, '')
    .trim();
}

/**
 * @param {string} text
 */
function readingMinutes(text) {
  const cjk = (text.match(/[\p{Script=Han}]/gu) ?? []).length;
  const latin = (text.replace(/[\p{Script=Han}]/gu, ' ').match(/[A-Za-z0-9]+/g) ?? []).length;
  return Math.max(1, Math.round(cjk / 350 + latin / 220));
}

/**
 * @param {string} text
 */
function slugFrom(text) {
  const slug = String(text)
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'reading-page';
}

/**
 * @param {string} url
 * @param {string} sourceDirectory
 */
function localImagePath(url, sourceDirectory) {
  if (/^(?:https?:|data:|#)/i.test(url)) return null;
  const clean = url.split(/[?#]/)[0];
  if (!clean) return null;
  return resolve(sourceDirectory, clean);
}

/**
 * Read local images through the caller's IO and turn them into data URIs.
 * Missing or unsupported files are left untouched and reported as warnings.
 * @param {string} markdown
 * @param {{
 *   sourceDirectory: string,
 *   embedImages: boolean,
 *   io: { readBytes?: (path: string) => Promise<Uint8Array>, isFile?: (path: string) => Promise<boolean> }
 * }} options
 */
async function buildImageMap(markdown, options) {
  /** @type {Map<string, string>} */
  const map = new Map();
  /** @type {string[]} */
  const warnings = [];
  if (!options.embedImages || typeof options.io.readBytes !== 'function') {
    return { map, warnings };
  }
  for (const url of collectImageUrls(markdown)) {
    const path = localImagePath(url, options.sourceDirectory);
    if (!path) continue;
    const mime = IMAGE_MIME[extname(path).toLowerCase()];
    if (!mime) {
      warnings.push(`image not embedded (unsupported type): ${url}`);
      continue;
    }
    try {
      const bytes = await options.io.readBytes(path);
      map.set(url, `data:${mime};base64,${Buffer.from(bytes).toString('base64')}`);
    } catch {
      warnings.push(`image not embedded (unreadable): ${url}`);
    }
  }
  return { map, warnings };
}

/**
 * @param {Record<string, string>} palette
 */
function paletteVars(palette) {
  return [
    `--bg:${palette.bg}`,
    `--surface:${palette.surface}`,
    `--text:${palette.text}`,
    `--soft:${palette.soft}`,
    `--mute:${palette.mute}`,
    `--border:${palette.border}`,
    `--accent:${palette.accent}`,
    `--accent-contrast:${palette.accentContrast}`,
    `--accent-soft:${palette.accentSoft}`,
    `--code-bg:${palette.codeBg}`,
    `--code-text:${palette.codeText}`,
    `--mark:${palette.mark}`,
    `--shadow:${palette.shadow}`,
  ].join(';');
}

/** Generate the preset token blocks shared by every generated page. */
function presetCss() {
  /** @type {string[]} */
  const chunks = [];
  for (const name of PRESET_NAMES) {
    const preset = PRESETS[name];
    chunks.push(
      `html[data-preset="${name}"]{--font-body:${preset.body};--font-heading:${preset.heading};--font-mono:${preset.mono};--radius:${preset.radius};--measure:${preset.measure};--preset-scale:${preset.scale};}`,
    );
    chunks.push(`html[data-preset="${name}"][data-theme="light"]{${paletteVars(preset.light)}}`);
    chunks.push(`html[data-preset="${name}"][data-theme="dark"]{${paletteVars(preset.dark)}}`);
    chunks.push(`html[data-preset="${name}"][data-theme="auto"]{${paletteVars(preset.light)}}`);
    chunks.push(
      `@media (prefers-color-scheme: dark){html[data-preset="${name}"][data-theme="auto"]{${paletteVars(preset.dark)}}}`,
    );
  }
  return chunks.join('\n');
}

const BASE_CSS = String.raw`
*,*::before,*::after{box-sizing:border-box}
html{--reader-scale:1;font-size:calc(17px * var(--preset-scale,1) * var(--reader-scale,1));scroll-behavior:smooth;scroll-padding-top:24px;background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--font-body);line-height:1.85;text-rendering:optimizeLegibility;-webkit-font-smoothing:antialiased}
::selection{background:var(--accent-soft);color:var(--text)}
a{color:var(--accent);text-underline-offset:.18em;text-decoration-thickness:.06em}
a:hover{text-decoration-thickness:.12em}
:focus-visible{outline:3px solid var(--accent);outline-offset:3px;border-radius:3px}
img{max-width:100%;height:auto}
.skip-link{position:fixed;left:12px;top:-60px;z-index:50;background:var(--accent);color:var(--accent-contrast);padding:10px 14px;border-radius:8px;transition:top .15s}
.skip-link:focus{top:12px}
.progress{position:fixed;inset:0 0 auto 0;height:3px;z-index:40;background:transparent}
.progress span{display:block;height:100%;width:0;background:var(--accent)}
.shell{max-width:1280px;margin:0 auto;padding:0 32px}
.hero{padding:72px 0 36px;border-bottom:1px solid var(--border)}
.kicker{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:18px;font-family:var(--font-mono);font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
.badge{background:var(--accent-soft);color:var(--accent);border:1px solid var(--border);padding:3px 10px;border-radius:999px;font-size:.72rem;letter-spacing:.04em}
.hero h1{font-family:var(--font-heading);font-size:clamp(2rem,4.4vw,3.3rem);line-height:1.22;letter-spacing:-.015em;margin:0 0 18px;max-width:22em;text-wrap:balance}
.lede{max-width:44em;color:var(--soft);font-size:1.06rem;margin:0 0 24px}
.meta{display:flex;gap:18px;flex-wrap:wrap;align-items:center;color:var(--mute);font-family:var(--font-mono);font-size:.8rem}
.meta span{display:inline-flex;align-items:center;gap:6px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}
.chip{border:1px solid var(--border);border-radius:999px;padding:2px 10px;font-size:.74rem;color:var(--soft);background:var(--surface)}
.chip.stack{font-family:var(--font-mono);color:var(--mute)}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 292px;gap:56px;max-width:calc(var(--measure) * 1ch + 348px);margin:0 auto;padding:44px 0 80px;align-items:start}
.article{min-width:0;max-width:calc(var(--measure) * 1ch)}
.reader-note{background:var(--surface);border:1px solid var(--border);border-left:5px solid var(--accent);border-radius:var(--radius);padding:20px 24px;margin:0 0 40px;box-shadow:0 8px 30px var(--shadow)}
.reader-note .note-kicker{font-family:var(--font-mono);font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
.reader-note p{margin:.4em 0}
.reader-note p:first-child{margin-top:0}
.reader-note p:last-child{margin-bottom:0}
.note-heading{font-family:var(--font-heading);font-size:1.02rem;margin:.9em 0 .3em}
.article h2,.article h3,.article h4{font-family:var(--font-heading);line-height:1.35;text-wrap:balance;scroll-margin-top:28px}
.article h2{font-size:1.62rem;margin:2.4em 0 .8em;padding-top:1.1em;border-top:1px solid var(--border)}
.article h2:first-child{margin-top:0;padding-top:0;border-top:0}
.article h3{font-size:1.22rem;margin:1.9em 0 .6em}
.article h4{font-size:1.04rem;margin:1.6em 0 .5em}
.article p{margin:0 0 1.15em;color:var(--text)}
.article strong{color:var(--text);font-weight:700}
.article ul,.article ol{margin:0 0 1.2em;padding-left:1.5em}
.article li{margin:.28em 0}
.article li.task{list-style:none;margin-left:-1.3em}
.article li>ul,.article li>ol{margin:.3em 0 .5em}
.article blockquote{margin:1.4em 0;padding:14px 20px;border-left:4px solid var(--accent);background:var(--surface);border-radius:0 var(--radius) var(--radius) 0;color:var(--soft)}
.article blockquote p:last-child{margin-bottom:0}
.article hr{border:0;border-top:1px solid var(--border);margin:2.4em 0}
.article code{font-family:var(--font-mono);font-size:.88em;background:var(--code-bg);color:var(--code-text);padding:.14em .36em;border-radius:5px;overflow-wrap:anywhere}
.article mark{background:var(--mark);color:var(--text);padding:0 .15em}
.article .table-wrap{overflow-x:auto;margin:1.5em 0;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}
.article table{border-collapse:collapse;width:100%;font-size:.92rem;line-height:1.6}
.article th,.article td{padding:10px 14px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}
.article th{background:var(--code-bg);font-family:var(--font-heading);font-weight:700;white-space:nowrap}
.article tr:last-child td{border-bottom:0}
.article table code{background:transparent;padding:0}
.code-block{margin:1.5em 0;border:1px solid var(--border);border-radius:var(--radius);background:var(--code-bg);overflow:hidden}
.code-block figcaption{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:7px 12px;border-bottom:1px solid var(--border);background:var(--surface);font-family:var(--font-mono);font-size:.72rem;color:var(--mute)}
.code-block pre{margin:0;padding:16px 18px;overflow-x:auto;line-height:1.65}
.code-block code{background:transparent;color:var(--code-text);padding:0;font-size:.84rem;white-space:pre}
.copy-button{border:1px solid var(--border);background:transparent;color:var(--mute);font:inherit;padding:2px 9px;border-radius:6px;cursor:pointer}
.copy-button:hover{color:var(--text);border-color:var(--accent)}
.sidebar{position:sticky;top:26px;display:flex;flex-direction:column;gap:26px;max-height:calc(100vh - 52px);overflow:auto;scrollbar-width:thin}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px 18px 20px;box-shadow:0 8px 30px var(--shadow)}
.panel h2{font-family:var(--font-mono);font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;color:var(--mute);margin:0 0 12px}
.toc-list,.toc-sub{list-style:none;margin:0;padding:0}
.toc-list>li{margin:0 0 3px}
.toc-list a{display:block;padding:5px 8px;border-radius:6px;color:var(--soft);text-decoration:none;font-size:.86rem;line-height:1.45;border-left:2px solid transparent}
.toc-list a:hover{background:var(--code-bg);color:var(--text)}
.toc-list a.active{color:var(--accent);border-left-color:var(--accent);background:var(--accent-soft)}
.toc-sub{margin:2px 0 6px;padding-left:10px}
.toc-sub a{font-size:.8rem;color:var(--mute);padding:3px 8px}
.preset-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.preset-button{display:flex;gap:7px;align-items:center;text-align:left;padding:7px 8px;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--text);font:inherit;cursor:pointer;line-height:1.35}
.preset-button:hover{border-color:var(--accent)}
.preset-button[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft);box-shadow:inset 0 0 0 1px var(--accent)}
.preset-swatch{flex:none;width:16px;height:16px;border-radius:50%;background:var(--swatch);border:2px solid var(--surface);box-shadow:0 0 0 1px var(--border)}
.preset-button strong{display:block;font-size:.84rem}
.more-settings{margin:10px 0 2px;border-top:1px solid var(--border);padding-top:8px}
.more-settings summary{cursor:pointer;color:var(--soft);font-size:.8rem;list-style:none;padding:2px 0}
.more-settings summary::-webkit-details-marker{display:none}
.more-settings summary::before{content:'▸';display:inline-block;width:1.1em;color:var(--mute)}
.more-settings[open] summary::before{content:'▾'}
.more-settings .field{margin-top:12px;margin-bottom:10px}
.field{margin:0 0 14px}
.field legend,.field-label{display:block;font-family:var(--font-mono);font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:var(--mute);margin:0 0 6px;padding:0}
fieldset{border:0;margin:0;padding:0}
.segmented{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.segmented button{flex:1;border:0;border-right:1px solid var(--border);background:transparent;color:var(--soft);font:inherit;font-size:.8rem;padding:6px 4px;cursor:pointer}
.segmented button:last-child{border-right:0}
.segmented button[aria-pressed="true"]{background:var(--accent);color:var(--accent-contrast)}
.color-row{display:flex;align-items:center;gap:8px}
.color-row input[type="color"]{width:38px;height:30px;border:1px solid var(--border);border-radius:6px;background:transparent;padding:2px;cursor:pointer}
.color-row span{color:var(--mute);font-family:var(--font-mono);font-size:.74rem}
.reset-button{width:100%;margin-top:4px;border:1px solid var(--border);background:transparent;color:var(--soft);font:inherit;font-size:.8rem;padding:7px;border-radius:8px;cursor:pointer}
.reset-button:hover{color:var(--text);border-color:var(--accent)}
.settings-hint{margin:10px 0 0;color:var(--mute);font-size:.72rem;line-height:1.5}
.footer{border-top:1px solid var(--border);padding:26px 0 48px;color:var(--mute);font-size:.8rem;display:flex;flex-wrap:wrap;gap:8px 26px;justify-content:space-between}
.footer code{background:var(--code-bg);color:var(--code-text);padding:.12em .35em;border-radius:4px;font-family:var(--font-mono)}
.toc-empty{color:var(--mute);font-size:.82rem;margin:0}
.style-toggle,.drawer-close{display:none}
.drawer-backdrop{display:none}
@media (max-width:1000px){
  .style-toggle{display:inline-flex;align-items:center;gap:6px;position:fixed;right:16px;bottom:16px;z-index:55;border:1px solid var(--border);background:var(--accent);color:var(--accent-contrast);font:inherit;font-size:.86rem;padding:10px 14px;border-radius:999px;box-shadow:0 8px 30px var(--shadow);cursor:pointer}
  .drawer-backdrop{display:block;position:fixed;inset:0;z-index:58;background:rgba(0,0,0,.45)}
  .drawer-backdrop[hidden]{display:none}
  .drawer-open{overflow:hidden}
  .drawer-close{display:inline-flex;align-items:center;justify-content:center;position:absolute;top:12px;right:12px;border:1px solid var(--border);background:var(--surface);color:var(--soft);font:inherit;font-size:.78rem;padding:5px 10px;border-radius:8px;cursor:pointer}
  .sidebar{position:fixed;top:0;right:0;bottom:0;width:min(370px,92vw);z-index:60;background:var(--bg);padding:20px;transform:translateX(102%);transition:transform .2s ease;max-height:none;overflow:auto;box-shadow:-18px 0 48px var(--shadow);order:initial}
  .sidebar.open{transform:none}
  .sidebar .panel{box-shadow:none}
  .toc-list a{padding:7px 8px}
  .hero{padding:48px 0 26px}
}
@media (prefers-reduced-motion:reduce){
  .sidebar{transition:none}
}
@media (max-width:1000px){
  .shell{padding:0 22px}
  .layout{grid-template-columns:minmax(0,1fr);gap:34px;max-width:none}
  .article{max-width:100%}
}
@media (max-width:620px){
  .shell{padding:0 16px}
  .hero h1{font-size:clamp(1.7rem,8vw,2.4rem)}
  .lede{font-size:1rem}
  .reader-note{padding:16px 18px}
  .article h2{font-size:1.36rem}
  .article h3{font-size:1.12rem}
  .footer{display:block}
  .footer p{margin:0 0 8px}
}
@media print{
  html{font-size:11pt;background:#fff}
  body{background:#fff;color:#000}
  .shell{max-width:none;padding:0}
  .progress,.sidebar,.skip-link,.copy-button,.style-toggle,.drawer-backdrop{display:none!important}
  .layout{display:block;padding:0}
  .article{max-width:none}
  .hero{padding:0 0 18px}
  .reader-note{break-inside:avoid;box-shadow:none}
  .article h2{break-after:avoid}
  .article table,.code-block,.reader-note{break-inside:avoid}
  a{color:#000;text-decoration:underline}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *{transition:none!important;animation:none!important}
}
`;

const SETTINGS_SCRIPT = String.raw`
(() => {
  const KEY = 'dsh-html-experience:style:v1';
  const root = document.documentElement;
  const defaults = {
    preset: root.dataset.preset || 'paper',
    theme: root.dataset.theme || 'auto',
    readerScale: root.style.getPropertyValue('--reader-scale').trim() || '1',
    measure: root.style.getPropertyValue('--measure').trim(),
    accent: root.style.getPropertyValue('--accent').trim(),
  };
  let state = { ...defaults };
  try { state = { ...defaults, ...JSON.parse(localStorage.getItem(KEY) || '{}') }; } catch {}
  const presetButtons = [...document.querySelectorAll('button[data-preset]')];
  const themeButtons = [...document.querySelectorAll('button[data-theme-choice]')];
  const scaleButtons = [...document.querySelectorAll('button[data-scale]')];
  const measureButtons = [...document.querySelectorAll('button[data-measure]')];
  const accentInput = document.getElementById('accent-input');
  const themeColor = document.querySelector('meta[name="theme-color"]');

  function apply() {
    root.dataset.preset = state.preset;
    root.dataset.theme = state.theme;
    root.style.setProperty('--reader-scale', state.readerScale);
    if (state.measure) root.style.setProperty('--measure', state.measure);
    else root.style.removeProperty('--measure');
    if (state.accent) root.style.setProperty('--accent', state.accent);
    else root.style.removeProperty('--accent');
    presetButtons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.preset === state.preset)));
    themeButtons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.themeChoice === state.theme)));
    scaleButtons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.scale === state.readerScale)));
    measureButtons.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.measure === state.measure)));
    if (accentInput) accentInput.value = state.accent || getComputedStyle(root).getPropertyValue('--accent').trim() || '#a34d2a';
    if (themeColor) themeColor.setAttribute('content', getComputedStyle(root).getPropertyValue('--bg').trim() || '#faf7f2');
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch {}
  }

  document.addEventListener('click', (event) => {
    const preset = event.target.closest('button[data-preset]');
    if (preset) { state.preset = preset.dataset.preset; apply(); return; }
    const theme = event.target.closest('button[data-theme-choice]');
    if (theme) { state.theme = theme.dataset.themeChoice; apply(); return; }
    const scale = event.target.closest('button[data-scale]');
    if (scale) { state.readerScale = scale.dataset.scale; apply(); return; }
    const measure = event.target.closest('button[data-measure]');
    if (measure) { state.measure = measure.dataset.measure; apply(); return; }
    const reset = event.target.closest('#reset-style');
    if (reset) { state = { ...defaults }; apply(); return; }
    const copy = event.target.closest('[data-copy]');
    if (copy) {
      const block = copy.closest('.code-block');
      const code = block && block.querySelector('code');
      if (!code) return;
      const text = code.innerText;
      const done = () => { copy.textContent = '已复制'; setTimeout(() => { copy.textContent = '复制'; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, done);
      else {
        const area = document.createElement('textarea');
        area.value = text; document.body.appendChild(area); area.select();
        try { document.execCommand('copy'); } catch {}
        area.remove(); done();
      }
    }
  });

  if (accentInput) accentInput.addEventListener('input', () => { state.accent = accentInput.value; apply(); });

  const drawer = document.getElementById('reader-sidebar');
  const toggle = document.getElementById('style-toggle');
  const backdrop = document.getElementById('drawer-backdrop');
  const close = document.getElementById('drawer-close');
  if (drawer && innerWidth <= 1000) drawer.inert = true;
  let drawerReturnFocus = null;
  const setDrawer = (open) => {
    if (!drawer || !toggle) return;
    if (!open && drawerReturnFocus) { try { drawerReturnFocus.focus(); } catch {} drawerReturnFocus = null; }
    if (open) drawerReturnFocus = document.activeElement;
    drawer.classList.toggle('open', open);
    drawer.inert = innerWidth <= 1000 ? !open : false;
    toggle.setAttribute('aria-expanded', String(open));
    if (backdrop) backdrop.hidden = !open;
    document.documentElement.classList.toggle('drawer-open', open);
    if (open && close) close.focus();
  };
  if (toggle) toggle.addEventListener('click', () => setDrawer(!drawer.classList.contains('open')));
  if (backdrop) backdrop.addEventListener('click', () => setDrawer(false));
  if (close) close.addEventListener('click', () => setDrawer(false));
  if (drawer) drawer.addEventListener('click', (event) => { if (event.target.closest('a[href^="#"]')) setDrawer(false); });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') setDrawer(false); });
  addEventListener('resize', () => { if (innerWidth > 1000) setDrawer(false); else if (drawer && !drawer.classList.contains('open')) drawer.inert = true; });

  const progress = document.getElementById('progress-bar');
  const links = new Map([...document.querySelectorAll('.toc-list a[href^="#"]')].map((link) => [link.getAttribute('href').slice(1), link]));
  const headingNodes = [...document.querySelectorAll('.article h2[id], .article h3[id]')];
  const syncToc = () => {
    if (!links.size || !headingNodes.length) return;
    const probe = scrollY + innerHeight * 0.55;
    let activeId = headingNodes[0].id;
    for (const heading of headingNodes) {
      if (heading.getBoundingClientRect().top + scrollY <= probe) activeId = heading.id;
      else break;
    }
    links.forEach((link) => link.classList.remove('active'));
    const active = links.get(activeId);
    if (active) active.classList.add('active');
  };
  const onScroll = () => {
    if (progress) {
      const max = document.documentElement.scrollHeight - innerHeight;
      progress.style.width = max > 0 ? Math.min(100, (scrollY / max) * 100) + '%' : '0%';
    }
    syncToc();
  };
  addEventListener('scroll', onScroll, { passive: true });
  addEventListener('resize', onScroll);
  onScroll();

  apply();
})();
`;

/**
 * @param {{ level: number, text: string, id: string }[]} headings
 */
function renderToc(headings) {
  const top = headings.filter((heading) => heading.level === 2);
  if (top.length === 0) return '<p class="toc-empty">本文没有小节标题。</p>';
  /** @type {Map<string, { heading: { level:number,text:string,id:string }, children: {level:number,text:string,id:string}[] }>} */
  const groups = new Map();
  let current = null;
  for (const heading of headings) {
    if (heading.level === 2) {
      current = { heading, children: [] };
      groups.set(heading.id, current);
    } else if (heading.level === 3 && current) {
      current.children.push(heading);
    }
  }
  return [
    '<ul class="toc-list">',
    ...[...groups.values()].map(
      (group) =>
        `<li><a href="#${escapeHtml(group.heading.id)}">${escapeHtml(group.heading.text)}</a>${
          group.children.length
            ? `<ul class="toc-sub">${group.children
                .map(
                  (child) =>
                    `<li><a href="#${escapeHtml(child.id)}">${escapeHtml(child.text)}</a></li>`,
                )
                .join('')}</ul>`
            : ''
        }</li>`,
    ),
    '</ul>',
  ].join('\n');
}

function renderPresetButtons(activePreset) {
  return PRESET_NAMES.map(
    (name) => {
      const preset = PRESETS[name];
      return `<button type="button" class="preset-button" data-preset="${name}" aria-pressed="${
        name === activePreset ? 'true' : 'false'
      }" title="${escapeHtml(preset.tagline)}" aria-label="${escapeHtml(preset.label)}：${escapeHtml(preset.tagline)}">
  <span class="preset-swatch" style="--swatch:${preset.swatch}"></span>
  <strong>${escapeHtml(preset.label)}</strong>
</button>`;
    },
  ).join('\n');
}

function renderSettings(activePreset, theme) {
  return `<section class="panel settings" aria-label="阅读样式设置">
  <h2>阅读样式</h2>
  <fieldset class="field">
    <legend>风格预设</legend>
    <div class="preset-grid">
${renderPresetButtons(activePreset)}
    </div>
  </fieldset>
  <fieldset class="field">
    <legend>明暗</legend>
    <div class="segmented">
      <button type="button" data-theme-choice="auto" aria-pressed="${theme === 'auto'}">跟随系统</button>
      <button type="button" data-theme-choice="light" aria-pressed="${theme === 'light'}">浅色</button>
      <button type="button" data-theme-choice="dark" aria-pressed="${theme === 'dark'}">深色</button>
    </div>
  </fieldset>
  <details class="more-settings">
    <summary>字号、行宽与强调色</summary>
    <fieldset class="field">
      <legend>字号</legend>
      <div class="segmented">
        <button type="button" data-scale="0.92" aria-pressed="false">小</button>
        <button type="button" data-scale="1" aria-pressed="true">中</button>
        <button type="button" data-scale="1.12" aria-pressed="false">大</button>
      </div>
    </fieldset>
    <fieldset class="field">
      <legend>行宽</legend>
      <div class="segmented">
        <button type="button" data-measure="62" aria-pressed="false">窄</button>
        <button type="button" data-measure="70" aria-pressed="false">标准</button>
        <button type="button" data-measure="80" aria-pressed="false">宽</button>
      </div>
    </fieldset>
    <div class="field">
      <label class="field-label" for="accent-input">强调色</label>
      <div class="color-row">
        <input id="accent-input" type="color" value="${PRESETS[activePreset].swatch}" aria-label="自定义强调色">
        <span>可覆盖预设</span>
      </div>
    </div>
  </details>
  <button type="button" class="reset-button" id="reset-style">恢复默认</button>
  <p class="settings-hint">设置保存在本机浏览器，不会上传。</p>
</section>`;
}

/**
 * Render an experience entry into a complete HTML document.
 *
 * @param {{
 *   source: string,
 *   sourcePath: string,
 *   sourceAbsolutePath?: string,
 *   prompt?: string,
 *   style?: string,
 *   theme?: string,
 *   accent?: string,
 *   width?: number,
 *   fontSize?: number,
 *   title?: string,
 *   embedImages?: boolean,
 *   generatedAt?: string,
 *   io?: { readBytes?: (path: string) => Promise<Uint8Array> }
 * }} options
 */
export async function renderReadingPage(options) {
  const {
    source,
    sourcePath,
    sourceAbsolutePath,
    prompt = '',
    style = 'paper',
    theme = 'auto',
    accent,
    width,
    fontSize,
    title: titleOverride,
    embedImages = true,
    generatedAt = new Date().toISOString(),
  } = options;

  const preset = getPreset(style);
  if (!THEMES.includes(theme)) throw new Error(`unknown theme "${theme}". Valid themes: ${THEMES.join(', ')}`);
  if (accent !== undefined && accent !== '' && !/^#[0-9a-f]{3,8}$/i.test(accent)) {
    throw new Error(`accent must be a CSS hex color such as #a34d2a, got "${accent}"`);
  }
  if (width !== undefined && (!Number.isFinite(width) || width < 40 || width > 120)) {
    throw new Error(`width must be between 40 and 120 characters, got "${width}"`);
  }
  if (fontSize !== undefined && (!Number.isFinite(fontSize) || fontSize < 12 || fontSize > 24)) {
    throw new Error(`font-size must be between 12 and 24 pixels, got "${fontSize}"`);
  }

  const { data, body } = parseFrontmatter(source);
  const sourceDirectory = dirname(sourceAbsolutePath ?? resolve(sourcePath));

  let title = String(titleOverride ?? data.title ?? '').trim();
  let content = body;
  const firstH1 = /^\s*#\s+(.+?)\s*#*\s*(?:\n|$)/.exec(content);
  if (firstH1) {
    if (!title) title = stripMarkdownTitle(firstH1[1]);
    content = content.replace(/^\s*#\s+.+?(?:\n|$)/, '');
  }
  if (!title) title = basename(sourcePath).replace(/\.md$/i, '');

  const description = String(data.description ?? '').trim();
  const category = String(data.category ?? 'other');
  const updated = String(data.updated ?? '');
  const tags = Array.isArray(data.tags) ? data.tags.map(String) : [];
  const stacks = Array.isArray(data.stacks) ? data.stacks.map(String) : [];
  const entryName = String(data.name ?? '');

  const { map: imageMap, warnings } = await buildImageMap(content, {
    sourceDirectory,
    embedImages,
    io: options.io ?? {},
  });
  const rendered = renderMarkdown(content, { imageMap });
  const note = String(prompt ?? '').trim();
  const noteHtml = note
    ? renderMarkdown(note)
        .html.replace(/<h[1-6](?:\s+id="[^"]*")?>/g, '<h4 class="note-heading">')
        .replace(/<\/h[1-6]>/g, '</h4>')
    : '';
  const minutes = readingMinutes(rendered.text);
  const headingCount = rendered.headings.filter((heading) => heading.level === 2).length;
  const slug = slugFrom(entryName || title);
  const categoryLabel = CATEGORY_LABELS[category] ?? CATEGORY_LABELS.other;

  const rootStyle = [
    accent ? `--accent:${accent}` : '',
    width ? `--measure:${Number(width)}` : '',
    fontSize ? `--reader-scale:${(Number(fontSize) / 17).toFixed(3)}` : '',
  ]
    .filter(Boolean)
    .join(';');
  const rootStyleAttribute = rootStyle ? ` style="${escapeHtml(rootStyle)}"` : '';

  const heroMeta = [
    categoryLabel ? `<span>分类 · ${escapeHtml(categoryLabel)}</span>` : '',
    updated ? `<span>更新 · ${escapeHtml(updated)}</span>` : '',
    `<span>阅读 · 约 ${minutes} 分钟</span>`,
    `<span>小节 · ${headingCount}</span>`,
  ]
    .filter(Boolean)
    .join('');

  const html = `<!doctype html>
<html lang="zh-CN" data-preset="${escapeHtml(style)}" data-theme="${escapeHtml(theme)}"${rootStyleAttribute}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="${preset.light.bg}">
<meta name="generator" content="${GENERATOR}">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data: https: http:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src data:; base-uri 'none'; form-action 'none'">
<title>${escapeHtml(title)}</title>
<meta name="description" content="${escapeHtml(description || `${title} — 经验库阅读页`)}">
<style>
${presetCss()}
:root{--measure:${preset.measure};--reader-scale:1}
${BASE_CSS}
</style>
</head>
<body>
<a class="skip-link" href="#article">跳到正文</a>
<div class="progress" aria-hidden="true"><span id="progress-bar"></span></div>
<button type="button" class="style-toggle" id="style-toggle" aria-controls="reader-sidebar" aria-expanded="false">样式 / 目录</button>
<div class="drawer-backdrop" id="drawer-backdrop" hidden></div>
<div class="shell">
  <header class="hero">
    <div class="kicker"><span>Experience Library</span><span aria-hidden="true">/</span><span>${escapeHtml(categoryLabel)}</span></div>
    <h1>${escapeHtml(title)}</h1>
    ${description ? `<p class="lede">${escapeHtml(description)}</p>` : ''}
    <div class="meta">${heroMeta}</div>
    ${
      tags.length || stacks.length
        ? `<div class="chips">${tags
            .map((tag) => `<span class="chip">#${escapeHtml(tag)}</span>`)
            .join('')}${stacks.map((stack) => `<span class="chip stack">${escapeHtml(stack)}</span>`).join('')}</div>`
        : ''
    }
  </header>
  <div class="layout">
    <main class="article" id="article">
      ${
        noteHtml
          ? `<section class="reader-note" aria-label="读者提示">
        <div class="note-kicker">读者提示 · 由提示词生成</div>
        ${noteHtml}
      </section>`
          : ''
      }
      ${rendered.html}
    </main>
    <aside class="sidebar" id="reader-sidebar" aria-label="目录与阅读样式">
      <button type="button" class="drawer-close" id="drawer-close" aria-label="关闭样式与目录">关闭</button>
      ${renderSettings(style, theme)}
      <nav class="panel" aria-label="本文目录">
        <h2>目录</h2>
        ${renderToc(rendered.headings)}
      </nav>
    </aside>
  </div>
  <footer class="footer">
    <p>本页由 <code>${GENERATOR}</code> 从经验库文档生成；内容以源文件为准。</p>
    <p>源文件：<code>${escapeHtml(sourcePath)}</code> · 生成于 ${escapeHtml(generatedAt.slice(0, 10))}</p>
  </footer>
</div>
<script>
${SETTINGS_SCRIPT}
</script>
</body>
</html>
`;

  return {
    html,
    title,
    slug,
    description,
    category,
    tags,
    stacks,
    updated,
    sourcePath,
    style,
    theme,
    headings: rendered.headings,
    sections: headingCount,
    minutes,
    characters: rendered.text.length,
    bytes: Buffer.byteLength(html, 'utf8'),
    warnings,
  };
}
