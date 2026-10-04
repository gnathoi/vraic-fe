// Appearance: PatternFly theme (Default | Red Hat "Project Felt"), colour scheme (Light | Dark | System) and contrast
// (Default | Glass | High contrast), applied as classes on <html>. Applied before the first render to avoid a flash.
import { useSyncExternalStore } from 'react';

export type Theme = 'default' | 'felt';
export type Scheme = 'light' | 'dark' | 'system';
export type Contrast = 'default' | 'glass' | 'high';
export interface Prefs { theme: Theme; scheme: Scheme; contrast: Contrast }
export interface Resolved extends Prefs {
  dark: boolean; glass: boolean; high: boolean;
  /** Why glass is unavailable (OS asks for less transparency or more contrast), or '' when it is allowed. */
  glassBlocked: string;
  key: string;
}

const KEY = 'jfe.appearance';
const DEFAULTS: Prefs = { theme: 'felt', scheme: 'dark', contrast: 'glass' };
const mq = (q: string) => (typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(q) : null);
const qDark = mq('(prefers-color-scheme: dark)');
const qLessTransparency = mq('(prefers-reduced-transparency: reduce)');
const qMoreContrast = mq('(prefers-contrast: more)');

function load(): Prefs {
  try {
    const p = JSON.parse(localStorage.getItem(KEY) || '{}');
    return {
      theme: p.theme === 'default' ? 'default' : DEFAULTS.theme,
      scheme: ['light', 'dark', 'system'].includes(p.scheme) ? p.scheme : DEFAULTS.scheme,
      contrast: ['default', 'glass', 'high'].includes(p.contrast) ? p.contrast : DEFAULTS.contrast,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

/** URL overrides for recording (?theme=felt&scheme=dark&contrast=glass); applied on top of storage, never persisted. */
function urlOverrides(p: Prefs): Prefs {
  if (typeof location === 'undefined') return p;
  const q = new URLSearchParams(location.search);
  const pick = <T extends string>(k: string, allowed: readonly T[], cur: T) => (allowed.includes(q.get(k) as T) ? (q.get(k) as T) : cur);
  return {
    theme: pick('theme', ['default', 'felt'] as const, p.theme),
    scheme: pick('scheme', ['light', 'dark', 'system'] as const, p.scheme),
    contrast: pick('contrast', ['default', 'glass', 'high'] as const, p.contrast),
  };
}

let prefs = urlOverrides(load());
let resolved = resolve(prefs);
const listeners = new Set<() => void>();

function resolve(p: Prefs): Resolved {
  const osHigh = !!qMoreContrast?.matches;
  const glassBlocked = qLessTransparency?.matches ? 'Your system asks for reduced transparency.' : osHigh ? 'Your system asks for more contrast.' : '';
  const high = p.contrast === 'high' || osHigh;
  const glass = p.contrast === 'glass' && !glassBlocked && !high; // never glass and high contrast together
  const dark = p.scheme === 'dark' || (p.scheme === 'system' && !!qDark?.matches);
  return { ...p, dark, glass, high, glassBlocked, key: `${p.theme}-${dark ? 'dark' : 'light'}-${glass ? 'glass' : high ? 'high' : 'default'}` };
}

function apply() {
  resolved = resolve(prefs);
  const c = document.documentElement.classList;
  c.toggle('pf-v6-theme-felt', resolved.theme === 'felt');
  c.toggle('pf-v6-theme-dark', resolved.dark);
  c.toggle('pf-v6-theme-glass', resolved.glass);
  c.toggle('pf-v6-theme-high-contrast', resolved.high);
  document.documentElement.style.colorScheme = resolved.dark ? 'dark' : 'light';
  listeners.forEach((l) => l());
}

/** Call once in main.tsx before rendering. Also follows live OS changes (scheme, transparency, contrast). */
export function initAppearance() {
  apply();
  [qDark, qLessTransparency, qMoreContrast].forEach((q) => q?.addEventListener('change', apply));
}

export function setPrefs(next: Partial<Prefs>) {
  prefs = { ...prefs, ...next };
  try { localStorage.setItem(KEY, JSON.stringify(prefs)); } catch { /* storage unavailable: choice lasts for this page only */ }
  apply();
}

export const useAppearance = () => useSyncExternalStore((l) => { listeners.add(l); return () => listeners.delete(l); }, () => resolved);
