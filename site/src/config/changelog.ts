export type Sdk = 'harness' | 'evals'
export type Language = 'python' | 'typescript'

export const LANGUAGE_META: Record<Language, { label: string; short: string }> = {
  python: { label: 'Python', short: 'Py' },
  typescript: { label: 'TypeScript', short: 'TS' },
}

export const SDK_META: Record<Sdk, { label: string; languages: Language[] }> = {
  harness: { label: 'Harness', languages: ['python', 'typescript'] },
  evals: { label: 'Evals', languages: ['python'] },
}

/**
 * Site-wide language preference, shared via localStorage with Starlight's synced
 * code tabs. The key is the one Starlight derives by djb2-hashing the tab labels
 * "Python|TypeScript" (→ "jarkqt"); the default mirrors LanguageToggle's. Single
 * source so LanguageToggle.astro and the changelog page can't drift — if
 * Starlight's key format or the tab labels change, update only here.
 */
export const LANG_STORAGE_KEY = 'starlight-synced-tabs__jarkqt'
export const DEFAULT_LANG_LABEL = 'TypeScript'

/**
 * Canonical human label for a release stream, e.g. "Harness Python", "Evals".
 * Single source so the detail page, markdown endpoints, RSS, and llms.txt don't
 * each hand-roll it (they previously drifted between "Harness Python" and
 * "Harness (python)").
 */
export function streamLabel(sdk: Sdk, language?: Language): string {
  return [SDK_META[sdk].label, language ? LANGUAGE_META[language].label : ''].filter(Boolean).join(' ')
}

// Package-registry URL construction lives in the devtools changelog-release-pr
// action (the producer of `packageUrl` frontmatter) — not duplicated here.
