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

// Package-registry URL construction lives in the devtools changelog-release-pr
// action (the producer of `packageUrl` frontmatter) — not duplicated here.
