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

// PyPI / npm package identifiers per (sdk, language).
const PYPI = (name: string, v: string) => `https://pypi.org/project/${name}/${v}/`
const NPM = (name: string, v: string) => `https://www.npmjs.com/package/${name}/v/${v}`

export function getPackageUrl(sdk: Sdk, language: Language | undefined, version: string): string {
  if (sdk === 'evals') return PYPI('strands-agents-evals', version)
  if (language === 'typescript') return NPM('@strands-agents/sdk', version)
  return PYPI('strands-agents', version) // harness python (default)
}
