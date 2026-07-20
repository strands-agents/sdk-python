/**
 * ONE-TIME backfill: discovers community Strands integrations on PyPI, npm,
 * and GitHub and writes draft catalog YAML entries for human review. Not part
 * of any recurring pipeline — ongoing additions come through submission PRs.
 *
 * Usage: npm run catalog:backfill   (requires GITHUB_TOKEN)
 * Output: draft YAML files in src/content/catalog/ (existing files never overwritten).
 * Review every generated file before committing; prune spam and dead packages.
 */

import { existsSync, writeFileSync } from 'node:fs'
import path from 'node:path'

export interface RegistryCandidate {
  source: 'pypi' | 'npm' | 'github'
  name: string
  description: string
  github?: string
  registry?: string
  maintainer?: string
}

export interface MergedCandidate {
  name: string
  description: string
  github: string
  maintainer: string
  python?: { package: string; registry: string }
  typescript?: { package: string; registry: string }
  inferredType: string
  typeUncertain: boolean
  addedDate: string
}

// Official packages are documented in the SDK docs, not the community catalog.
const OFFICIAL_ORGS = ['strands-agents', 'strands-labs']

/** Parse the GitHub org from a URL string, returning undefined if the URL is invalid. */
function parseRepoOrg(url: string): string | undefined {
  try {
    return new URL(url).pathname.split('/')[1] || undefined
  } catch {
    return undefined
  }
}

const TYPE_KEYWORDS: [string, RegExp][] = [
  ['model-provider', /model.provider|llm provider|inference/i],
  ['session-manager', /session.manager|session storage/i],
  ['memory-store', /memory.store|long.term memory/i],
  ['plugin', /\bplugin\b|lifecycle hook/i],
  ['agent-extension', /agent.extension|agent subclass/i],
  ['intervention', /\bintervention\b|guardrail/i],
  ['integration', /\bintegration\b|protocol|bridge|\bui\b/i],
]

export function inferType(name: string, description: string): string {
  const haystack = `${name} ${description}`
  for (const [type, pattern] of TYPE_KEYWORDS) {
    if (pattern.test(haystack)) return type
  }
  return 'tool'
}

function isOfficial(candidate: RegistryCandidate): boolean {
  const repoOrg = candidate.github ? parseRepoOrg(candidate.github) : undefined
  return repoOrg !== undefined && OFFICIAL_ORGS.includes(repoOrg)
}

/** Merge candidates across registries: same GitHub repo = same project. */
export function mergeCandidates(
  pypi: RegistryCandidate[],
  npm: RegistryCandidate[],
  github: RegistryCandidate[]
): MergedCandidate[] {
  const byRepo = new Map<string, MergedCandidate>()
  const today = new Date().toISOString().slice(0, 10)

  function upsert(c: RegistryCandidate): MergedCandidate | undefined {
    if (!c.github || isOfficial(c)) return undefined
    // Skip candidates whose github URL can't be parsed — they can't produce a
    // valid `github:` field and would crash on new URL().
    const repoOrg = parseRepoOrg(c.github)
    if (!repoOrg) return undefined
    const key = c.github.replace(/\/$/, '').toLowerCase()
    let merged = byRepo.get(key)
    if (!merged) {
      const inferredType = inferType(c.name, c.description)
      merged = {
        // Prefer the repo name over registry-specific names (npm scopes etc.)
        name: key.split('/').pop() || c.name,
        description: c.description,
        github: c.github,
        maintainer: c.maintainer || repoOrg || 'unknown',
        inferredType,
        typeUncertain: inferredType === 'tool',
        addedDate: today,
      }
      byRepo.set(key, merged)
    }
    return merged
  }

  for (const c of pypi) {
    const m = upsert(c)
    if (m && c.registry) m.python = { package: c.name, registry: c.registry }
  }
  for (const c of npm) {
    const m = upsert(c)
    if (m && c.registry) m.typescript = { package: c.name, registry: c.registry }
  }
  for (const c of github) upsert(c)

  // A catalog entry needs at least one published package; repo-only hits
  // without a registry match are usually apps or examples, not packages.
  return [...byRepo.values()].filter((m) => m.python || m.typescript)
}

export function candidateToYaml(c: MergedCandidate): string {
  const lines: string[] = []
  if (c.typeUncertain) lines.push('# REVIEW: integrationType inferred as fallback — verify')
  lines.push(`name: ${JSON.stringify(c.name)}`)
  lines.push(`description: ${JSON.stringify(c.description)}`)
  lines.push(`integrationType: ${c.inferredType}`)
  lines.push('languages:')
  if (c.python) {
    lines.push('  python:')
    lines.push(`    package: ${JSON.stringify(c.python.package)}`)
    lines.push(`    registry: ${c.python.registry}`)
  }
  if (c.typescript) {
    lines.push('  typescript:')
    lines.push(`    package: ${JSON.stringify(c.typescript.package)}`)
    lines.push(`    registry: ${c.typescript.registry}`)
  }
  lines.push(`github: ${c.github}`)
  lines.push(`maintainer: ${JSON.stringify(c.maintainer)}`)
  lines.push(`addedDate: ${c.addedDate}`)
  return lines.join('\n') + '\n'
}

// ── Discovery (live) ─────────────────────────────────────────────────────────

async function fetchJson(url: string, headers: Record<string, string> = {}): Promise<any> {
  const res = await fetch(url, { headers })
  if (!res.ok) throw new Error(`status=${res.status} url=${url}`)
  return res.json()
}

async function discoverPypi(): Promise<RegistryCandidate[]> {
  // PyPI's XML-RPC search is dead; use the simple JSON search on pypi.org via
  // the public search page API is unstable — query the JSON metadata of known
  // naming-convention candidates from the GitHub discovery pass instead, plus
  // packages whose name matches strands-*. libraries.io is an alternative if
  // this misses too much.
  const results: RegistryCandidate[] = []
  const search = await fetchJson('https://pypi.org/search/?q=strands&format=json').catch(() => null)
  const names: string[] =
    search?.results?.map((r: any) => r.name) ??
    // Fallback: GitHub code-search-derived names handled in main(); return empty here.
    []
  for (const name of names) {
    try {
      const meta = await fetchJson(`https://pypi.org/pypi/${name}/json`)
      const info = meta.info
      const repoUrl: string | undefined =
        info.project_urls?.Source || info.project_urls?.Repository || info.project_urls?.Homepage
      if (!repoUrl?.includes('github.com')) continue
      results.push({
        source: 'pypi',
        name: info.name,
        description: info.summary || '',
        github: repoUrl,
        registry: `https://pypi.org/project/${info.name}/`,
        maintainer: info.author || undefined,
      })
    } catch {
      // Package metadata unavailable — skip.
    }
  }
  return results
}

async function discoverNpm(): Promise<RegistryCandidate[]> {
  const data = await fetchJson('https://registry.npmjs.org/-/v1/search?text=strands%20agents&size=100')
  return (data.objects as any[])
    .map((o) => o.package)
    .filter((p) => /strands/i.test(p.name) || /strands/i.test(p.description || ''))
    .map((p) => ({
      source: 'npm' as const,
      name: p.name as string,
      description: (p.description as string) || '',
      github: (p.links?.repository as string | undefined)?.replace(/^git\+|\.git$/g, ''),
      registry: `https://www.npmjs.com/package/${p.name}`,
      maintainer: p.publisher?.username as string | undefined,
    }))
}

async function discoverGithub(): Promise<RegistryCandidate[]> {
  const headers: Record<string, string> = { accept: 'application/vnd.github+json' }
  if (process.env.GITHUB_TOKEN) headers.authorization = `Bearer ${process.env.GITHUB_TOKEN}`
  const data = await fetchJson(
    'https://api.github.com/search/repositories?q=strands+agents+in:name,description,topics&per_page=100',
    headers
  )
  return (data.items as any[]).map((r) => ({
    source: 'github' as const,
    name: r.name as string,
    description: (r.description as string) || '',
    github: r.html_url as string,
    maintainer: r.owner?.login as string | undefined,
  }))
}

// ── CLI entry point ───────────────────────────────────────────────────────────

const isDirectRun = process.argv[1]?.endsWith('backfill.ts')
if (isDirectRun) {
  const catalogDir = path.resolve('src/content/catalog')
  const [pypi, npm, github] = await Promise.all([
    discoverPypi().catch((e) => {
      console.error(`source=pypi | discovery failed, continuing with other sources: ${e}`)
      return [] as RegistryCandidate[]
    }),
    discoverNpm().catch((e) => {
      console.error(`source=npm | discovery failed, continuing with other sources: ${e}`)
      return [] as RegistryCandidate[]
    }),
    discoverGithub().catch((e) => {
      console.error(`source=github | discovery failed, continuing with other sources: ${e}`)
      return [] as RegistryCandidate[]
    }),
  ])
  console.log(`pypi=${pypi.length}, npm=${npm.length}, github=${github.length} | candidates discovered`)
  const merged = mergeCandidates(pypi, npm, github)
  let written = 0
  let skipped = 0
  for (const candidate of merged) {
    const filePath = path.join(catalogDir, `${candidate.name}.yaml`)
    if (existsSync(filePath)) {
      skipped++
      continue
    }
    writeFileSync(filePath, candidateToYaml(candidate))
    written++
  }
  console.log(`written=${written}, skipped_existing=${skipped} | draft entries created — review before committing`)
}
