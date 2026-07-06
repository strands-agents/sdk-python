// Orchestrate one GitHub release into a rendered changelog file. Pure given
// injected deps (enrich + readExisting), so it's unit-testable without network.

import { tagToMeta, getPackageUrl } from './tag-meta'
import { parseNewContributors } from './parse-release-body'
import { renderMarkdown, mergePreserving } from './render-markdown'
import type { Release, Enrichment, ParsedEntry, RenderedEntry, ReleaseFile } from './types'

export interface BuildDeps {
  deriveEntries(repo: string, release: Release): Promise<{ entries: ParsedEntry[]; warning?: string }>
  enrich(prRepo: string, pr: number): Promise<Enrichment>
  readExisting(path: string): Promise<string | null>
  skipExisting?: boolean
}

function fileNameFor(sdk: string, language: string | undefined, version: string): string {
  if (sdk === 'evals') return `evals/v${version}.md`
  return `harness/${language}-v${version}.md`
}

export async function buildReleaseFile(
  repo: string,
  release: Release,
  deps: BuildDeps
): Promise<{ path: string; contents: string; warning?: string } | null> {
  const meta = tagToMeta(repo, release.tag_name)
  if (!meta) return null

  const path = `site/src/content/changelog/${fileNameFor(meta.sdk, meta.language, meta.version)}`
  const existing = await deps.readExisting(path)

  // skipExisting (used by the daily cron backstop): only generate files for
  // releases that don't have one yet. Checked BEFORE enrichment so a skipped
  // release costs zero PR API calls, and existing files (possibly carrying
  // richer enrichment from when labels were fresher) are never regressed by a
  // rate-limited re-run. A full refresh is an explicit backfill dispatch.
  if (deps.skipExisting && existing) return null

  // Entries come from the GitHub compare API (every merged PR between the prior
  // tag and this one) -- deterministic and independent of release-note format.
  // The release body is NOT parsed for entries; it's preserved as curated
  // narrative via mergePreserving below.
  const { entries: parsed, warning } = await deps.deriveEntries(repo, release)

  // Two gates apply to every entry:
  //
  // 1. Docs-only (ALL streams): a PR confined to docs/blog/website dirs never
  //    lines up with an SDK+language, so it's dropped everywhere -- including
  //    pre-monorepo bare-`v` and evals, which are otherwise unfiltered. This
  //    keeps the changelog focused on SDK+language work (a blog-only PR or a
  //    pure docs change won't appear in any stream).
  // 2. Language (monorepo prefixed tags only): those releases list every merged
  //    PR regardless of language, so gate by which SDK dirs the PR touched --
  //    python stream keeps python-touching PRs, ts keeps ts-touching, both ->
  //    both. Unknown file info -> kept (degrade open).
  //
  //    CRUCIAL: only gate when the PR has a POSITIVE dir signal -- i.e. it
  //    touches strands-py/ and/or strands-ts/. A PR with EMPTY languages
  //    (touches neither: root config/CI, or a pre-monorepo flat-layout PR whose
  //    code lived under src/ before the strands-py/ dir existed) must be KEPT,
  //    not dropped. Gating on empty languages would wrongly empty pre-monorepo
  //    releases whose tags were re-applied as python/v* in the monorepo.
  //    Pre-monorepo bare-`v` and evals are single-language: no language gate.
  const isMonorepoStream =
    meta.sdk === 'harness' && (release.tag_name.startsWith('python/') || release.tag_name.startsWith('typescript/'))

  // Shared keep/drop decision for both entries and new contributors: drop a
  // docs-only PR from every stream, and on a monorepo stream drop a PR with a
  // POSITIVE dir signal for the OTHER language (empty/unknown languages are
  // kept -- see the language-gate note above).
  const dropFromStream = (enr: Enrichment) =>
    enr.docsOnly ||
    (isMonorepoStream &&
      Array.isArray(enr.languages) &&
      enr.languages.length > 0 &&
      !enr.languages.includes(meta.language!))

  const entries: RenderedEntry[] = []
  for (const p of parsed) {
    const prRepo = p.prRepo || repo
    const enr = p.pr
      ? await deps.enrich(prRepo, p.pr)
      : { areas: [], breaking: false, commit: null, author: null, languages: null, docsOnly: false }
    if (dropFromStream(enr)) continue
    const breaking = p.breaking || enr.breaking
    entries.push({
      type: breaking && p.type === 'other' ? 'breaking' : p.type,
      breaking,
      scope: p.scope,
      areas: enr.areas,
      title: p.title,
      pr: p.pr,
      prUrl: p.pr ? `https://github.com/${prRepo}/pull/${p.pr}` : null,
      commit: enr.commit,
      commitUrl: enr.commit ? `https://github.com/${prRepo}/commit/${enr.commit}` : null,
      author: enr.author || p.author,
    })
  }

  // New contributors use the same keep/drop rule as entries (dropFromStream):
  // a docs-only first PR doesn't belong in an SDK+language changelog, and a
  // monorepo PR with a positive other-language signal is gated out -- but a
  // first PR touching no sdk dir (e.g. ci) or with unknown files is kept in
  // both streams: people aren't noise.
  const rawContributors = parseNewContributors(release.body)
  const newContributors = []
  for (const c of rawContributors) {
    // Use the PR's own repo (mirrors the entries path) -- first-contribution
    // links can point at the pre-monorepo repos.
    const enr = await deps.enrich(c.prRepo || repo, c.pr)
    if (dropFromStream(enr)) continue
    newContributors.push(c)
  }

  const file: ReleaseFile = {
    sdk: meta.sdk,
    language: meta.language,
    version: meta.version,
    tag: release.tag_name,
    date: release.published_at!.slice(0, 10), // safe: run.ts filters out releases with null published_at before calling here
    releaseUrl: release.html_url,
    packageUrl: getPackageUrl(meta.sdk, meta.language, meta.version),
    entries,
    newContributors,
  }

  const contents = existing ? mergePreserving(file, existing) : renderMarkdown(file)
  return warning ? { path, contents, warning } : { path, contents }
}
