# Community Integration Catalog — Design

**Date:** 2026-07-17
**Status:** Approved
**Scope:** `site/` (strandsagents.com), plus a one-time backfill script and a scheduled stats workflow.

## Problem

The community page (`/docs/community/community-packages`) is a set of static tables generated from doc-page frontmatter. Customers cannot search, filter, or compare community integrations; there is no way to feature entries, no trust signaling, and every catalog entry requires a full doc page. Dual-language integrations (Python + TypeScript) have no structured representation of their per-language packages. The upcoming evals SDK will grow its own plugin/community ecosystem and must fit the same catalog without restructuring.

## Decisions (settled during brainstorming)

- **One catalog entry per integration**, with per-language package metadata (not one entry per language). Dual-language detail pages use the existing synced `<Tabs>` pattern driven by the global language toggle.
- **Single-language pages disable the global language toggle** (rendered disabled with the available language selected) and show the existing `LanguageSupportAside` banner.
- **Badges:** trust/status (curated), language (derived), category (derived), freshness/popularity (from scheduled stats refresh).
- **Stats via scheduled GitHub Actions job** that commits a JSON file — not build-time or client-side fetching.
- **Featured = curated flag**, rendered as a hero row above the grid.
- **Evals SDK:** full design now (an `sdk` field and an SDK facet), but the facet stays hidden until at least one evals entry exists.
- **Backfill is one-time**; ongoing submissions come through a documented PR flow with a PR template, reviewed by the team.
- **Cards can link out**: an entry minimally needs metadata; a doc page is optional. Card links to the doc page when present, otherwise to GitHub.
- **Catalog lives at a dedicated top-level page** (`/catalog`), outside the docs layout.
- **Source of truth is one YAML file per entry** in an Astro content collection — native Astro mechanisms, existing repo conventions, no new framework dependencies, no over-engineering.

## 1. Data model — `catalog` content collection

New collection in `site/src/content.config.ts` using Astro's built-in `glob` loader over `site/src/content/catalog/**/*.yaml`, one file per integration, zod-validated:

```yaml
# src/content/catalog/strands-deepgram.yaml
name: strands-deepgram
description: Deepgram speech-to-text, text-to-speech, and audio intelligence
integrationType: tool          # reuses the existing enum
sdk: agents                    # 'agents' | 'evals'; defaults to 'agents'
languages:
  python:
    package: strands-deepgram
    registry: https://pypi.org/project/strands-deepgram/
  # typescript block absent = Python-only
github: https://github.com/eraykeskinmac/strands-deepgram
maintainer: eraykeskinmac
docsPage: community/tools/strands-deepgram   # optional
featured: false                # maintainer-granted
badges: [verified]             # maintainer-granted trust badges only
addedDate: 2026-07-17          # drives the "New" badge
```

- `integrationType` reuses the existing enum from `integration-content.ts`: `tool`, `model-provider`, `session-manager`, `memory-store`, `integration`, `plugin`, `agent-extension`, `intervention`. The enum is extended when evals entry types are known.
- **Derived vs declared badges:** language and category badges derive from entry data. Only trust badges (`verified`) and `featured` are editorial, and only maintainers set them.
- **Stats are separate:** `site/src/data/catalog-stats.json`, keyed by entry filename, written only by the scheduled job. Shape per entry: GitHub stars, last-release date, and per-language download counts. The catalog page joins entries with stats at build time; a missing record means no popularity badges for that card.
- Existing doc pages keep their frontmatter (it drives in-docs banners), but the catalog reads only the YAML collection. `integration-content.ts` continues to serve official-integration tables elsewhere in the docs.
- Zod validation makes a malformed submission fail `npm run build` — a mechanical review gate.

## 2. Catalog page — `/catalog`

`site/src/pages/catalog.astro`, a standalone top-level page (same pattern as `index.astro`), using existing site layout and styles.

**Build time:** `getCollection('catalog')` + stats join; all cards rendered into static HTML (SEO- and llms.txt-friendly).

- **Featured hero row**: `featured: true` entries, larger card treatment, above the grid.
- **Filter bar**: text search input; filter chips for integration type, language, trust badges. An SDK facet (Agents/Evals) is implemented but rendered only when the collection contains ≥1 `sdk: evals` entry.
- **Card grid**: name, description, category chip, language badges, trust badges, stars/downloads when stats exist, "New" badge when `addedDate` is within ~30 days of build. Primary link: `docsPage` if set, else `github`. Registry links (PyPI/npm) as secondary icons.
- **Submission call-to-action** at the bottom linking to the submission guide.

**Client side:** one small inline vanilla `<script>` (the existing site pattern, e.g. `AutoSyncTabs.astro`). Cards carry `data-type`, `data-langs`, `data-badges`, `data-sdk`, `data-search`; the script toggles `hidden` as filters change. Substring match on name + description — no search library. Filter state syncs to the URL query string for shareable views. With JS disabled the full catalog is visible unfiltered.

**Components:** `CatalogCard.astro`, `CatalogFilterBar.astro` in the existing flat `site/src/components/`.

**Navigation:** header nav gains a Catalog link. `community-packages.mdx` becomes a redirect to `/catalog` via the site's existing redirect mechanism. The build's broken-links checker guards the migration.

## 3. Detail pages and the language toggle

Integration doc pages stay under `site/src/content/docs/community/**` with existing frontmatter. Dual-language pages author install/setup/usage in synced `<Tabs>`, following the global language toggle as SDK docs already do.

For pages whose `languages` frontmatter declares a single language, the global `LanguageToggle` renders **disabled with the available language selected**, and the existing `LanguageSupportAside` banner states the limitation. Both components exist; this is a conditional-state change, not a new mechanism. Catalog cards always show language badges, so users know support before clicking through.

## 4. Submission flow

- `get-featured.mdx` is rewritten as **"Add your integration to the catalog"**: add one YAML file to `src/content/catalog/` via PR; optionally add a doc page under `community/` for an on-site page.
- **PR template** at `.github/PULL_REQUEST_TEMPLATE/catalog-submission.md`: package published, repo public, accurate description, correct `integrationType`, license present.
- Submitters leave `featured` and `badges` unset; the team grants them. Schema validation catches structural errors in CI.

## 5. Scheduled stats refresh

GitHub Actions workflow `catalog-stats.yml` (weekly cron + `workflow_dispatch`) runs a script in `site/scripts/` that:

1. Reads all catalog YAML entries.
2. Queries the GitHub API (stars, last release) and npm + PyPI download-stats APIs per entry.
3. Writes `site/src/data/catalog-stats.json`.
4. Opens/updates an automated PR when the file changed (matching the repo's existing automated-sync PR pattern).

Per-package failures are logged and skipped — the entry keeps its previous stats or has none. The workflow fails only if it cannot produce valid JSON.

## 6. One-time backfill

A one-time script (in `site/scripts/`, documented as one-time) that:

1. **Discovers candidates**: PyPI (`strands` in name/keywords/description), npm (`strands` search, `@strands-agents/sdk` dependents), GitHub (`strands-*` repos, dependents of the SDKs).
2. **Excludes** official `strands-agents` org packages and the SDKs themselves.
3. **Generates one draft YAML entry per candidate**: name, description from registry metadata, inferred `integrationType` (keyword/README heuristic; defaults to `tool` with a `# REVIEW` comment when uncertain), language blocks from the registries it appears on — same-project PyPI/npm packages merge into one entry — github link, maintainer.
4. **Migrates existing entries**: generates YAML (with `docsPage:` set) for the ~30 integrations that already have doc pages under `community/`.
5. Lands as **one PR for human review** — the editorial pass removes spam, abandoned, or miscategorized packages before merge.

## 7. Testing and error handling

- **Schema:** zod validation is the primary gate (build fails on bad YAML). Vitest unit tests for entry processing: language derivation, badge derivation, stats join with missing records, featured ordering.
- **Filter logic:** the matching function is extracted and unit-tested, consistent with existing util tests.
- **Stats job:** per-package failures non-fatal; stale stats degrade silently.
- **Links:** the existing broken-links checker validates redirects and card links to doc pages.

## Out of scope

- Evals SDK catalog UI beyond the hidden facet (ships when evals entries exist).
- Fuzzy search or a client framework island — revisit only if the catalog outgrows substring filtering (hundreds of entries).
- Automated ongoing package discovery (the backfill is one-time by decision).
