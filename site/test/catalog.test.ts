import { describe, it, expect } from 'vitest'
import { toCardModel, sortEntries, NEW_BADGE_DAYS } from '../src/util/catalog'
import type { CatalogEntryData } from '../src/content.config'

const BUILD_DATE = new Date('2026-07-17')

function entry(overrides: Partial<CatalogEntryData> = {}): CatalogEntryData {
  return {
    name: 'strands-example',
    description: 'Example integration',
    integrationType: 'tool',
    sdk: 'agents',
    languages: { python: { package: 'strands-example', registry: 'https://pypi.org/project/strands-example/' } },
    github: 'https://github.com/example/strands-example',
    maintainer: 'example',
    featured: false,
    badges: [],
    addedDate: new Date('2026-01-01'),
    ...overrides,
  }
}

describe('toCardModel', () => {
  it('links to the github repo when no docs link is set', () => {
    const card = toCardModel('strands-example', entry(), undefined, BUILD_DATE)
    expect(card.href).toBe('https://github.com/example/strands-example')
    expect(card.external).toBe(true)
  })

  it('prefers docsUrl over the github repo', () => {
    const card = toCardModel(
      'strands-example',
      entry({ docsUrl: 'https://example.com/docs/strands' }),
      undefined,
      BUILD_DATE
    )
    expect(card.href).toBe('https://example.com/docs/strands')
    expect(card.external).toBe(true)
  })

  it('prefers an on-site docsPage over both docsUrl and github', () => {
    const card = toCardModel(
      'strands-example',
      entry({ docsUrl: 'https://example.com/docs/strands', docsPage: 'docs/community/tools/strands-example' }),
      undefined,
      BUILD_DATE
    )
    expect(card.href).toBe('/docs/community/tools/strands-example/')
    expect(card.external).toBe(false)
  })

  it('derives language list and registry links', () => {
    const card = toCardModel(
      'strands-example',
      entry({
        languages: {
          python: { package: 'strands-example', registry: 'https://pypi.org/project/strands-example/' },
          typescript: { package: '@example/strands', registry: 'https://www.npmjs.com/package/@example/strands' },
        },
      }),
      undefined,
      BUILD_DATE
    )
    expect(card.languages).toEqual(['python', 'typescript'])
    expect(card.registryLinks).toEqual([
      { label: 'PyPI', href: 'https://pypi.org/project/strands-example/' },
      { label: 'npm', href: 'https://www.npmjs.com/package/@example/strands' },
    ])
  })

  it('adds the new badge when addedDate is within the window', () => {
    const recent = new Date(BUILD_DATE.getTime() - (NEW_BADGE_DAYS - 1) * 86_400_000)
    const card = toCardModel(
      'strands-example',
      entry({ addedDate: recent, badges: ['verified'] }),
      undefined,
      BUILD_DATE
    )
    expect(card.badges).toEqual(['verified', 'new'])
  })

  it('omits the new badge when addedDate is outside the window', () => {
    const card = toCardModel('strands-example', entry(), undefined, BUILD_DATE)
    expect(card.badges).toEqual([])
  })

  it('joins stats and sums downloads, tolerating missing stats', () => {
    const withStats = toCardModel(
      'strands-example',
      entry(),
      { stars: 42, downloads: { python: 100, typescript: 50 } },
      BUILD_DATE
    )
    expect(withStats.stars).toBe(42)
    expect(withStats.downloads).toBe(150)

    const withoutStats = toCardModel('strands-example', entry(), undefined, BUILD_DATE)
    expect(withoutStats.stars).toBeUndefined()
    expect(withoutStats.downloads).toBeUndefined()
  })
})

describe('sortEntries', () => {
  it('puts featured entries first, then sorts by name', () => {
    const cards = [
      toCardModel('zeta', entry({ name: 'zeta' }), undefined, BUILD_DATE),
      toCardModel('alpha', entry({ name: 'alpha' }), undefined, BUILD_DATE),
      toCardModel('feat', entry({ name: 'feat', featured: true }), undefined, BUILD_DATE),
    ]
    expect(sortEntries(cards).map((c) => c.name)).toEqual(['feat', 'alpha', 'zeta'])
  })
})
