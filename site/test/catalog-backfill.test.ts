import { describe, it, expect } from 'vitest'
import { mergeCandidates, candidateToYaml, inferType, type RegistryCandidate } from '../scripts/catalog/backfill'

describe('mergeCandidates', () => {
  it('merges pypi and npm packages that share a github repo into one candidate', () => {
    const pypi: RegistryCandidate[] = [
      { source: 'pypi', name: 'strands-widget', description: 'Widget tool for Strands', github: 'https://github.com/ex/strands-widget' },
    ]
    const npm: RegistryCandidate[] = [
      { source: 'npm', name: '@ex/strands-widget', description: 'Widget tool for Strands', github: 'https://github.com/ex/strands-widget' },
    ]
    const merged = mergeCandidates(pypi, npm, [])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.python?.package).toBe('strands-widget')
    expect(merged[0]!.typescript?.package).toBe('@ex/strands-widget')
  })

  it('keeps packages with different repos separate and excludes official org packages', () => {
    const pypi: RegistryCandidate[] = [
      { source: 'pypi', name: 'strands-a', description: 'a', github: 'https://github.com/ex/strands-a' },
      { source: 'pypi', name: 'strands-agents-tools', description: 'official', github: 'https://github.com/strands-agents/tools' },
    ]
    const merged = mergeCandidates(pypi, [], [])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.name).toBe('strands-a')
  })
})

describe('inferType', () => {
  it('classifies from keywords with tool as the fallback', () => {
    expect(inferType('strands-cohere', 'Cohere model provider for Strands')).toBe('model-provider')
    expect(inferType('strands-redis-session', 'Redis session manager')).toBe('session-manager')
    expect(inferType('strands-something', 'Does something')).toBe('tool')
  })
})

describe('candidateToYaml', () => {
  it('emits a valid entry with a REVIEW marker for uncertain types', () => {
    const yaml = candidateToYaml({
      name: 'strands-something',
      description: 'Does something',
      github: 'https://github.com/ex/strands-something',
      python: { package: 'strands-something' },
      inferredType: 'tool',
      typeUncertain: true,
      addedDate: '2026-07-17',
    })
    expect(yaml).toContain('# REVIEW: integrationType inferred as fallback — verify')
    expect(yaml).toContain('name: "strands-something"')
    expect(yaml).toContain('integrationType: tool')
    expect(yaml).toContain('package: "strands-something"')
  })

  it('emits no derived fields (maintainer, registry)', () => {
    const yaml = candidateToYaml({
      name: 'strands-pkg',
      description: 'A package',
      github: 'https://github.com/ex/strands-pkg',
      python: { package: 'strands-pkg' },
      inferredType: 'integration',
      typeUncertain: false,
      addedDate: '2026-07-18',
    })
    expect(yaml).not.toContain('maintainer:')
    expect(yaml).not.toContain('registry:')
  })
})

describe('mergeCandidates – malformed github URLs', () => {
  it('excludes candidates with malformed github URLs without throwing', () => {
    // Prefixed URL ("See https://...")
    const withPrefix: RegistryCandidate[] = [
      { source: 'pypi', name: 'bad-url-pkg', description: 'Has bad url', github: 'See https://github.com/x/bad-url-pkg' },
    ]
    expect(mergeCandidates(withPrefix, [], [])).toHaveLength(0)

    // Scheme-less URL
    const noScheme: RegistryCandidate[] = [
      { source: 'pypi', name: 'no-scheme-pkg', description: 'Has no scheme', github: 'github.com/x/no-scheme-pkg' },
    ]
    expect(mergeCandidates(noScheme, [], [])).toHaveLength(0)
  })

  it('excludes candidates whose repository is not on github.com', () => {
    // The catalog schema requires a https://github.com/ repo URL; registry
    // metadata pointing elsewhere would generate an entry the build rejects.
    const gitlab: RegistryCandidate[] = [
      { source: 'pypi', name: 'gitlab-pkg', description: 'Hosted on GitLab', github: 'https://gitlab.com/x/gitlab-pkg' },
    ]
    expect(mergeCandidates(gitlab, [], [])).toHaveLength(0)
  })
})
