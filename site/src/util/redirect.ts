/**
 * Redirect helper: maps old MkDocs-style URLs to new CMS slugs or explicit external URLs.
 *
 * resolveRedirectFromUrl normalises a versioned URL to a slug, then
 * resolveRedirect applies rename rules for paths that changed structure.
 *
 * Only explicit rules in SLUG_RULES or redirectFromMap may produce external URLs.
 * The path-normalization fallback never returns an external URL (prevents open redirects).
 */

import { exactly } from '../utils/regex'

// ── Slug-level rename rules ───────────────────────────────────────────────────

type SlugRule =
  | { match: RegExp; to: string }
  | { match: RegExp; to: (m: RegExpMatchArray) => string }

const SLUG_RULES: SlugRule[] = [
  // gemini was renamed to google
  {
    match: exactly('docs/user-guide/concepts/model-providers/gemini'),
    to: 'docs/user-guide/concepts/model-providers/google',
  },

  // python-tools was renamed to custom-tools
  {
    match: exactly('docs/user-guide/concepts/tools/python-tools'),
    to: 'docs/user-guide/concepts/tools/custom-tools',
  },

  // multi_agent_example index redirects to the main example page
  {
    match: exactly('docs/examples/python/multi_agent_example'),
    to: 'docs/examples/python/multi_agent_example/multi_agent_example',
  },

  // Vanity URLs for community links
  {
    match: exactly('discord'),
    to: 'https://discord.gg/strands',
  },

  // CDK and deployment examples now live on GitHub
  {
    match: exactly('docs/examples/cdk/deploy_to_apprunner'),
    to: 'https://github.com/strands-agents/docs/blob/main/docs/examples/cdk/deploy_to_apprunner/README.md',
  },
  {
    match: exactly('docs/examples/cdk/deploy_to_ec2'),
    to: 'https://github.com/strands-agents/docs/blob/main/docs/examples/cdk/deploy_to_ec2/README.md',
  },
  {
    match: exactly('docs/examples/cdk/deploy_to_fargate'),
    to: 'https://github.com/strands-agents/docs/blob/main/docs/examples/cdk/deploy_to_fargate/README.md',
  },
  {
    match: exactly('docs/examples/cdk/deploy_to_lambda'),
    to: 'https://github.com/strands-agents/docs/blob/main/docs/examples/cdk/deploy_to_lambda/README.md',
  },
  {
    match: exactly('docs/examples/deploy_to_eks'),
    to: 'https://github.com/strands-agents/docs/blob/main/docs/examples/deploy_to_eks/README.md',
  },
  {
    match: exactly('docs/examples/typescript/deploy_to_bedrock_agentcore'),
    to: 'https://github.com/strands-agents/docs/blob/main/docs/examples/typescript/deploy_to_bedrock_agentcore/README.md',
  },
]

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Apply slug-level rename rules to an already-normalised slug.
 * Checks SLUG_RULES first (highest priority), then falls back to redirectFromMap.
 * Returns the renamed slug, or null if no rule matched (slug is already current).
 *
 * @param slug - The slug to resolve
 * @param redirectFromMap - Optional map of source slugs to target slugs (from frontmatter redirectFrom)
 */
export function resolveRedirect(slug: string, redirectFromMap?: Record<string, string>): string | null {
  // Check SLUG_RULES first (highest priority)
  for (const rule of SLUG_RULES) {
    const m = slug.match(rule.match)
    if (m) return typeof rule.to === 'function' ? rule.to(m) : rule.to
  }

  // Then check redirectFromMap (frontmatter-based redirects)
  if (redirectFromMap && slug in redirectFromMap) {
    return redirectFromMap[slug] ?? null
  }

  return null
}

/**
 * Given a path from the old site, normalise it to a slug and apply redirect rules.
 * Returns '/' for versioned root paths, or null if the path isn't recognisable.
 *
 * External URLs (https://) are only returned when an explicit rule in SLUG_RULES
 * or redirectFromMap matched. The path-normalization fallback never produces an
 * external URL, preventing open redirect attacks via crafted paths like
 * /latest/https://evil.com.
 *
 * @param path - The URL path to resolve (e.g. "/docs/user-guide/...")
 * @param redirectFromMap - Optional map of source slugs to target slugs (from frontmatter redirectFrom)
 */
export function resolveRedirectFromUrl(
  path: string,
  redirectFromMap?: Record<string, string>
): string | null {
  // Strip leading version segment: /latest/, /1.x/, /1.5.x/, etc.
  path = path.replace(/^\/?(latest|[\d]+(?:\.[\dx]+)*)\//, '/')

  // /documentation/docs/... -> /docs/...
  path = path.replace(/^\/?documentation\/docs(\/|$)/, 'docs$1')

  // Remember if the original path had a trailing slash
  const hadTrailingSlash = path.endsWith('/')

  // Trim leading/trailing slashes
  path = path.replace(/^\/+|\/+$/g, '')

  if (path === '') return '/'

  // Prevent open redirects: absolute URLs can only come from explicit SLUG_RULES,
  // never from the path-normalization fallback.
  if (/^https?:\/\//i.test(path)) return null

  const resolved = resolveRedirect(path, redirectFromMap) ?? path
  return hadTrailingSlash ? `${resolved}/` : resolved
}
