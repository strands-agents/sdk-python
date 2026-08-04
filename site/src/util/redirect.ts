/**
 * Redirect helper: maps old MkDocs-style URLs to new CMS slugs or explicit external URLs.
 *
 * resolveRedirectFromUrl normalises a versioned URL to a slug, then
 * resolveRedirect applies rename rules for paths that changed structure.
 *
 * Only explicit rules in SLUG_RULES or redirectFromMap may produce external URLs.
 * The path-normalization fallback never returns an external URL (prevents open redirects).
 */

import { exactly, startsWith } from '../utils/regex'

// ── Slug-level rename rules ───────────────────────────────────────────────────

/**
 * Exact-match slug renames: old slug → new slug or explicit external URL.
 *
 * Exported so astro.config.mjs (via redirect.static.ts) can enumerate every
 * entry into a build-time redirect stub — a static HTML page with a meta
 * refresh and canonical link. Crawlers don't execute the client-side 404
 * redirect, so these stubs are what preserve backlink equity for renamed pages.
 *
 * Only add enumerable, exact-match renames here. Dynamic rules (regex matches,
 * computed targets) belong in SLUG_RULES below and are handled solely by the
 * client-side 404 fallback.
 */
export const STATIC_SLUG_REDIRECTS: Record<string, string> = {
  // gemini was renamed to google
  'docs/user-guide/concepts/model-providers/gemini': 'docs/user-guide/concepts/model-providers/google',

  // python-tools was renamed to custom-tools
  'docs/user-guide/concepts/tools/python-tools': 'docs/user-guide/concepts/tools/custom-tools',

  // multi_agent_example index redirects to the main example page
  'docs/examples/python/multi_agent_example': 'docs/examples/python/multi_agent_example/multi_agent_example',

  // Vanity URLs for community links
  discord: 'https://discord.gg/strands',

  // cli-reference-agent was archived (strands-agents/agent-builder)
  'docs/examples/python/cli-reference-agent': 'docs/examples',

  // community-packages content lives on the interactive integrations page
  // (an Astro page — buildStaticRedirects validates those targets against
  // src/pages as well as docs content).
  'docs/community/community-packages': 'integrations',

  // The community docs section overview URL predates the docs/integrations
  // rename; its landing content is the interactive integrations page.
  'docs/community': 'integrations',

  // The docs/community/ section was renamed to docs/integrations/ so its URLs
  // match the /integrations page. Every page that existed at rename time gets
  // a static stub (crawler-followable); the COMMUNITY_PREFIX_RULE below covers
  // any docs/community/ URL not listed here via the client-side 404 fallback.
  'docs/community/agent-extensions/strands-code-agent': 'docs/integrations/agent-extensions/strands-code-agent',
  'docs/community/get-featured': 'docs/integrations/get-featured',
  'docs/community/integrations/ag-ui': 'docs/integrations/integrations/ag-ui',
  'docs/community/interventions/overview': 'docs/integrations/interventions/overview',
  'docs/community/interventions/strands-agt': 'docs/integrations/interventions/strands-agt',
  'docs/community/memory-stores/agentcore-memory-store': 'docs/integrations/memory-stores/agentcore-memory-store',
  'docs/community/memory-stores/overview': 'docs/integrations/memory-stores/overview',
  'docs/community/memory-stores/strands-dakera': 'docs/integrations/memory-stores/strands-dakera',
  'docs/community/model-providers/clova-studio': 'docs/integrations/model-providers/clova-studio',
  'docs/community/model-providers/cohere': 'docs/integrations/model-providers/cohere',
  'docs/community/model-providers/crusoe': 'docs/integrations/model-providers/crusoe',
  'docs/community/model-providers/fireworksai': 'docs/integrations/model-providers/fireworksai',
  'docs/community/model-providers/mlx': 'docs/integrations/model-providers/mlx',
  'docs/community/model-providers/nebius-token-factory': 'docs/integrations/model-providers/nebius-token-factory',
  'docs/community/model-providers/nvidia-nim': 'docs/integrations/model-providers/nvidia-nim',
  'docs/community/model-providers/ovhcloud-ai-endpoints': 'docs/integrations/model-providers/ovhcloud-ai-endpoints',
  'docs/community/model-providers/sglang': 'docs/integrations/model-providers/sglang',
  'docs/community/model-providers/vllm': 'docs/integrations/model-providers/vllm',
  'docs/community/model-providers/xai': 'docs/integrations/model-providers/xai',
  'docs/community/plugins/agent-control': 'docs/integrations/plugins/agent-control',
  'docs/community/plugins/agentcore-payments': 'docs/integrations/plugins/agentcore-payments',
  'docs/community/plugins/agentcore-tool-search': 'docs/integrations/plugins/agentcore-tool-search',
  'docs/community/plugins/datadog-ai-guard': 'docs/integrations/plugins/datadog-ai-guard',
  'docs/community/plugins/s3-vectors-memory': 'docs/integrations/plugins/s3-vectors-memory',
  'docs/community/session-managers/agentcore-memory': 'docs/integrations/session-managers/agentcore-memory',
  'docs/community/session-managers/strands-valkey-session-manager':
    'docs/integrations/session-managers/strands-valkey-session-manager',
  'docs/community/storage/overview': 'docs/integrations/storage/overview',
  'docs/community/tools/strands-apify': 'docs/integrations/tools/strands-apify',
  'docs/community/tools/strands-deepgram': 'docs/integrations/tools/strands-deepgram',
  'docs/community/tools/strands-google': 'docs/integrations/tools/strands-google',
  'docs/community/tools/strands-hubspot': 'docs/integrations/tools/strands-hubspot',
  'docs/community/tools/strands-perplexity': 'docs/integrations/tools/strands-perplexity',
  'docs/community/tools/strands-spraay': 'docs/integrations/tools/strands-spraay',
  'docs/community/tools/strands-sql': 'docs/integrations/tools/strands-sql',
  'docs/community/tools/strands-teams': 'docs/integrations/tools/strands-teams',
  'docs/community/tools/strands-telegram': 'docs/integrations/tools/strands-telegram',
  'docs/community/tools/strands-telegram-listener': 'docs/integrations/tools/strands-telegram-listener',
  'docs/community/tools/utcp': 'docs/integrations/tools/utcp',

  // CDK and deployment examples now live on GitHub
  'docs/examples/cdk/deploy_to_apprunner':
    'https://github.com/strands-agents/harness-sdk/blob/main/site/docs/examples/cdk/deploy_to_apprunner/README.md',
  'docs/examples/cdk/deploy_to_ec2':
    'https://github.com/strands-agents/harness-sdk/blob/main/site/docs/examples/cdk/deploy_to_ec2/README.md',
  'docs/examples/cdk/deploy_to_fargate':
    'https://github.com/strands-agents/harness-sdk/blob/main/site/docs/examples/cdk/deploy_to_fargate/README.md',
  'docs/examples/cdk/deploy_to_lambda':
    'https://github.com/strands-agents/harness-sdk/blob/main/site/docs/examples/cdk/deploy_to_lambda/README.md',
  'docs/examples/deploy_to_eks':
    'https://github.com/strands-agents/harness-sdk/blob/main/site/docs/examples/deploy_to_eks/README.md',
  'docs/examples/typescript/deploy_to_bedrock_agentcore':
    'https://github.com/strands-agents/harness-sdk/blob/main/site/docs/examples/typescript/deploy_to_bedrock_agentcore/README.md',
}

type SlugRule =
  | { match: RegExp; to: string }
  | { match: RegExp; to: (m: RegExpMatchArray) => string }

// Exact-match rules generated from STATIC_SLUG_REDIRECTS, plus any dynamic
// (regex-based) rules. Dynamic rules can't be enumerated into static stubs,
// so they are only applied by the client-side 404 fallback.
const SLUG_RULES: SlugRule[] = [
  ...Object.entries(STATIC_SLUG_REDIRECTS).map(([from, to]) => ({
    match: exactly(from),
    to,
  })),

  // Catch-all for the docs/community/ → docs/integrations/ section rename.
  // Exact static entries above win for pages that existed at rename time;
  // this covers any other docs/community/ URL (e.g. a page added on a branch
  // that predates the rename) via the client-side 404 fallback.
  {
    match: startsWith('docs/community'),
    to: (m) => `docs/integrations/${m[1]}`,
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
