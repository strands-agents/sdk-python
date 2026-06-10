import type { APIRoute } from 'astro'
import rss from '@astrojs/rss'
import { getReleases } from '../../util/changelog'

export const GET: APIRoute = async (context) => {
  const releases = await getReleases()
  return rss({
    title: 'Strands Agents Changelog',
    description: 'Releases across the Strands Agents Harness and Evals SDKs.',
    site: context.site!,
    items: releases.map((r) => ({
      title: `${r.data.sdk === 'evals' ? 'Evals' : 'Harness'} ${r.data.language ? `(${r.data.language}) ` : ''}v${r.data.version}`,
      pubDate: r.data.date,
      description:
        r.data.highlights ||
        r.data.entries.slice(0, 5).map((e) => `• ${e.title}`).join('\n') ||
        `Release ${r.data.version}`,
      link: r.data.releaseUrl,
      categories: [...new Set(r.data.entries.flatMap((e) => e.areas))],
    })),
    customData: '<language>en-us</language>',
  })
}
