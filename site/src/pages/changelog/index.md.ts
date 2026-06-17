import type { APIRoute } from 'astro'
import { getReleases, groupEntries, HIDDEN_AREAS, escapeMarkdownInline } from '../../util/changelog'

export const GET: APIRoute = async () => {
  const releases = await getReleases()
  const lines: string[] = ['# Strands Agents Changelog', '']
  for (const r of releases) {
    const d = r.data
    const label = d.sdk === 'evals' ? 'Evals' : `Harness${d.language ? ` (${d.language})` : ''}`
    lines.push(`## ${label} v${d.version} — ${d.date.toISOString().slice(0, 10)}`)
    lines.push(`Release: ${d.releaseUrl} · Package: ${d.packageUrl}`)
    if (d.highlights) lines.push('', d.highlights.trim())
    const { features, fixes, other } = groupEntries(d.entries)
    const section = (title: string, items: typeof d.entries) => {
      if (!items.length) return
      lines.push('', `### ${title}`)
      for (const e of items) {
        const tags = e.areas.filter((a) => !HIDDEN_AREAS.has(a))
        const areas = tags.length ? ` [${tags.join(', ')}]` : ''
        const pr = e.prUrl ? ` (${e.prUrl})` : ''
        lines.push(`- ${escapeMarkdownInline(e.title)}${areas}${pr}`)
      }
    }
    section('Features', features)
    section('Fixes', fixes)
    section('Other', other)
    lines.push('')
  }
  return new Response(lines.join('\n'), {
    headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
  })
}
