const REPOS = ['strands-agents/sdk-python', 'strands-agents/sdk-typescript']
const FALLBACK = '5,800+'
const TIMEOUT_MS = 5000

export async function getStarCount(): Promise<string> {
  try {
    const counts = await Promise.all(
      REPOS.map(async (repo) => {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS)
        try {
          const res = await fetch(`https://api.github.com/repos/${repo}`, {
            signal: controller.signal,
          })
          if (!res.ok) return 0
          const data = await res.json()
          return data.stargazers_count ?? 0
        } finally {
          clearTimeout(timeout)
        }
      })
    )
    const total = counts.reduce((a: number, b: number) => a + b, 0)
    if (total === 0) return FALLBACK
    const rounded = Math.floor(total / 100) * 100
    return rounded.toLocaleString() + '+'
  } catch {
    return FALLBACK
  }
}
