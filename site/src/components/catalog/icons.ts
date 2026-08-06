// Python/TypeScript marks shared with the changelog page.
import { ICONS } from '../changelog/icons'

export { ICONS as LANG_ICONS } from '../changelog/icons'

/** Star glyph for the GitHub-stars stat (24x24, fill=currentColor). */
export const STAR_ICON = 'M12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z'

/** Arrow-into-tray glyph for the downloads stat (24x24, fill=currentColor). */
export const DOWNLOAD_ICON = 'M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z'

/* Brand marks for card footer links (24x24, fill=currentColor). Each is a path
   list so multi-path marks (PyPI's two snakes) render the same as single-path
   ones. */

/** GitHub octocat mark. */
export const GITHUB_ICON: readonly string[] = [
  'M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12',
]

/** PyPI: the Python two-snakes mark, shared with the language toggle. */
export const PYPI_ICON: readonly string[] = ICONS.python

/** npm block logo: the "n" cut out of a filled square. */
export const NPM_ICON: readonly string[] = [
  'M1.763 0C.786 0 0 .786 0 1.763v20.474C0 23.214.786 24 1.763 24h20.474c.977 0 1.763-.786 1.763-1.763V1.763C24 .786 23.214 0 22.237 0zM5.13 5.323l13.837.019-.009 13.836h-3.464l.01-10.382h-3.456L12.04 19.17H5.113z',
]
