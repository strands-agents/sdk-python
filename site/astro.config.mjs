// @ts-check
import { defineConfig } from 'astro/config'
import starlight from '@astrojs/starlight'
import path from 'node:path'
import remarkMkdocsSnippets from './src/plugins/remark-mkdocs-snippets.ts'
import sdkSetupPlugin from './src/plugins/vite-plugin-sdk-setup.ts'
import remarkReadingTime from './src/plugins/remark-reading-time.ts'
import watchNavigationPlugin from './src/plugins/vite-plugin-watch-navigation.ts'

import { loadSidebarFromConfig } from "./src/sidebar.ts"
import { sitemapWithLastmod } from "./src/plugins/sitemap-lastmod.ts"
import AutoImport from './src/plugins/astro-auto-import.ts'
import astroExpressiveCode from "astro-expressive-code"
import mdx from '@astrojs/mdx';
import astroBrokenLinksChecker from './scripts/astro-broken-links-checker-index.js';

// Generate sidebar from navigation.yml config (validates against existing content files)
// Top-level groups will be rendered as tabs by the custom Sidebar component
const sidebar = loadSidebarFromConfig(
  path.resolve('./src/config/navigation.yml'),
  path.resolve('./src/content')
)

// https://astro.build/config
export default defineConfig({
  site: 'https://strandsagents.com',
  base: process.env.ASTRO_BASE_PATH || '/',
  vite: {
    plugins: [sdkSetupPlugin(), watchNavigationPlugin()],
    // TODO once we separate out CMS build from TS verification, fix this
    // https://github.com/withastro/astro/issues/14117
		ssr: {
			noExternal: ['zod'],
		},
	},
  markdown: {
    remarkPlugins: [remarkMkdocsSnippets, remarkReadingTime],
  },
  integrations: [
    // Sitemap with git-based <lastmod> dates — must be before Starlight
    // so Starlight detects it and skips its own sitemap integration
    sitemapWithLastmod('src/content'),
    astroExpressiveCode({
      themes: ['github-light', 'github-dark'],
      // Follow Starlight's data-theme attribute instead of the browser's prefers-color-scheme
      themeCssSelector: (theme) => `[data-theme='${theme.type}']`,
      styleOverrides: {
        // Match the accent color from the site theme
        frames: {
          shadowColor: 'transparent',
        },
      },
    }),
    mdx(),
    starlight({
      social: [],
      head: [
        { tag: 'meta', attrs: { property: 'og:image', content: 'https://strandsagents.com/og-image.png' } },
        { tag: 'meta', attrs: { name: 'twitter:image', content: 'https://strandsagents.com/og-image.png' } },
      ],
      markdown: {
        // API docs are symlinked from .build/api-docs; processedDirs ensures Starlight's
        // rehype plugins (e.g. heading anchor links) run on the real resolved paths.
        processedDirs: [path.resolve('.build/api-docs')],
      },
      title: 'Strands Agents SDK',
      description: 'A model-driven approach to building AI agents in just a few lines of code.',
      sidebar: sidebar,
      routeMiddleware: './src/route-middleware.ts',
      customCss: [
        './src/styles/custom.css',
      ],
      logo: {
        light: './src/assets/logo-light.svg',
        dark: './src/assets/logo-dark.svg',
        replacesTitle: false,
      },
      editLink: {
        baseUrl: 'https://github.com/strands-agents/docs/edit/main/',
      },
      components: {
        Head: './src/components/overrides/Head.astro',
        Header: './src/components/overrides/Header.astro',
        Hero: './src/components/overrides/Hero.astro',
        MarkdownContent: './src/components/overrides/MarkdownContent.astro',
        Sidebar: './src/components/overrides/Sidebar.astro',
        PageFrame: './src/components/overrides/PageFrame.astro',
      },
  }),
   astroBrokenLinksChecker({
      checkExternalLinks: false,      // Optional: check external links (default: false)
      cacheExternalLinks: false,      // Optional: cache verified external links to disk (default: true)
      throwError: true,               // Optional: fail the build if broken links are found (default: false)
      linkCheckerDir: '.link-checker' // Optional: directory for cache and log files (default: '.link-checker')
    }),
   AutoImport({
      imports: [
        {
          '@astrojs/starlight/components': [
            ['TabItem', 'Tab']
          ],
          './src/components/AutoSyncTabs.astro': [
            ['default', "Tabs"]
          ]
        },
      ],
      defaultComponents: {
        // override 'a' links so that we can use relative urls
        a: './src/components/PageLink.astro'
      }
    }),
  ],
})