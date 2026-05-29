import { defineCollection, type SchemaContext } from 'astro:content'
import { z } from 'astro/zod'
import { docsSchema } from '@astrojs/starlight/schema'
// github-slugger is used by Astro internally for default slug generation.
// We use it here to maintain parity with Astro's default behavior while adding a docs/ prefix.
import { slug as githubSlug } from 'github-slugger'
import { glob, file } from 'astro/loaders'
import { normalizePathToSlug } from './util/links'
import { TagSchema } from './config/tags'

const authorSchema = z.object({
  name: z.string(),
  role: z.string(),
  bio: z.string(),
  avatar: z.string().optional(),
})

export const sourceLinkSchema = z.object({
  repo: z.enum(['sdk-python', 'sdk-typescript']),
  path: z.string(),
})
export type SourceLink = z.infer<typeof sourceLinkSchema>

const blogSchema = z.object({
  title: z.string(),
  date: z.coerce.date(),
  description: z.string(),
  authors: z.array(z.string()),
  tags: z.array(z.string()).default([]),
  draft: z.boolean().default(false),
  coverImage: z.string().optional(),
  // For syndicated posts: set to the original URL so search engines credit the source
  canonicalUrl: z.string().url().optional(),
  // Injected by remark-reading-time plugin at build time
  readingTime: z.string().optional(),
})

export const collections = {
  authors: defineCollection({
    loader: file('src/content/authors.yaml'),
    schema: authorSchema,
  }),
  blog: defineCollection({
    loader: glob({
      base: 'src/content/blog',
      pattern: '**/*.{md,mdx}',
    }),
    schema: blogSchema,
  }),
  testimonials: defineCollection({
    loader: glob({
      base: 'src/content',
      pattern: 'testimonials/**/*.md',
    }),
    schema: ({ image }: SchemaContext) => z.object({
      name: z.string(),
      title: z.string().optional(),
      logo: image().optional(),
      dark_logo: image().optional(),
      link: z.string().url().optional(),
      order: z.number().default(0),
    }),
  }),
  docs: defineCollection({
    loader: glob({
      base: "src/content",
      // We explicitly declare the folders we want to include, as otherwise it includes index.md files
      // in examples which are not intended to be rendered on the site.
      // Long-term we'll be moving examples into the sdk-python repository instead, solving this problem.
      pattern: [
        "404.mdx",
        "docs/README.mdx",

        "docs/user-guide/**/*.mdx",
        "docs/community/**/*.mdx",
        "docs/contribute/**/*.mdx",
        "docs/examples/**/[!index]*.mdx",
        "docs/labs/**/*.mdx",
        "docs/api/python/**/*.mdx",
        "docs/api/typescript/**/*.(md|mdx)",
      ],
      generateId: generateDocsId,
    }),
    schema: docsSchema({
      // We have certain flags/behavior based on the following properties; see CMS-README.md for more info
      extend: z.object({
        // Can be a single value or an array of supported values
        languages: z.union([z.string(), z.array(z.string())]).optional(),
        community: z.boolean().default(false),
        experimental: z.boolean().default(false),
        // Category for TypeScript API docs (classes, interfaces, type-aliases, functions)
        category: z.string().optional(),
        // Integration type for filtering (e.g., 'model-provider' for model providers)
        integrationType: z.enum(['model-provider', 'tool', 'session-manager', 'integration', 'plugin', 'agent-extension']).optional(),
        // Short description for catalog listings
        description: z.string().optional(),
        // Array of slugs that should redirect to this page (e.g., old URLs)
        redirectFrom: z.array(z.string()).optional(),
        // Tags from src/config/tags.yml — drive the build-time "Related pages" block
        tags: z.array(TagSchema).default([]),
        // Pointers to the SDK implementation behind this page. Rendered as an
        // "Implementation" section on headless surfaces only (index.md, llms-full.txt).
        sourceLinks: z.array(sourceLinkSchema).optional(),
      }),
    }),
  }),
}

/**
 * Custom generateId function for docs content collection.
 * This mimics Astro's default slug generation (see node_modules/astro/dist/content/loaders/glob.js)
 * but uses our shared normalizePathToSlug utility for consistency with link resolution.
 */
function generateDocsId({ entry, data }: { entry: string; data: Record<string, unknown> }): string {
  // If frontmatter has a slug, use it directly
  if (data.slug) {
    return `${data.slug}`
  }

  // Normalize the entry path and slugify each segment using github-slugger (same as Astro default)
  const normalized = normalizePathToSlug(entry)
  
  // Handle root README/index -> use 'index' as the slug (Starlight convention for homepage)
  if (!normalized) {
    return 'index'
  }
  
  const slug = normalized
    .split('/')
    .map((segment) => githubSlug(segment))
    .join('/')

  return slug
}