# Custom Landing Page and Testimonials

Part of the site architecture reference. See [SITE-ARCHITECTURE.md](../SITE-ARCHITECTURE.md) for the overview. Paths are relative to `site/`.

## Custom Landing Page

The landing page uses a custom layout that provides the Starlight header without the full documentation page structure, allowing for full-width marketing content.

### Landing Layout (`src/layouts/LandingLayout.astro`)

**What it does:** Provides a minimal layout with the Starlight header, theme support, and CSS variables, but without the sidebar, table of contents, or content constraints of documentation pages.

**Key features:**
- Mocks `Astro.locals.starlightRoute` with minimal data needed for the Header component
- Mocks `Astro.locals.t` translation function (with `.all()` method for Search component)
- Includes `SiteScripts` for Shortbread consent and WebSDK

**Usage:**
```astro
---
import LandingLayout from '../layouts/LandingLayout.astro'
---

<LandingLayout title="Page Title" description="Optional description">
  <!-- Full-width content here -->
</LandingLayout>
```

### Landing Page (`src/pages/index.astro`)

The main landing page includes:
- Animated parallax curves background (replicating strandsagents.com effect)
- Hero section with frosted glass effect
- Feature cards that expand on hover to show descriptions
- Testimonials slider with fade transitions and auto-play
- Footer with `Copyright` component (left-aligned, `--sl-color-bg-nav` background)

**Assets:**
- `src/assets/curve-primary.svg` and `src/assets/curve-secondary.svg` - Animated strand patterns
- `src/assets/icons/icon-*.svg` - Feature card icons

## Testimonials Content Collection

Testimonials are managed as a content collection of Markdown files, with company logos stored alongside them.

### Schema (`src/content.config.ts`)

```typescript
testimonials: defineCollection({
  loader: glob({ base: 'src/content', pattern: 'testimonials/**/*.md' }),
  schema: ({ image }) => z.object({
    name: z.string(),
    title: z.string().optional(),
    logo: image().optional(),       // Light-mode company logo
    dark_logo: image().optional(),  // Dark-mode variant (falls back to logo)
    order: z.number().default(0),
  }),
})
```

Using Astro's `image()` helper ensures logos are processed through the asset pipeline (hashed, optimized) at build time.

### Content Location

`src/content/testimonials/` — each company has a `.md` file and its logo(s) stored alongside it:

```
src/content/testimonials/
├── smartsheet.md
├── smartsheet-logo.svg
├── smartsheet-logo-white.svg   ← dark-mode variant
├── landchecker.md
├── landchecker-logo.svg
└── ...
```

### File Format

Each testimonial is a Markdown file with frontmatter metadata and the quote as the body:

```markdown
---
name: JB Brown
title: VP Engineering, Smartsheet
logo: ./smartsheet-logo.svg
dark_logo: ./smartsheet-logo-white.svg
order: 1
---

At Smartsheet, we chose Strands...
```

The `order` field controls display sequence in the slider. Logo paths are relative to the file.

### Dark/Light Logo Switching

The landing page renders both `logo` and `dark_logo` (falling back to `logo` when no dark variant exists) and uses CSS to show the appropriate one based on Starlight's `[data-theme]` attribute:

```css
.logo-dark { display: none; }
[data-theme='dark'] .logo-light { display: none; }
[data-theme='dark'] .logo-dark { display: block; }
```
