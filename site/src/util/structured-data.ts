/**
 * Shared JSON-LD structured data schemas for SEO.
 * Used by both Head.astro (docs pages) and LandingLayout.astro (homepage).
 */

export function baseSchemas(siteUrl: string) {
  return [
    {
      "@type": "Organization",
      "name": "Strands Agents",
      "url": siteUrl,
      "logo": siteUrl + "/favicon.svg",
      "sameAs": ["https://github.com/strands-agents"],
      "parentOrganization": {
        "@type": "Organization",
        "name": "Amazon Web Services",
        "url": "https://aws.amazon.com"
      }
    },
    {
      "@type": "SoftwareApplication",
      "name": "Strands Agents SDK",
      "applicationCategory": "DeveloperApplication",
      "operatingSystem": "Cross-platform",
      "programmingLanguage": ["Python", "TypeScript"],
      "license": "https://github.com/strands-agents/sdk-python/blob/main/LICENSE",
      "url": siteUrl,
      "author": { "@type": "Organization", "name": "Strands Agents" },
      "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
    }
  ]
}
