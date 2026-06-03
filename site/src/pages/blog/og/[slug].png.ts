import { OGImageRoute } from 'astro-og-canvas'
import { getCollection } from 'astro:content'

const posts = await getCollection('blog', ({ data }) => {
  return import.meta.env.PROD ? !data.draft : true
})

const pages = Object.fromEntries(
  posts.map((post) => [
    post.id,
    {
      title: post.data.title,
      description: post.data.description,
    },
  ])
)

export const { getStaticPaths, GET } = await OGImageRoute({
  param: 'slug',
  pages,
  getSlug: (path) => path,
  getImageOptions: (_path, page) => ({
    title: page.title,
    description: page.description,
    bgGradient: [[14, 14, 14]],
    font: {
      title: {
        families: ['sans-serif'],
        weight: 'Bold',
        color: [255, 255, 255],
        size: 64,
      },
      description: {
        families: ['sans-serif'],
        weight: 'Normal',
        color: [160, 168, 176],
        size: 32,
      },
    },
    border: {
      color: [0, 204, 95],
      width: 20,
      side: 'inline-start',
    },
    padding: 80,
  }),
})
