import { defineConfig } from 'vitest/config'
import { resolve } from 'node:path'

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: { label: 'unit' },
          include: ['__tests__/**/*.test.ts'],
        },
        resolve: {
          alias: {
            'strands:agent/tool-provider': resolve(__dirname, '__fixtures__/tool-provider.ts'),
            'strands:agent/host-log': resolve(__dirname, '__fixtures__/host-log.ts'),
            '$/fixtures': resolve(__dirname, '../strands-ts/src/__fixtures__'),
          },
        },
      },
      {
        test: {
          name: { label: 'guest' },
          include: ['test/guest/**/*.test.ts'],
          testTimeout: 60_000,
          pool: 'forks',
        },
      },
    ],
  },
})
