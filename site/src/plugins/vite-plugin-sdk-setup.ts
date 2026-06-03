/**
 * Vite plugin that automatically runs `npm run sdk:clone` and `npm run sdk:generate`
 * if the SDK build artifacts are not present, simplifying the getting-started experience.
 */
import { execSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import type { Plugin } from 'vite'

const SDK_PATHS = ['.build/sdk-typescript', '.build/sdk-python']

function areSdksPresent(): boolean {
  return SDK_PATHS.every((p) => existsSync(p))
}

function run(command: string): void {
  console.log(`[sdk-setup] Running: ${command}`)
  execSync(command, { stdio: 'inherit' })
}

function setupSdks(): void {
  if (areSdksPresent()) return

  console.log('[sdk-setup] SDK build artifacts not found — running sdk:clone and sdk:generate...')
  run('npm run sdk:clone')
  run('npm run sdk:generate')
  console.log('[sdk-setup] SDK setup complete.')
}

export default function sdkSetupPlugin(): Plugin {
  return {
    name: 'vite-plugin-sdk-setup',
    enforce: 'pre',
    // Used during `astro build`
    buildStart() {
      setupSdks()
    },
    // Used during `astro dev` — buildStart fires before the file watcher is running,
    // so newly generated files would be missed by Astro's content layer. configureServer
    // runs after the watcher is set up, and we emit a change event to trigger a re-sync.
    configureServer(server) {
      setupSdks()
      server.watcher.emit('change', '.build/api-docs')
    },
  }
}
