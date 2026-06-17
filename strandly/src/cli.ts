#!/usr/bin/env node

import { execSync } from 'node:child_process'
import { resolve } from 'node:path'
import { program } from 'commander'

const ROOT = resolve(import.meta.dirname, '../..')

program.name('strandly').description('Strands monorepo development CLI')

program
  .command('setup')
  .description('Install toolchains and dependencies')
  .action(() => setup())

program
  .command('build')
  .description('Compile the TypeScript SDK')
  .action(() => build())

program
  .command('test')
  .description('Run TypeScript tests')
  .action(() => test())

program
  .command('check')
  .description('Type-check without building')
  .action(() => check())

program
  .command('fmt')
  .description('Format all code')
  .option('--check', 'Fail if anything would change')
  .action((opts) => fmt(opts))

program
  .command('example')
  .description('Run a TypeScript example by name')
  .argument('<name>', 'Example name')
  .action((name) => run('npm start', { cwd: `${ROOT}/strands-ts/examples/${name}` }))

program
  .command('clean')
  .description('Remove all build artifacts')
  .action(() => clean())

program
  .command('ci')
  .description('Full CI pipeline')
  .action(() => {
    fmt({ check: true })
    check()
    build()
    test()
  })

program
  .command('bootstrap')
  .description('First-time setup, build, and test')
  .action(() => {
    setup()
    linkCli()
    build()
    test()
  })

program
  .command('link')
  .description('Install `strandly` on PATH as a live symlink to this repo')
  .action(() => linkCli())

program
  .command('rebuild')
  .description('Clean rebuild from scratch')
  .action(() => {
    clean()
    build()
  })

program.parse()

function run(cmd: string, opts?: { cwd?: string }): void {
  try {
    execSync(cmd, { stdio: 'inherit', cwd: opts?.cwd ?? ROOT })
  } catch (e: unknown) {
    const status = (e as { status?: number }).status ?? 1
    console.error(`\nfailed: ${cmd} (exit ${status})`)
    process.exit(status)
  }
}

function setup(): void {
  run('npm install')
}

function linkCli(): void {
  run('npm link -w strandly')
}

function build(): void {
  run('npm install')
  run('npm run build -w strands-ts')
}

function test(): void {
  run('npm test -w strands-ts')
}

function check(): void {
  run('npm run type-check -w strands-ts')
}

function fmt(opts?: { check?: boolean }): void {
  run(`npx prettier ${opts?.check ? '--check' : '--write'} 'strands-ts/**/*.ts' --ignore-path .gitignore`)
}

function clean(): void {
  run('npm run clean --workspaces --if-present')
}
