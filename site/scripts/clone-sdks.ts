import { execSync } from 'child_process'
import { existsSync } from 'fs'
import { join } from 'path'

interface RepoConfig {
  name: string
  url: string
  path: string
}

const repos: RepoConfig[] = [
  {
    name: 'sdk-typescript',
    url: 'https://github.com/strands-agents/sdk-typescript.git',
    path: '.build/sdk-typescript',
  },
  {
    name: 'sdk-python',
    url: 'https://github.com/strands-agents/sdk-python.git',
    path: '.build/sdk-python',
  },
]

function run(command: string, cwd?: string): void {
  console.log(`Running: ${command}${cwd ? ` (in ${cwd})` : ''}`)
  execSync(command, { stdio: 'inherit', cwd })
}

function cloneOrPull(repo: RepoConfig): void {
  const repoPath = join(process.cwd(), repo.path)

  if (existsSync(join(repoPath, '.git'))) {
    console.log(`\n📦 ${repo.name}: Repository exists, pulling latest changes...`)
    run('git fetch origin', repoPath)
    run('git reset --hard origin/main', repoPath)
  } else {
    console.log(`\n📦 ${repo.name}: Cloning repository...`)
    run(`git clone ${repo.url} ${repo.path}`)
  }
}

function main(): void {
  const args = process.argv.slice(2)
  const targetRepo = args[0]

  console.log('🔄 SDK Clone/Pull Script')

  const reposToProcess = targetRepo ? repos.filter((r) => r.name === targetRepo || r.path.includes(targetRepo)) : repos

  if (targetRepo && reposToProcess.length === 0) {
    console.error(`❌ Unknown repository: ${targetRepo}`)
    console.log('Available repos:', repos.map((r) => r.name).join(', '))
    process.exit(1)
  }

  for (const repo of reposToProcess) {
    cloneOrPull(repo)
  }

  console.log('\n✅ Done!')
}

main()
