/**
 * AgentSkills plugin for integrating Agent Skills into Strands agents.
 *
 * This module provides the AgentSkills class that implements the Plugin
 * interface to add Agent Skills support. The plugin registers a tool for
 * activating skills and injects skill metadata into the system prompt.
 */

import { readdirSync, statSync, existsSync } from 'fs'
import { join, resolve, relative, sep } from 'path'
import { z } from 'zod'
import { tool } from '../../tools/tool-factory.js'
import { BeforeInvocationEvent } from '../../hooks/events.js'
import { TextBlock, type SystemContentBlock } from '../../types/messages.js'
import { logger } from '../../logging/logger.js'
import { Skill } from './skill.js'
import type { Plugin } from '../../plugins/plugin.js'
import type { LocalAgent } from '../../types/agent.js'
import type { Tool } from '../../tools/tool.js'
import type { ToolContext } from '../../tools/tool.js'

/** A single skill source: filesystem path string, HTTPS URL string, or Skill instance. */
export type SkillSource = string | Skill

/** Configuration for the AgentSkills plugin. */
export interface AgentSkillsConfig {
  /**
   * One or more skill sources. Each element can be:
   * - A `Skill` instance
   * - A path to a skill directory (containing SKILL.md)
   * - A path to a parent directory (containing skill subdirectories)
   * - An `https://` URL pointing directly to raw SKILL.md content
   */
  skills: SkillSource[]

  /** Maximum number of resource files to list in skill responses. Defaults to 20. */
  maxResourceFiles?: number | undefined

  /** If true, throw on skill validation issues. If false (default), warn and load anyway. */
  strict?: boolean | undefined

  /** Custom key for storing plugin state in `agent.appState`. Defaults to `'agent_skills'`. */
  stateKey?: string | undefined
}

const DEFAULT_STATE_KEY = 'agent_skills'
const RESOURCE_DIRS = ['scripts', 'references', 'assets']
const DEFAULT_MAX_RESOURCE_FILES = 20

/**
 * Escape XML special characters in text content.
 */
function escapeXml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;')
}

/**
 * Plugin that integrates Agent Skills into a Strands agent.
 *
 * Provides:
 * 1. A `skills` tool that allows the agent to activate skills on demand
 * 2. System prompt injection of available skill metadata before each invocation
 * 3. Session persistence of activated skill state via `agent.appState`
 *
 * Skills can be provided as filesystem paths (to individual skill directories or
 * parent directories containing multiple skills), HTTPS URLs pointing to raw
 * SKILL.md content, or as pre-built `Skill` instances.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { Skill, AgentSkills } from '@strands-agents/sdk/vended-plugins/skills'
 *
 * // Load from filesystem
 * const plugin = new AgentSkills({
 *   skills: ['./skills/pdf-processing', './skills/'],
 * })
 *
 * // Or provide Skill instances directly
 * const skill = new Skill({ name: 'my-skill', description: 'A custom skill', instructions: 'Do the thing' })
 * const plugin = new AgentSkills({ skills: [skill] })
 *
 * const agent = new Agent({ model, plugins: [plugin] })
 * ```
 */
export class AgentSkills implements Plugin {
  readonly name = 'strands:agent-skills'

  private _skills: Map<string, Skill>
  private readonly _maxResourceFiles: number
  /** When true, skill validation errors throw instead of logging warnings. */
  private readonly _strict: boolean
  private readonly _stateKey: string
  /** Resolves when all async skill sources (e.g. URLs) have been loaded. */
  private _ready: Promise<void>

  constructor(config: AgentSkillsConfig) {
    this._strict = config.strict ?? false
    this._maxResourceFiles = config.maxResourceFiles ?? DEFAULT_MAX_RESOURCE_FILES
    this._stateKey = config.stateKey ?? DEFAULT_STATE_KEY
    const { skills, ready } = this._resolveSkills(config.skills)
    this._skills = skills
    this._ready = ready
  }

  /**
   * Initialize the plugin with the agent instance.
   *
   * Waits for any async skill sources (e.g. URLs) to finish loading, then
   * registers a BeforeInvocationEvent hook that injects skill metadata
   * into the system prompt before each invocation.
   */
  async initAgent(agent: LocalAgent): Promise<void> {
    await this._ready

    if (this._skills.size === 0) {
      logger.warn('no skills were loaded, the agent will have no skills available')
    }
    logger.debug(`skill_count=<${this._skills.size}> | skills plugin initialized`)

    agent.addHook(BeforeInvocationEvent, async (event) => {
      await this._ready
      this._injectSkillsXml(event.agent)
    })
  }

  /**
   * Returns the skills activation tool for auto-registration with the agent.
   */
  getTools(): Tool[] {
    return [this._createSkillsTool()]
  }

  /**
   * Get the list of available skills.
   */
  async getAvailableSkills(): Promise<readonly Skill[]> {
    await this._ready
    return [...this._skills.values()]
  }

  /**
   * Replace all available skills.
   *
   * Each element can be a `Skill` instance, a path to a skill directory
   * (containing SKILL.md), a path to a parent directory containing skill
   * subdirectories, or an `https://` URL pointing directly to raw SKILL.md
   * content.
   *
   * Note: this does not persist state or deactivate skills on any agent.
   * Active skill state is managed per-agent and will be reconciled on the
   * next tool call or invocation.
   */
  setAvailableSkills(skills: SkillSource[]): void {
    const { skills: resolved, ready } = this._resolveSkills(skills)
    this._skills = resolved
    this._ready = ready
  }

  /**
   * Get the list of skills activated by the given agent.
   * Returns skill names in activation order (most recent last).
   */
  getActivatedSkills(agent: LocalAgent): readonly string[] {
    return (this._getStateField(agent, 'activatedSkills') as string[] | undefined) ?? []
  }

  /**
   * Resolve a list of skill sources into Skill instances.
   *
   * Each source can be a Skill instance, a path to a skill directory,
   * a path to a parent directory containing multiple skills, or an
   * HTTPS URL pointing to a SKILL.md file.
   *
   * Synchronous sources (Skill instances and filesystem paths) are resolved
   * immediately into the returned map. Async sources (URLs) are resolved in
   * the background; the returned `ready` promise resolves when all URL
   * fetches have completed and the map has been updated.
   */
  private _resolveSkills(sources: SkillSource[]): { skills: Map<string, Skill>; ready: Promise<void> } {
    const resolved = new Map<string, Skill>()
    const asyncTasks: Promise<void>[] = []

    for (const source of sources) {
      if (source instanceof Skill) {
        if (resolved.has(source.name)) {
          logger.warn(`name=<${source.name}> | duplicate skill name, overwriting previous skill`)
        }
        resolved.set(source.name, source)
      } else if (typeof source === 'string' && source.startsWith('https://')) {
        asyncTasks.push(
          Skill.fromUrl(source, { strict: this._strict }).then(
            (skill) => {
              if (resolved.has(skill.name)) {
                logger.warn(`name=<${skill.name}> | duplicate skill name, overwriting previous skill`)
              }
              resolved.set(skill.name, skill)
            },
            (error) => {
              logger.warn(`url=<${source}> | failed to load skill from URL: ${error}`)
            }
          )
        )
      } else {
        const p = source as string
        const resolvedPath = resolve(p)

        // Probe the filesystem to decide which loader to use instead of
        // relying on exceptions for control flow.
        const isDir = existsSync(resolvedPath) && statSync(resolvedPath).isDirectory()
        const isSkillFile =
          existsSync(resolvedPath) && statSync(resolvedPath).isFile() && resolvedPath.toLowerCase().endsWith('skill.md')
        const hasSkillMd =
          isDir &&
          ['SKILL.md', 'skill.md'].some((name) => {
            const candidate = join(resolvedPath, name)
            return existsSync(candidate) && statSync(candidate).isFile()
          })

        if (isSkillFile || hasSkillMd) {
          // Single skill directory (or direct SKILL.md path)
          try {
            const skill = Skill.fromFile(p, { strict: this._strict })
            if (resolved.has(skill.name)) {
              logger.warn(`name=<${skill.name}> | duplicate skill name, overwriting previous skill`)
            }
            resolved.set(skill.name, skill)
          } catch (error) {
            logger.warn(`path=<${p}> | failed to load skill: ${error}`)
          }
        } else if (isDir) {
          // Parent directory containing skill subdirectories
          try {
            for (const skill of Skill.fromDirectory(p, { strict: this._strict })) {
              if (resolved.has(skill.name)) {
                logger.warn(`name=<${skill.name}> | duplicate skill name, overwriting previous skill`)
              }
              resolved.set(skill.name, skill)
            }
          } catch (error) {
            logger.warn(`path=<${p}> | failed to load skills from directory: ${error}`)
          }
        } else {
          logger.warn(`path=<${p}> | skill source does not exist or is not a valid path`)
        }
      }
    }

    let ready: Promise<void>
    if (asyncTasks.length > 0) {
      ready = Promise.all(asyncTasks).then(() => {
        logger.debug(
          `source_count=<${sources.length}>, resolved_count=<${resolved.size}> | skills resolved (including async)`
        )
      })
    } else {
      logger.debug(`source_count=<${sources.length}>, resolved_count=<${resolved.size}> | skills resolved`)
      ready = Promise.resolve()
    }

    return { skills: resolved, ready }
  }

  /**
   * Create the skills activation tool using the tool() factory with Zod schema.
   */
  private _createSkillsTool(): Tool {
    return tool({
      name: 'skills',
      description:
        'Activate a skill to load its full instructions. ' +
        'Use this tool to load the complete instructions for a skill listed in ' +
        'the available_skills section of your system prompt.',
      inputSchema: z.object({
        skill_name: z.string().min(1).describe('Name of the skill to activate'),
      }),
      callback: async (input: { skill_name: string }, context?: ToolContext): Promise<string> => {
        if (context == null) {
          throw new Error('skills tool requires a ToolContext with an agent reference')
        }
        await this._ready
        return this._activateSkill(input.skill_name, context)
      },
    })
  }

  /**
   * Handle skill activation from the tool callback.
   */
  private _activateSkill(skillName: string, context: ToolContext): string {
    const found = this._skills.get(skillName)
    if (found == null) {
      const available = [...this._skills.keys()].join(', ')
      return `Skill '${skillName}' not found. Available skills: ${available}`
    }

    logger.debug(`skill_name=<${skillName}> | skill activated`)
    this._trackActivatedSkill(context.agent, skillName)
    return this._formatSkillResponse(found)
  }

  /**
   * Record a skill activation in agent state.
   * Maintains an ordered list of activated skill names (most recent last), without duplicates.
   */
  private _trackActivatedSkill(agent: LocalAgent, skillName: string): void {
    const activated = (this._getStateField(agent, 'activatedSkills') as string[] | undefined) ?? []
    this._setStateField(agent, 'activatedSkills', [...activated.filter((n) => n !== skillName), skillName])
  }

  /**
   * Get a field from the plugin's per-agent state dict.
   */
  private _getStateField(agent: LocalAgent, key: string): unknown {
    const data = agent.appState.get(this._stateKey)
    if (data != null && typeof data === 'object' && !Array.isArray(data)) {
      return (data as Record<string, unknown>)[key]
    }
    return undefined
  }

  /**
   * Set a single field in the plugin's per-agent state dict.
   */
  private _setStateField(agent: LocalAgent, key: string, value: unknown): void {
    const data = agent.appState.get(this._stateKey)
    if (data != null && (typeof data !== 'object' || Array.isArray(data))) {
      throw new TypeError(`expected object for state key '${this._stateKey}', got ${typeof data}`)
    }
    const record = (data ?? {}) as Record<string, unknown>
    record[key] = value
    agent.appState.set(this._stateKey, record)
  }

  /**
   * Inject skill metadata into the agent's system prompt.
   *
   * Removes the previously injected XML block (if any) via exact string
   * replacement, then appends a fresh one. Uses agent state to track the
   * injected XML per-agent, so a single plugin instance can be shared
   * across multiple agents safely.
   */
  private _injectSkillsXml(agent: LocalAgent): void {
    const skillsXml = this._generateSkillsXml()
    const systemPrompt = agent.systemPrompt

    if (systemPrompt == null || typeof systemPrompt === 'string') {
      let currentPrompt = systemPrompt ?? ''

      // Remove previously injected XML by exact match
      const lastInjectedXml = this._getStateField(agent, 'lastInjectedXml') as string | undefined
      if (lastInjectedXml != null) {
        if (currentPrompt.includes(lastInjectedXml)) {
          currentPrompt = currentPrompt.replace(lastInjectedXml, '')
        } else {
          logger.warn('unable to find previously injected skills XML in system prompt, re-appending')
        }
      }

      const injection = `\n\n${skillsXml}`
      const newPrompt = currentPrompt ? `${currentPrompt}${injection}` : skillsXml
      const newInjectedXml = currentPrompt ? injection : skillsXml

      this._setStateField(agent, 'lastInjectedXml', newInjectedXml)
      agent.systemPrompt = newPrompt
    } else {
      // SystemContentBlock[] — remove previous block by exact text match, append new one
      const lastInjectedXml = this._getStateField(agent, 'lastInjectedXml') as string | undefined
      let filtered: SystemContentBlock[]
      if (lastInjectedXml != null) {
        filtered = systemPrompt.filter((block) => !(block.type === 'textBlock' && block.text === lastInjectedXml))
        if (filtered.length === systemPrompt.length) {
          logger.warn('unable to find previously injected skills XML in system prompt, re-appending')
        }
      } else {
        filtered = [...systemPrompt]
      }

      this._setStateField(agent, 'lastInjectedXml', skillsXml)
      filtered.push(new TextBlock(skillsXml))
      agent.systemPrompt = filtered
    }
  }

  /**
   * Generate the XML block listing available skills for the system prompt.
   *
   * @example Output with skills:
   * ```xml
   * <available_skills>
   * <skill>
   * <name>pdf-processing</name>
   * <description>Extract text and tables from PDF files</description>
   * <location>/path/to/pdf-processing/SKILL.md</location>
   * </skill>
   * </available_skills>
   * ```
   */
  private _generateSkillsXml(): string {
    if (this._skills.size === 0) {
      return '<available_skills>\nNo skills are currently available.\n</available_skills>'
    }

    const lines: string[] = ['<available_skills>']

    for (const skill of this._skills.values()) {
      lines.push('<skill>')
      lines.push(`<name>${escapeXml(skill.name)}</name>`)
      lines.push(`<description>${escapeXml(skill.description)}</description>`)
      if (skill.path != null) {
        lines.push(`<location>${escapeXml(join(skill.path, 'SKILL.md'))}</location>`)
      }
      lines.push('</skill>')
    }

    lines.push('</available_skills>')
    return lines.join('\n')
  }

  /**
   * Format the tool response when a skill is activated.
   *
   * Includes the full instructions along with relevant metadata fields
   * and a listing of available resource files.
   */
  private _formatSkillResponse(skill: Skill): string {
    if (!skill.instructions) {
      return `Skill '${skill.name}' activated (no instructions available).`
    }

    const parts: string[] = [skill.instructions]

    const metadataLines: string[] = []
    if (skill.allowedTools != null && skill.allowedTools.length > 0) {
      metadataLines.push(`Allowed tools: ${skill.allowedTools.join(', ')}`)
    }
    if (skill.compatibility != null) {
      metadataLines.push(`Compatibility: ${skill.compatibility}`)
    }
    if (skill.path != null) {
      metadataLines.push(`Location: ${join(skill.path, 'SKILL.md')}`)
    }

    if (metadataLines.length > 0) {
      parts.push('\n---\n' + metadataLines.join('\n'))
    }

    if (skill.path != null) {
      const resources = this._listSkillResources(skill.path)
      if (resources.length > 0) {
        parts.push('\nAvailable resources:\n' + resources.map((r) => `  ${r}`).join('\n'))
      }
    }

    return parts.join('\n')
  }

  /**
   * List resource files in a skill's optional directories.
   *
   * Scans `scripts/`, `references/`, and `assets/` subdirectories for files,
   * returning relative paths. Results are capped at maxResourceFiles.
   */
  private _listSkillResources(skillPath: string): string[] {
    const files: string[] = []

    for (const dirName of RESOURCE_DIRS) {
      const resourceDir = join(skillPath, dirName)
      if (!existsSync(resourceDir) || !statSync(resourceDir).isDirectory()) {
        continue
      }

      const entries = readdirSync(resourceDir, { recursive: true, encoding: 'utf-8' })
      for (const entry of entries.sort()) {
        const fullPath = join(resourceDir, entry)
        if (!existsSync(fullPath) || !statSync(fullPath).isFile()) continue

        files.push(relative(skillPath, fullPath).split(sep).join('/'))
        if (files.length >= this._maxResourceFiles) {
          files.push(`... (truncated at ${this._maxResourceFiles} files)`)
          return files
        }
      }
    }

    return files
  }
}
