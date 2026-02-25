## Description

### Motivation

See design https://github.com/strands-agents/docs/pull/528

This PR adds `SkillsPlugin`, a first-class plugin that brings Agent Skills support to Strands. It follows the spec's progressive disclosure model: lightweight metadata (name + description) is injected into the system prompt at startup, and full instructions are loaded on-demand when the agent activates a skill via a tool call. This keeps context usage low while giving agents access to rich, task-specific instructions when needed.

### Public API Changes

New `SkillsPlugin` class and `Skill` dataclass, both exported from the top-level `strands` package:

```python
from strands import Agent, Skill
from strands.plugins.skills import SkillsPlugin

# Load skills from filesystem (individual dirs or parent dirs)
plugin = SkillsPlugin(skills=["./skills/pdf-processing", "./skills/"])

# Or provide Skill instances directly
skill = Skill(name="my-skill", description="A custom skill", instructions="Do the thing")
plugin = SkillsPlugin(skills=[skill])

agent = Agent(plugins=[plugin])
```

The plugin registers a `skills` tool that the agent calls to activate a skill by name. When activated, the tool returns the full instructions along with metadata (allowed tools, compatibility, location) and a listing of available resource files (`scripts/`, `references/`, `assets/`) for filesystem-based skills.

Skill metadata is injected into the system prompt as XML before each invocation, following the recommended format from the integration spec:

```xml
<available_skills>
<skill>
  <name>pdf-processing</name>
  <description>Extract text and tables from PDF files.</description>
  <location>/path/to/pdf-processing/SKILL.md</location>
</skill>
</available_skills>
```

The active skill selection is persisted to `agent.state` for session recovery.

### Use Cases

- **Skill libraries**: Point the plugin at a directory of skills and let the agent pick the right one based on the user's request
- **Dynamic specialization**: Swap agent behavior at runtime without rebuilding prompts or agents
- **Portable skills**: Share skill directories across teams and agents using the Agent Skills standard format

## Related Issues

https://github.com/strands-agents/sdk-python/issues/1181

## Documentation PR

TBD

## Type of Change

New feature

## Testing

- Manually tested using jupyter notebook and set of skills from `anthropic/skills` repository
- 56 unit tests covering the plugin, tool, XML generation, response formatting, resource listing, session persistence, and skill resolution
- 2 integration tests against a real Bedrock model: model-driven skill activation with codeword verification, and direct tool invocation with state persistence checks
- All existing plugin tests (136 total) continue to pass

- [x] I ran `hatch run prepare`

## Checklist
- [x] I have read the CONTRIBUTING document
- [x] I have added any necessary tests that prove my fix is effective or my feature works
- [ ] I have updated the documentation accordingly
- [ ] I have added an appropriate example to the documentation to outline the feature, or no new docs are needed
- [x] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published

----

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
