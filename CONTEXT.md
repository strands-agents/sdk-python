The current workspace is an open-source project called Strands Agent SDK. It enabled developers to quickly create AI agents on AWS. Currently it uses hatch for dependency management, formatting, and other deployment tasks. I would like to modify the project to use uv instead. 

Pay special attention to the contribution guidelines described in CONTRIBUTING.md.

There is an existing GitHub issue describing this change at https://github.com/strands-agents/sdk-python/issues/58. Here is the content of that issue:

[TASK] Migrate from hatch to uv for developing/contributing #58
Open
Task
@theagenticguy
Description
theagenticguy
opened on May 20, 2025
Problem Statement

I'm currently implementing #25 and I have had a tough time with the dev experience using hatch.

    I had to install venv's myself.
    The hatch fmt and hatch run commands weren't very intuitive and hard to debug my lint and test errors.
    Trying to use mypy outside of hatch resulting in not following the mypy config inside pyproject.toml.

uv from Astral fully manages pyproject.toml, manages and installs venv's, installing and managing dependencies and resolution, supports tool configuration, builds, etc. It's also really fast
Proposed Solution

No response
Use Case

I would have a better developer/contributor experience using uv
Alternatives Solutions

No response
Additional Context

No response
Activity
theagenticguy
added
enhancementNew feature or request
on May 20, 2025
zastrowm
zastrowm commented on May 20, 2025
zastrowm
on May 20, 2025
Member

Hey, thanks for the request! The team is open to switching to uv and would gladly accept community contributions to make the switch, as we're not sure when we would be able to get to this ourselves.

If anyone is interested in picking this up, the requirements would be:

    Update all strands-agents repositories to use uv, including sdk-python, tools, and agent-builder.
    Switch all existing commands that use hatch (formatting, linting, etc.) over to uv.
    Update the GitHub workflows from hatch to uv.
    Update the build documentation to mention uv instead of hatch.

zastrowm
added
ready for contributionPull requests welcome
on May 20, 2025
zastrowm
added the
Task
issue type on May 20, 2025
zastrowm
changed the title [-][FEATURE] Consider migrating from hatch to uv for developing/contributing to python-sdk[/-] [+][FEATURE] Migrate from hatch to uv for developing/contributing[/+] on May 20, 2025
zastrowm
changed the title [-][FEATURE] Migrate from hatch to uv for developing/contributing[/-] [+][TASK] Migrate from hatch to uv for developing/contributing[/+] on May 20, 2025
kenliao94
kenliao94 commented on May 20, 2025
kenliao94
on May 20, 2025

Do I just submit a PR if I am interested in taking on this task? Or is there a process that I need to follow?
awsarron
awsarron commented on May 20, 2025
awsarron
on May 20, 2025 · edited by awsarron
Member

PRs are very welcome! Our contribution guide has some details - https://github.com/strands-agents/sdk-python/blob/main/CONTRIBUTING.md.
brianloyal
Add a comment
new Comment
Markdown input: edit mode selected.
Remember, contributions to this repository should follow its contributing guidelines and security policy.
Metadata
Assignees
No one assigned

Labels
enhancementNew feature or request
ready for contributionPull requests welcome
Type
Task
Projects
No projects
Milestone
No milestone

Relationships
None yet

Development
No branches or pull requests

Notifications
You're not receiving notifications from this thread.
Participants
@zastrowm
@kenliao94
@theagenticguy
@awsarron
Issue actions

Footer
© 2025 GitHub, Inc.
Footer navigation

    Terms
    Privacy
    Security
    Status
    Docs
    Contact

[TASK] Migrate from hatch to uv for developing/contributing · Issue #58 · strands-agents/sdk-python
