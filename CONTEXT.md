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

## UV as a Python build backend

The uv build backend

Note

The uv build backend is currently in preview and may change without warning.

When preview mode is not enabled, uv uses hatchling as the default build backend.

A build backend transforms a source tree (i.e., a directory) into a source distribution or a wheel. While uv supports all build backends (as specified by PEP 517), it includes a uv_build backend that integrates tightly with uv to improve performance and user experience.

The uv build backend currently only supports Python code. An alternative backend is required if you want to create a library with extension modules.

To use the uv build backend as build system in an existing project, add it to the [build-system] section in your pyproject.toml:

[build-system]
requires = ["uv_build>=0.7.6,<0.8.0"]
build-backend = "uv_build"

Important

The uv build backend follows the same versioning policy, setting an upper bound on the uv_build version ensures that the package continues to build in the future.

You can also create a new project that uses the uv build backend with uv init:

uv init --build-backend uv

uv_build is a separate package from uv, optimized for portability and small binary size. The uv command includes a copy of the build backend, so when running uv build, the same version will be used for the build backend as for the uv process. Other build frontends, such as python -m build, will choose the latest compatible uv_build version.
Modules

The default module name is the package name in lower case with dots and dashes replaced by underscores, and the default module location is under the src directory, i.e., the build backend expects to find src/<package_name>/__init__.py. These defaults can be changed with the module-name and module-root setting. The example below expects a module in the project root with PIL/__init__.py instead:

[tool.uv.build-backend]
module-name = "PIL"
module-root = ""

Include and exclude configuration

To select which files to include in the source distribution, uv first adds the included files and directories, then removes the excluded files and directories. This means that exclusions always take precedence over inclusions.

When building the source distribution, the following files and directories are included:

    pyproject.toml
    The module under tool.uv.build-backend.module-root, by default src/<module-name or project_name_with_underscores>/**.
    project.license-files and project.readme.
    All directories under tool.uv.build-backend.data.
    All patterns from tool.uv.build-backend.source-include.

From these, tool.uv.build-backend.source-exclude and the default excludes are removed.

When building the wheel, the following files and directories are included:

    The module under tool.uv.build-backend.module-root, by default src/<module-name or project_name_with_underscores>/**.
    project.license-files and project.readme, as part of the project metadata.
    Each directory under tool.uv.build-backend.data, as data directories.

From these, tool.uv.build-backend.source-exclude, tool.uv.build-backend.wheel-exclude and the default excludes are removed. The source dist excludes are applied to avoid source tree to wheel source builds including more files than source tree to source distribution to wheel build.

There are no specific wheel includes. There must only be one top level module, and all data files must either be under the module root or in the appropriate data directory. Most packages store small data in the module root alongside the source code.
Include and exclude syntax

Includes are anchored, which means that pyproject.toml includes only <project root>/pyproject.toml. For example, assets/**/sample.csv includes all sample.csv files in <project root>/assets or any child directory. To recursively include all files under a directory, use a /** suffix, e.g. src/**.

Note

For performance and reproducibility, avoid patterns without an anchor such as **/sample.csv.

Excludes are not anchored, which means that __pycache__ excludes all directories named __pycache__ and its children anywhere. To anchor a directory, use a / prefix, e.g., /dist will exclude only <project root>/dist.

All fields accepting patterns use the reduced portable glob syntax from PEP 639, with the addition that characters can be escaped with a backslash.