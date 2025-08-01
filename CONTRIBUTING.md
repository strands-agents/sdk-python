# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the [Bug Reports](../../issues/new?template=bug_report.yml) file to report bugs or [Feature Requests](../../issues/new?template=feature_request.yml) to suggest features.

For a list of known bugs and feature requests:
- Check [Bug Reports](../../issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) for currently tracked issues
- See [Feature Requests](../../issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement) for requested enhancements

When filing an issue, please check for already tracked items

Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used (commit ID)
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Development Environment

This project uses [hatchling](https://hatch.pypa.io/latest/build/#hatchling) as the build backend and [hatch](https://hatch.pypa.io/latest/) for development workflow management.

### Setting Up Your Development Environment

1. Entering virtual environment using `hatch` (recommended), then launch your IDE in the new shell.
   ```bash
   hatch shell dev
   ```

   Alternatively, install development dependencies in a manually created virtual environment:
   ```bash
   pip install -e ".[dev]" && pip install -e ".[litellm]"
   ```


2. Set up pre-commit hooks:
   ```bash
   pre-commit install -t pre-commit -t commit-msg
   ```
   This will automatically run formatters and conventional commit checks on your code before each commit.

3. Run code formatters manually:
   ```bash
   hatch fmt --formatter
   ```

4. Run linters:
   ```bash
   hatch fmt --linter
   ```

5. Run unit tests:
   ```bash
   hatch test
   ```

6. Run integration tests:
   ```bash
   hatch run test-integ
   ```

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically run quality checks before each commit. The hook will run `hatch run format`, `hatch run lint`, `hatch run test`, and `hatch run cz check` on when you make a commit, ensuring code consistency.

The pre-commit hook is installed with:

```bash
pre-commit install
```

You can also run the hooks manually on all files:

```bash
pre-commit run --all-files
```

### Code Formatting and Style Guidelines

We use the following tools to ensure code quality:
1. **ruff** - For formatting and linting
2. **mypy** - For static type checking

These tools are configured in the [pyproject.toml](./pyproject.toml) file. Please ensure your code passes all linting and type checks before submitting a pull request:

```bash
# Run all checks
hatch fmt --formatter
hatch fmt --linter
```

If you're using an IDE like VS Code or PyCharm, consider configuring it to use these tools automatically.

For additional details on styling, please see our dedicated [Style Guide](./STYLE_GUIDE.md).


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Create a branch.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Format your code using `hatch fmt --formatter`.
4. Run linting checks with `hatch fmt --linter`.
5. Ensure local tests pass with `hatch test` and `hatch run test-integ`.
6. Commit to your branch using clear commit messages following the [Conventional Commits](https://www.conventionalcommits.org) specification.
7. Send us a pull request, answering any default questions in the pull request interface.
8. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute to.

You can check:
- Our known bugs list in [Bug Reports](../../issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) for issues that need fixing
- Feature requests in [Feature Requests](../../issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement) for new functionality to implement


## Dependency Management with uv lock

This project uses `uv lock` to ensure reproducible dependency resolution across all environments. The `uv.lock` file contains exact versions of all dependencies and **must be committed to git**.

### For Developers

**Initial Setup:**
```bash
# Install exact dependencies from lock file
uv sync --dev
```

**Daily Development:**
```bash
# Use locked dependencies for all commands
uv run poe test
uv run poe lint
uv run poe format
```

**Adding New Dependencies:**
```bash
# Add new dependency and update lock file
uv add "new-package>=1.0.0"
# The lock file is automatically updated
git add pyproject.toml uv.lock
git commit -m "feat: add new-package dependency"
```

**Updating Dependencies:**
```bash
# Update to latest compatible versions
uv lock

# Or upgrade to latest versions (major updates)
uv lock --upgrade

# Always commit lock file changes
git add uv.lock
git commit -m "chore: update dependencies"
```

### Why Lock Files Matter

- **Reproducible Builds**: Same exact dependencies across dev, CI, and production
- **Faster CI**: Pre-resolved dependencies eliminate resolution time
- **Security**: Pinned versions prevent supply chain attacks
- **Team Consistency**: Everyone gets identical dependency trees

### Lock File Commands

| Command | Purpose |
|---------|---------|
| `uv sync --dev` | Install exact dependencies from lock file |
| `uv sync` | Install production dependencies only |
| `uv lock` | Update lock file with latest compatible versions |
| `uv lock --upgrade` | Upgrade all dependencies to latest versions |

**Important**: Always commit `uv.lock` changes alongside `pyproject.toml` changes!


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
