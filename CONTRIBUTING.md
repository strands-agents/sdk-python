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


## Finding contributions to work on
Looking at the existing issues is a great way to find something to contribute to. We label issues that are well-defined and ready for community contributions with the "ready for contribution" label.

Check our [Ready for Contribution](../../issues?q=is%3Aissue%20state%3Aopen%20label%3A%22ready%20for%20contribution%22) issues for items you can work on.

Before starting work on any issue:
1. Check if someone is already assigned or working on it
2. Comment on the issue to express your interest and ask any clarifying questions
3. Wait for maintainer confirmation before beginning significant work


## Development Environment

This project uses [hatchling](https://hatch.pypa.io/latest/build/#hatchling) as the build backend and [hatch](https://hatch.pypa.io/latest/) for development workflow management.

### Setting Up Your Development Environment

1. Entering virtual environment using `hatch` (recommended), then launch your IDE in the new shell.
   ```bash
   hatch shell
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
   Or run them with coverage:
   ```bash
   hatch test -c
   ```

6. Run integration tests:
   ```bash
   hatch run test-integ
   ```

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically run quality checks before each commit. The hook will run `hatch run format`, `hatch run lint`, `hatch run test`, and `hatch run cz check` when you make a commit, ensuring code consistency.

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


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
