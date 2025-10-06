# Contributing to Strands RDS Discovery Tool

Thank you for your interest in contributing to the Strands RDS Discovery Tool! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Standards](#development-standards)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful**: Treat all contributors with respect and professionalism
- **Be inclusive**: Welcome contributors from all backgrounds and experience levels
- **Be collaborative**: Work together constructively and share knowledge
- **Be constructive**: Provide helpful feedback and suggestions

## Getting Started

### Prerequisites

- Python 3.8 or higher
- SQL Server access for testing
- Basic understanding of SQL Server and AWS RDS
- Git for version control

### Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/strands-rds-discovery.git
   cd strands-rds-discovery
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```bash
   python test.py --input-file real_servers.csv --auth-type sql --username test --password Password1!
   ```

## Development Setup

### Project Structure

```
strands-rds-discovery/
├── src/
│   ├── rds_discovery.py      # Main tool implementation
│   └── sql_queries.py        # SQL query definitions
├── tests/
│   └── test_*.py            # Test files
├── docs/
│   ├── README.md            # Project documentation
│   ├── USAGE_GUIDE.md       # Usage instructions
│   └── AWS_PRICING.md       # Pricing documentation
├── examples/
│   └── sample_servers.csv   # Example server lists
└── requirements.txt         # Python dependencies
```

### Key Components

- **Core Assessment Engine**: `src/rds_discovery.py`
- **SQL Query Framework**: `src/sql_queries.py`
- **Test Suite**: `test.py` and `tests/`
- **Documentation**: Comprehensive guides and examples

## Contributing Guidelines

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new functionality
3. **Code Contributions**: Implement features or fix bugs
4. **Documentation**: Improve guides, examples, and comments
5. **Testing**: Add test cases and improve coverage
6. **Performance**: Optimize existing functionality

### Contribution Areas

#### High Priority
- **AWS Pricing Enhancements**: Reserved Instance pricing, multi-region support
- **Feature Detection**: Additional SQL Server features and edge cases
- **Performance Optimization**: Large-scale assessment improvements
- **Error Handling**: Enhanced error recovery and reporting

#### Medium Priority
- **Storage Cost Estimation**: Complete TCO calculations
- **Reporting Enhancements**: Executive summaries and detailed reports
- **API Development**: RESTful API for integration
- **Multi-Database Support**: PostgreSQL, MySQL assessment capabilities

#### Low Priority
- **UI Development**: Web interface for assessments
- **Visualization**: Charts and graphs for assessment results
- **Integration**: Third-party tool integrations
- **Automation**: CI/CD pipeline improvements

## Pull Request Process

### Before Submitting

1. **Create an issue** describing the problem or feature
2. **Fork the repository** and create a feature branch
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Test thoroughly** with real SQL Server instances

### Pull Request Guidelines

1. **Branch Naming**
   ```
   feature/pricing-enhancements
   bugfix/connection-timeout
   docs/usage-guide-update
   ```

2. **Commit Messages**
   ```
   feat: add Reserved Instance pricing support
   fix: resolve connection timeout issues
   docs: update usage guide with pricing examples
   test: add comprehensive pricing test cases
   ```

3. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement

   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

### Review Process

1. **Automated Checks**: All tests must pass
2. **Code Review**: At least one maintainer review required
3. **Documentation Review**: Ensure docs are updated
4. **Testing Verification**: Manual testing with real data
5. **Merge**: Squash and merge after approval

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

```markdown
## Bug Description
Clear description of the issue

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- SQL Server Version: [e.g., 2019, 2022]
- Tool Version: [e.g., v2.0.0]

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Error Messages
```
[Include any error messages or logs]
```

## Additional Context
Any other relevant information
```

### Feature Requests

For feature requests, please provide:

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Solution
How should this feature work?

## Alternatives Considered
Other approaches you've considered

## Additional Context
Any other relevant information
```

## Development Standards

### Code Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type hints for function parameters and returns
- **Docstrings**: Document all functions and classes
- **Comments**: Explain complex logic and business rules

### Example Code Style

```python
def assess_sql_server(
    server_name: str, 
    auth_type: str, 
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Assess SQL Server instance for RDS compatibility.
    
    Args:
        server_name: SQL Server instance name or IP
        auth_type: Authentication type ('windows' or 'sql')
        username: SQL Server username (required for SQL auth)
        password: SQL Server password (required for SQL auth)
        timeout: Connection timeout in seconds
        
    Returns:
        Dictionary containing assessment results
        
    Raises:
        ConnectionError: If unable to connect to SQL Server
        AuthenticationError: If authentication fails
    """
    # Implementation here
    pass
```

### Testing Standards

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test complete workflows
- **Real Data Testing**: Test with actual SQL Server instances
- **Error Case Testing**: Test error handling and edge cases

### Documentation Standards

- **README**: Keep main README current and comprehensive
- **Usage Guide**: Detailed usage instructions with examples
- **API Documentation**: Document all public functions
- **Change Log**: Maintain detailed change history

## Security Considerations

### Sensitive Data

- **Never commit credentials** or connection strings
- **Use environment variables** for sensitive configuration
- **Sanitize logs** to remove sensitive information
- **Follow security best practices** for database connections

### Code Security

- **Input validation**: Validate all user inputs
- **SQL injection prevention**: Use parameterized queries
- **Error handling**: Don't expose sensitive information in errors
- **Dependency management**: Keep dependencies updated

## Performance Guidelines

### Optimization Priorities

1. **Connection Management**: Efficient database connections
2. **Query Optimization**: Fast and efficient SQL queries
3. **Memory Usage**: Minimize memory footprint for large assessments
4. **Concurrent Processing**: Optimize for multiple server assessments

### Performance Testing

- **Benchmark tests**: Measure performance improvements
- **Load testing**: Test with large server lists
- **Memory profiling**: Monitor memory usage patterns
- **Network optimization**: Minimize network overhead

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Release notes prepared
- [ ] Security review completed

## Getting Help

### Resources

- **Documentation**: Check existing documentation first
- **Issues**: Search existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Examples**: Review example code and usage patterns

### Contact

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [Maintainer email if applicable]

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **Documentation**: Attribution for significant contributions

Thank you for contributing to the Strands RDS Discovery Tool!
