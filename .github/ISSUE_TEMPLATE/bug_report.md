---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Environment
- **OS**: [e.g., Windows 10, Ubuntu 20.04, macOS 12.0]
- **Python Version**: [e.g., 3.9.7, 3.10.2]
- **Tool Version**: [e.g., v2.0.0, v1.0.0]
- **SQL Server Version**: [e.g., 2019, 2022, 2017]
- **SQL Server Edition**: [e.g., Standard, Enterprise, Express]

## Steps to Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Error Messages
```
Paste any error messages or stack traces here
```

## Log Output
```
Paste relevant log output here (remove any sensitive information)
```

## Assessment Configuration
```python
# Example configuration used when bug occurred
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="sql",
    username="username",
    password="[REDACTED]",
    timeout=30
)
```

## Server List (if applicable)
```csv
# Sample of server list (remove sensitive information)
server_name
server1.example.com
server2.example.com
```

## Screenshots
If applicable, add screenshots to help explain your problem.

## Additional Context
Add any other context about the problem here.

## Workaround
If you found a workaround, please describe it here.

## Impact
- [ ] Blocks assessment completely
- [ ] Causes incorrect results
- [ ] Performance issue
- [ ] Minor inconvenience
- [ ] Other: ___________

## Frequency
- [ ] Always occurs
- [ ] Occurs sometimes
- [ ] Occurred once
- [ ] Unable to reproduce consistently
