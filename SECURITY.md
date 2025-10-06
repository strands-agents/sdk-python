# Security Policy

## Supported Versions

We provide security updates for the following versions of the Strands RDS Discovery Tool:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Considerations

### Data Handling

The Strands RDS Discovery Tool handles sensitive information during SQL Server assessments:

- **Database Credentials**: SQL Server usernames and passwords
- **Server Information**: Server names, IP addresses, and configurations
- **Assessment Data**: Database schemas, sizes, and feature usage
- **Network Traffic**: Encrypted connections to SQL Server instances

### Security Features

#### Credential Protection
- **No Credential Storage**: Credentials are never stored in files or logs
- **Memory Handling**: Credentials are cleared from memory after use
- **Connection Encryption**: All SQL Server connections use TLS encryption
- **Parameter Sanitization**: All inputs are validated and sanitized

#### Network Security
- **Encrypted Connections**: Uses ODBC Driver 18 with TrustServerCertificate
- **Connection Timeouts**: Configurable timeouts prevent hanging connections
- **Error Handling**: Sensitive information is not exposed in error messages
- **Audit Logging**: All connection attempts are logged (without credentials)

#### Output Security
- **Log Sanitization**: Credentials and sensitive data are excluded from logs
- **File Permissions**: Output files use appropriate file system permissions
- **Data Minimization**: Only necessary data is collected and stored
- **Temporary Files**: No temporary files containing sensitive data

### Best Practices for Users

#### Credential Management
```python
# ✅ Good: Use environment variables
import os
username = os.getenv('SQL_USERNAME')
password = os.getenv('SQL_PASSWORD')

# ❌ Bad: Hard-coded credentials
username = "sa"
password = "MyPassword123"
```

#### Secure Deployment
- **Network Isolation**: Run assessments from secure network segments
- **Access Control**: Limit tool access to authorized personnel only
- **Audit Trails**: Monitor and log all assessment activities
- **Data Retention**: Implement appropriate data retention policies

#### SQL Server Permissions
Use least-privilege accounts for assessments:

```sql
-- Create dedicated assessment account
CREATE LOGIN rds_assessment WITH PASSWORD = 'SecurePassword123!';
CREATE USER rds_assessment FOR LOGIN rds_assessment;

-- Grant minimum required permissions
GRANT VIEW SERVER STATE TO rds_assessment;
GRANT VIEW ANY DEFINITION TO rds_assessment;
GRANT CONNECT SQL TO rds_assessment;
```

### Threat Model

#### Potential Threats
1. **Credential Exposure**: Unauthorized access to SQL Server credentials
2. **Data Interception**: Network traffic interception during assessment
3. **Privilege Escalation**: Misuse of assessment account permissions
4. **Data Exfiltration**: Unauthorized access to assessment results
5. **Denial of Service**: Resource exhaustion during large assessments

#### Mitigations
1. **Credential Protection**: No storage, memory clearing, encrypted transmission
2. **Network Encryption**: TLS encryption for all database connections
3. **Least Privilege**: Minimal required permissions for assessment accounts
4. **Access Control**: Secure file permissions and access restrictions
5. **Resource Management**: Connection limits and timeout controls

### Compliance Considerations

#### Data Privacy
- **GDPR Compliance**: No personal data collection or processing
- **Data Minimization**: Only technical metadata collected
- **Purpose Limitation**: Data used solely for migration assessment
- **Retention Limits**: Implement appropriate data retention policies

#### Industry Standards
- **SOC 2**: Security controls for service organizations
- **ISO 27001**: Information security management standards
- **NIST Framework**: Cybersecurity framework compliance
- **PCI DSS**: Payment card industry security standards (if applicable)

## Reporting a Vulnerability

### How to Report

If you discover a security vulnerability in the Strands RDS Discovery Tool, please report it responsibly:

1. **Do NOT** create a public GitHub issue
2. **Do NOT** discuss the vulnerability publicly
3. **DO** send details to our security team privately

### Contact Information

**Email**: [security-email@domain.com]
**Subject**: "Security Vulnerability - Strands RDS Discovery Tool"

### Report Format

Please include the following information:

```
## Vulnerability Summary
Brief description of the vulnerability

## Affected Versions
Which versions are affected

## Vulnerability Details
Detailed description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Impact Assessment
Potential impact and severity

## Proof of Concept
Code or screenshots demonstrating the issue

## Suggested Fix
If you have suggestions for remediation

## Disclosure Timeline
Your preferred disclosure timeline
```

### Response Process

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Investigation**: We will investigate and assess the vulnerability
3. **Communication**: We will provide regular updates on our progress
4. **Resolution**: We will develop and test a fix
5. **Disclosure**: We will coordinate responsible disclosure

### Response Timeline

- **Initial Response**: Within 48 hours
- **Investigation**: Within 7 days
- **Fix Development**: Within 30 days (depending on severity)
- **Public Disclosure**: After fix is available and deployed

## Security Updates

### Update Notifications

Security updates will be communicated through:
- **GitHub Security Advisories**: Official security notifications
- **Release Notes**: Detailed information about security fixes
- **CHANGELOG.md**: Version history with security updates
- **Email Notifications**: For critical security issues (if applicable)

### Update Process

1. **Assessment**: Evaluate the security impact
2. **Development**: Create and test security fixes
3. **Testing**: Comprehensive testing of security patches
4. **Release**: Deploy security updates as patch releases
5. **Communication**: Notify users of security updates

### Critical Security Updates

For critical security vulnerabilities:
- **Immediate Response**: Within 24 hours
- **Emergency Patches**: Released as soon as possible
- **Direct Communication**: Email notifications to known users
- **Public Advisory**: GitHub Security Advisory published

## Security Hardening

### Deployment Security

#### Production Environment
```bash
# Set secure file permissions
chmod 600 assessment_results_*.json
chmod 600 assessment_results_*.log
chmod 644 assessment_results_*.csv

# Use dedicated service account
sudo -u rds-assessment python src/rds_discovery.py

# Network isolation
# Run from secure network segment with limited access
```

#### Configuration Security
```python
# Use secure configuration
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="sql",
    username=os.getenv('SQL_USERNAME'),  # From environment
    password=os.getenv('SQL_PASSWORD'),  # From environment
    timeout=30,  # Reasonable timeout
    output_file="secure_assessment"
)
```

### Monitoring and Alerting

#### Security Monitoring
- **Failed Authentication**: Monitor for authentication failures
- **Unusual Access Patterns**: Detect abnormal assessment activities
- **Resource Usage**: Monitor for resource exhaustion attacks
- **Network Traffic**: Monitor database connection patterns

#### Alerting Rules
```
# Example monitoring rules
- Alert on > 5 failed authentications per hour
- Alert on assessment duration > 10 minutes per server
- Alert on > 100 concurrent connections
- Alert on output file access by unauthorized users
```

## Incident Response

### Security Incident Types
1. **Credential Compromise**: SQL Server credentials exposed
2. **Data Breach**: Assessment data accessed by unauthorized parties
3. **System Compromise**: Assessment system compromised
4. **Denial of Service**: Assessment system unavailable

### Response Procedures
1. **Detection**: Identify and confirm security incident
2. **Containment**: Isolate affected systems and limit damage
3. **Investigation**: Determine scope and impact of incident
4. **Recovery**: Restore systems and implement fixes
5. **Lessons Learned**: Document and improve security measures

### Contact Information

**Security Team**: [security-email@domain.com]
**Emergency Contact**: [emergency-phone-number]
**Business Hours**: Monday-Friday, 9 AM - 5 PM UTC

## Security Resources

### Documentation
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SQL Server Security Best Practices](https://docs.microsoft.com/en-us/sql/relational-databases/security/)

### Tools and Libraries
- **pyodbc**: Secure database connectivity
- **boto3**: AWS SDK with security features
- **logging**: Secure logging framework
- **os**: Environment variable access

### Training and Awareness
- Regular security training for contributors
- Security code review processes
- Vulnerability assessment procedures
- Incident response training

---

**Last Updated**: October 5, 2025
**Next Review**: January 5, 2026
