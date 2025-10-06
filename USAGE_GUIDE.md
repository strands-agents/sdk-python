# Strands RDS Discovery Tool - Usage Guide

## Overview

The Strands RDS Discovery Tool performs comprehensive SQL Server assessments for AWS RDS migration planning. It works exactly like the original PowerShell script - just provide a server file and authentication details.

## Basic Usage

### Function Signature
```python
strands_rds_discovery(
    input_file: str,
    auth_type: str = "windows",
    username: Optional[str] = None,
    password: Optional[str] = None,
    timeout: int = 30
) -> str
```

### Windows Authentication
```python
from rds_discovery import strands_rds_discovery

result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='windows'
)
```

### SQL Server Authentication
```python
result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='sql',
    username='sa',
    password='YourPassword123!'
)
```

**Output**: Creates CSV template with sample server entries

### 2. SQL Server Assessment

Assess SQL Server instances for RDS compatibility:

```python
# Windows Authentication
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="windows",
    output_file="assessment_results"
)

# SQL Server Authentication  
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv", 
    auth_type="sql",
    username="sa",
    password="MyPassword123",
    timeout=60,
    output_file="assessment_results"
)
```

## Output Files

The tool generates **3 files** with matching timestamps:

### CSV Output (`RDSdiscovery_[timestamp].csv`)

PowerShell-compatible format with 41 columns:

- **Server Information**: Name, Edition, Version, CPU, Memory
- **Feature Matrix**: 25+ compatibility checks (Y/N format)
- **AWS Recommendations**: Instance type and sizing
- **RDS Compatibility**: Final Y/N determination
- **Pricing**: Monthly cost estimates

**Example:** `RDSdiscovery_1759703573.csv`

**Key Columns:**
- `Server Name`: SQL Server instance name/IP
- `RDS Compatible`: Y/N compatibility status
- `Instance Type`: Recommended AWS RDS instance
- `Memory`: Server memory in MB
- `CPU`: Number of CPU cores
- `Total Storage(GB)`: Total storage capacity

### JSON Output (`RDSdiscovery_[timestamp].json`)

Detailed assessment data with enhanced metadata:

**Example:** `RDSdiscovery_1759703573.json`

```json
{
  "batch_status": "complete",
  "authentication": {
    "type": "sql",
    "username": "sa"
  },
  "performance": {
    "total_servers": 5,
    "total_time": 12.45,
    "average_time_per_server": 2.49,
    "timeout_setting": 30
  },
  "summary": {
    "total_servers": 5,
    "successful_assessments": 4,
    "failed_assessments": 1,
    "rds_compatible": 3,
    "rds_incompatible": 1,
    "success_rate": 80.0
  },
  "results": [
    {
      "server": "sql-server-01",
      "connection": "successful",
      "assessment_time": 2.34,
      "server_info": {
        "edition": "Enterprise Edition",
        "version": "16.0.4085.2",
        "clustered": false
      },
      "resources": {
        "cpu_count": 8,
        "max_memory_mb": 16384
      },
      "database_size_gb": 125.5,
      "total_storage_gb": 500.0,
      "feature_compatibility": {
        "linked_servers": "N",
        "filestream": "Y",
        "always_on_ag": "N"
      },
      "rds_compatible": "N",
      "blocking_features": ["filestream"],
      "aws_recommendation": {
        "instance_type": "db.m6i.2xlarge",
        "match_type": "scaled_up",
        "explanation": "Scaled up from 8 CPU/16GB to meet minimum requirements",
        "pricing": {
          "hourly_rate": 0.768,
          "monthly_estimate": 562.18,
          "currency": "USD",
          "unit": "Hrs"
        }
      }
    }
  ],
  "report_metadata": {
    "generated_at": "2025-10-05 21:30:00 UTC",
    "tool_version": "2.0",
    "csv_output": "RDSdiscovery_1234567890.csv",
    "json_output": "RDSdiscovery_1234567890.json",
    "assessment_type": "SQL Server to RDS Migration Assessment"
  },
  "pricing_summary": {
    "total_monthly_cost": 1687.54,
    "currency": "USD",
    "note": "Costs are estimates and may vary by region and usage"
  }
}
```

### Log Output (`RDSdiscovery_[timestamp].log`)

Complete execution log with success/failure documentation:

**Example:** `RDSdiscovery_1759703573.log`

```
2025-10-05 21:30:00,123 - INFO - Starting RDS Discovery - Action: assess
2025-10-05 21:30:00,124 - INFO - Starting SQL Server assessment - File: servers.csv, Auth: sql
2025-10-05 21:30:00,125 - INFO - Found 5 servers to assess
2025-10-05 21:30:00,126 - INFO - Assessing server 1/5: sql-server-01
2025-10-05 21:30:02,456 - INFO - Assessment completed for sql-server-01 - RDS Compatible: N
2025-10-05 21:30:02,457 - INFO - Assessing server 2/5: sql-server-02
2025-10-05 21:30:04,789 - ERROR - Connection failed for sql-server-02: Authentication failed
2025-10-05 21:30:04,790 - INFO - Assessing server 3/5: sql-server-03
2025-10-05 21:30:07,123 - INFO - Assessment completed for sql-server-03 - RDS Compatible: Y
2025-10-05 21:30:12,456 - INFO - Assessment completed - Files: CSV=RDSdiscovery_1234567890.csv, JSON=RDSdiscovery_1234567890.json, LOG=RDSdiscovery_1234567890.log
```

## Pricing Integration

### AWS Instance Recommendations

The tool provides intelligent instance sizing with cost estimates:

#### Scaling Logic:
1. **exact_match**: Perfect CPU/memory match found
2. **scaled_up**: Scaled to next available size (recommended)
3. **closest_fit**: Best available approximation
4. **fallback**: Estimated when AWS API unavailable

#### Supported Instance Families:
- **General Purpose**: db.m6i, db.m5, db.m4
- **Memory Optimized**: db.r6i, db.r5, db.r4  
- **High Memory**: db.x2iedn, db.x1e

#### Cost Estimation:
- **Hourly rates**: Real-time AWS pricing or fallback estimates
- **Monthly estimates**: 24/7 usage calculation (hours × 30.44 days)
- **Currency**: USD pricing
- **Regional**: Defaults to us-east-1 pricing

### Example Pricing Scenarios:

**Small Server (2 CPU, 4GB)**:
- Recommendation: `db.m6i.large`
- Hourly: $0.192
- Monthly: $140.54

**Large Server (16 CPU, 32GB)**:
- Recommendation: `db.m6i.4xlarge`  
- Hourly: $1.536
- Monthly: $1,124.35

**High Memory Server (8 CPU, 128GB)**:
- Recommendation: `db.r6i.4xlarge`
- Hourly: $2.016
- Monthly: $1,474.46

## Feature Detection

### RDS Blocking Features

Features that **prevent** RDS migration:

- **Linked Servers**: Cross-server queries
- **Log Shipping**: Transaction log shipping
- **FILESTREAM**: File system integration
- **Resource Governor**: Resource management
- **Transaction Replication**: Data replication
- **Extended Procedures**: Custom stored procedures
- **TSQL Endpoints**: HTTP/SOAP endpoints
- **PolyBase**: Big data integration
- **File Tables**: FileTable feature
- **Buffer Pool Extension**: SSD caching
- **Stretch Database**: Azure integration
- **Trustworthy Databases**: Elevated permissions
- **Server Triggers**: DDL triggers
- **Machine Learning**: R/Python integration
- **Data Quality Services**: DQS components
- **Policy Based Management**: Enterprise policies
- **CLR Enabled**: .NET integration
- **Online Indexes**: Enterprise indexing

### RDS Compatible Features

Features that **do not block** RDS migration:

- **Always On Availability Groups**: High availability
- **Always On Failover Cluster Instances**: Clustering
- **Service Broker**: Message queuing
- **SQL Server Integration Services (SSIS)**: ETL processes
- **SQL Server Reporting Services (SSRS)**: Reporting

## Authentication Methods

### Windows Authentication
```python
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="windows"
)
```
- Uses current Windows user credentials
- Requires domain authentication
- No username/password needed

### SQL Server Authentication
```python
result = strands_rds_discovery(
    action="assess", 
    input_file="servers.csv",
    auth_type="sql",
    username="sa",
    password="MySecurePassword123"
)
```
- Uses SQL Server login credentials
- Requires valid username/password
- Password validation (minimum 8 characters)

## Configuration Options

### Timeout Settings
```python
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="sql",
    username="sa", 
    password="password",
    timeout=60  # 60 seconds
)
```
- **Default**: 30 seconds
- **Range**: 5-300 seconds
- **Applies to**: Connection and query timeouts

### Output File Naming
```python
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="windows",
    output_file="production_assessment"
)
```
**Generates**:
- `RDSdiscovery_1234567890.csv`
- `RDSdiscovery_1234567890.json`
- `RDSdiscovery_1234567890.log`

## Return Value Structure

```json
{
  "status": "success|error",
  "outputs": {
    "csv_file": "RDSdiscovery_1234567890.csv",
    "json_file": "RDSdiscovery_1234567890.json", 
    "log_file": "RDSdiscovery_1234567890.log"
  },
  "summary": {
    "servers_assessed": 10,
    "successful_assessments": 8,
    "rds_compatible": 6,
    "success_rate": 80.0
  },
  "message": "Assessment completed successfully"
}
```

## Best Practices

### Server List Preparation
1. **Use FQDNs**: Fully qualified domain names preferred
2. **Test Connectivity**: Verify network access before assessment
3. **Batch Size**: Limit to 50-100 servers per batch for performance
4. **Authentication**: Ensure consistent credentials across servers

### Performance Optimization
1. **Timeout Tuning**: Adjust based on network conditions
2. **Concurrent Processing**: Tool handles multiple servers efficiently
3. **Resource Monitoring**: Monitor memory usage for large batches
4. **Network Bandwidth**: Consider network capacity for remote servers

### Security Considerations
1. **Credential Management**: Use secure credential storage
2. **Network Security**: Ensure encrypted connections
3. **Audit Logging**: Review log files for security events
4. **Permission Principle**: Use least-privilege accounts

## Troubleshooting

### Common Issues

**Connection Failures**:
- Verify server name/IP accuracy
- Check network connectivity and firewall rules
- Validate authentication credentials
- Confirm SQL Server is running and accessible

**Feature Detection Issues**:
- Ensure account has sufficient permissions
- Check for SQL Server version compatibility
- Review query timeout settings
- Verify extended stored procedure availability

**AWS Pricing Issues**:
- AWS API credentials not required (fallback available)
- Regional pricing differences may apply
- Pricing estimates are subject to AWS changes
- Consider Reserved Instance pricing for production

**Output File Issues**:
- Verify write permissions to output directory
- Check available disk space
- Ensure unique output file names
- Review file path validity

### Performance Tuning

**Large Server Lists**:
- Process in smaller batches (50-100 servers)
- Increase timeout for slow networks
- Monitor system resources during execution
- Consider parallel processing for very large environments

**Network Optimization**:
- Use local network connections when possible
- Adjust timeout based on network latency
- Consider VPN performance impact
- Monitor bandwidth usage during assessment
