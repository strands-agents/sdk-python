# RDS Discovery Tool - Run Guide

This guide provides step-by-step instructions for running the Strands RDS Discovery Tool.

## Prerequisites

- Python 3.8 or higher
- Access to SQL Server instances you want to assess
- Network connectivity to target SQL Servers

## Step 1: Setup Environment

### Clone and Navigate to Directory
```bash
git clone <repository-url>
cd strands-rds-discovery
```

### Activate Virtual Environment
**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Prepare Server List

Create a CSV file with your SQL Server instances:

```csv
server_name
server1.domain.com
192.168.1.100
sql-prod-01
```

**Example files provided:**
- `real_servers.csv` - Template with sample server
- `test_template.csv` - Empty template

## Step 3: Run Assessment

### For Windows Authentication
```python
python3 -c "
import sys
sys.path.append('src')
from rds_discovery import strands_rds_discovery

result = strands_rds_discovery(
    input_file='your_servers.csv',
    auth_type='windows'
)
print('Assessment completed')
"
```

### For SQL Server Authentication
```python
python3 -c "
import sys
sys.path.append('src')
from rds_discovery import strands_rds_discovery

result = strands_rds_discovery(
    input_file='your_servers.csv',
    auth_type='sql',
    username='your_username',
    password='your_password'
)
print('Assessment completed')
"
```

## Step 4: Review Results

The tool generates three output files with matching timestamps (single log file per assessment):

- **`RDSdiscovery_[timestamp].csv`** - PowerShell-compatible results
- **`RDSdiscovery_[timestamp].json`** - Detailed JSON data with pricing
- **`RDSdiscovery_[timestamp].log`** - Assessment log for this run only (no persistent log files)

### Example Output Files
```
RDSdiscovery_1759712214.csv
RDSdiscovery_1759712214.json  
RDSdiscovery_1759712214.log
```

## Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `input_file` | Yes | CSV file with server names | `'servers.csv'` |
| `auth_type` | No | Authentication type | `'windows'` or `'sql'` |
| `username` | Conditional | SQL username (required if auth_type='sql') | `'sa'` |
| `password` | Conditional | SQL password (required if auth_type='sql') | `'Password123!'` |
| `timeout` | No | Connection timeout in seconds | `30` (default) |

## Troubleshooting

### Virtual Environment Issues
If you get "ModuleNotFoundError", ensure virtual environment is activated:
```bash
# Check if activated (should show (venv) in prompt)
which python3

# If not activated, run:
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Connection Issues
- Verify server names/IPs are correct
- Check network connectivity: `ping server_name`
- Ensure SQL Server allows remote connections
- Verify authentication credentials
- Check firewall settings (default SQL port: 1433)

### Permission Issues
- For Windows auth: Run from domain-joined machine with appropriate permissions
- For SQL auth: Ensure user has sysadmin or appropriate database permissions

## Quick Test

Test with the provided sample:
```bash
# Activate environment
source venv/bin/activate

# Run assessment
python3 -c "
import sys
sys.path.append('src')
from rds_discovery import strands_rds_discovery

result = strands_rds_discovery(
    input_file='real_servers.csv',
    auth_type='sql',
    username='test',
    password='Password1!'
)
print('Test completed')
"
```

## Support

For issues or questions:
1. Check the log file for detailed error messages
2. Verify all prerequisites are met
3. Review the troubleshooting section
4. Check GitHub issues for similar problems
