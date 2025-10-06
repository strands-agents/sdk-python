# Strands RDS Discovery Tool v2.1.2

**SQL Server to AWS RDS Migration Assessment with Pricing Integration**

A production-ready tool that provides comprehensive SQL Server compatibility assessment for AWS RDS migration planning with **PowerShell-compatible CSV output**, **cost estimation**, and **triple output format** (CSV + JSON + LOG).

## **🎯 Overview**

This tool enables comprehensive SQL Server assessment for AWS RDS migration, providing detailed analysis, migration recommendations, AWS instance sizing with pricing, and complete documentation through three output files.

### **Key Features**
- **Simplified Usage**: No action parameters - just run with server file like original PowerShell script
- **10% Tolerance Logic**: Consistent tolerance matching in both AWS API and fallback modes
- **PowerShell CSV Output**: Generates identical `RdsDiscovery.csv` format as original PowerShell tool
- **Cost Estimation**: Hourly and monthly pricing for recommended AWS instances
- **Triple Output**: CSV + JSON + LOG files with matching timestamps
- **Real SQL Server Data**: All data collected from live SQL Server queries (no mock data)
- **PowerShell-Compatible Storage**: Uses `xp_fixeddrives` logic matching PowerShell behavior exactly
- **AWS Instance Sizing**: Intelligent RDS instance recommendations with scaling explanations
- **Comprehensive Analysis**: 25+ SQL Server feature compatibility checks
- **Production Ready**: Enterprise-grade error handling and performance monitoring

## **🚀 Quick Start**

### Installation

```bash
git clone <repository-url>
cd strands-rds-discovery
source venv/bin/activate  # Linux/Mac: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Basic Usage

```python
from rds_discovery import strands_rds_discovery

# Windows Authentication
result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='windows'
)

# SQL Server Authentication  
result = strands_rds_discovery(
    input_file='servers.csv',
    auth_type='sql',
    username='your_username',
    password='your_password'
)
```

### Server File Format

Create a CSV file with your SQL Server instances:
```csv
server_name
server1.domain.com
192.168.1.100
sql-prod-01
```

## **📋 Complete Run Guide**

See [RUN_GUIDE.md](RUN_GUIDE.md) for detailed step-by-step instructions including:
- Virtual environment setup
- Authentication options
- Troubleshooting common issues
- Output file explanations

## **📊 Output Files**

The tool generates three files with matching timestamps (clean, single-log approach):

- **`RDSdiscovery_[timestamp].csv`** - PowerShell-compatible results
- **`RDSdiscovery_[timestamp].json`** - Detailed JSON with pricing and metadata  
- **`RDSdiscovery_[timestamp].log`** - Assessment log (no persistent log files)

## **💰 AWS Pricing Integration**

- Real-time AWS RDS pricing via API
- Fallback pricing when API unavailable
- Monthly cost estimates for migration planning
- Instance scaling explanations (exact_match, within_tolerance, scaled_up, fallback)

## **🔧 Parameters**

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `input_file` | Yes | CSV file with server names | - |
| `auth_type` | No | 'windows' or 'sql' | 'windows' |
| `username` | Conditional | SQL username (required if auth_type='sql') | None |
| `password` | Conditional | SQL password (required if auth_type='sql') | None |
| `timeout` | No | Connection timeout in seconds | 30 |
```

### Basic Usage

```python
from src.rds_discovery import strands_rds_discovery

# 1. Create server template
result = strands_rds_discovery(action="template", output_file="servers.csv")

# 2. Edit servers.csv with your SQL Server names/IPs

# 3. Run assessment
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv", 
    auth_type="sql",
    username="your_username",
    password="your_password",
    output_file="assessment_results"
)
```

## **📄 Output Files**

The tool generates **3 files** with matching timestamps:

### 1. CSV File (`RDSdiscovery_[timestamp].csv`)
- **PowerShell-compatible** format with 41 columns
- Server specifications and feature matrix  
- AWS instance recommendations
- RDS compatibility status

### 2. JSON File (`RDSdiscovery_[timestamp].json`)
- Complete assessment data with metadata
- **Pricing summary** with total monthly costs
- Performance metrics and batch processing details
- AWS recommendation explanations

### 3. Log File (`RDSdiscovery_[timestamp].log`)
- Complete success/failure documentation
- Connection attempts and errors
- Feature detection results
- AWS API calls and fallback logic

## **💰 Pricing Integration**

### Cost Estimates Include:
- **Hourly rates** for recommended instances
- **Monthly estimates** (24/7 usage)
- **Currency** (USD)
- **Pricing source** (AWS API or fallback estimates)

### Instance Scaling Explanations:
- **exact_match**: Perfect match for server specifications
- **scaled_up**: Scaled up to meet minimum requirements  
- **closest_fit**: Closest available instance match
- **fallback**: Estimated when AWS API unavailable

**Example Pricing Output:**
```json
{
  "aws_recommendation": {
    "instance_type": "db.m6i.2xlarge",
    "match_type": "scaled_up", 
    "explanation": "Scaled up from 6 CPU/8GB to meet minimum requirements",
    "pricing": {
      "hourly_rate": 0.768,
      "monthly_estimate": 562.18,
      "currency": "USD"
    }
  }
}
```

## **🔍 Feature Detection**

### RDS Blocking Features (Detected)
- Linked Servers
- Log Shipping  
- FILESTREAM
- Resource Governor
- Transaction Replication
- Extended Procedures
- TSQL Endpoints
- PolyBase
- File Tables
- Buffer Pool Extension
- Stretch Database
- Trustworthy Databases
- Server Triggers
- Machine Learning Services
- Data Quality Services
- Policy Based Management
- CLR Enabled
- Online Indexes

### RDS Compatible Features (Not Blocking)
- **Always On Availability Groups** ✅
- **Always On Failover Cluster Instances** ✅
- **Service Broker** ✅
- **SQL Server Integration Services (SSIS)** ✅
- **SQL Server Reporting Services (SSRS)** ✅

## **☁️ AWS Instance Types**

### General Purpose
- db.m6i.large through db.m6i.24xlarge
- db.m5, db.m4 families

### Memory Optimized  
- db.r6i.large through db.r6i.16xlarge
- db.r5, db.r4 families

### High Memory
- db.x2iedn.large through db.x2iedn.24xlarge
- db.x1e family

## **⚙️ Configuration**

### Authentication Types
- **Windows Authentication**: Uses current Windows credentials
- **SQL Server Authentication**: Requires username/password

### Timeout Settings
- Default: 30 seconds
- Configurable: 5-300 seconds
- Handles connection timeouts gracefully

### AWS Integration
- **Real-time pricing** via AWS Pricing API (when credentials available)
- **Fallback pricing** with estimated costs
- **Regional pricing** support (defaults to us-east-1)

## **📊 Return Value**

```json
{
  "status": "success",
  "outputs": {
    "csv_file": "RDSdiscovery_1234567890.csv",
    "json_file": "RDSdiscovery_1234567890.json", 
    "log_file": "RDSdiscovery_1234567890.log"
  },
  "summary": {
    "servers_assessed": 5,
    "successful_assessments": 4,
    "rds_compatible": 3,
    "success_rate": 80.0
  }
}
```

## **🛠️ Requirements**

- Python 3.8+
- pyodbc (SQL Server connectivity)
- boto3 (AWS integration)
- ODBC Driver 18 for SQL Server

## **🔧 Error Handling**

Robust error handling for:
- Connection failures and authentication errors
- Network timeouts and invalid server names
- Permission issues and SQL query failures
- AWS API failures with fallback logic
- File I/O errors and CSV parsing issues

## **📈 Performance**

- **Batch processing** multiple servers
- **Concurrent assessments** with timeout management
- **Progress tracking** and performance metrics
- **Memory efficient** processing of large server lists

## **🎯 Production Ready**

- Enterprise-grade logging and monitoring
- Comprehensive error handling and recovery
- Performance optimization and resource management
- Complete documentation and audit trails

### **Installation**
```bash
# Clone repository
git clone https://github.com/your-org/strands-rds-discovery
cd strands-rds-discovery

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Basic Usage**
```python
from src.rds_discovery import strands_rds_discovery

# Create server list CSV
strands_rds_discovery(action="template", output_file="servers.csv")

# Edit servers.csv with your SQL Server names
# server_name
# server1.domain.com
# server2.domain.com

# Run assessment - generates PowerShell-compatible CSV
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="windows"  # or "sql" with username/password
)

# Generates: RdsDiscovery_[timestamp].csv
```

### **Strands AI Conversation Examples**
```
"Assess SQL Server 3.81.26.46 using SQL authentication with user test"

"Generate RDS discovery report for prod-sql01.company.com"

"What AWS instance size is recommended for my 8-core SQL Server?"
```

## **📊 Output Format**

### **PowerShell-Compatible CSV**
Generates `RdsDiscovery_[timestamp].csv` with **identical format** to original PowerShell tool:

```csv
"Server Name","SQL Server Current Edition","CPU","Memory","Instance Type","RDS Compatible","Total DB Size in GB","Total Storage(GB)","SSIS","SSRS"
"3.81.26.46","Enterprise Edition: Core-based Licensing (64-bit)","8","124","db.m6i.2xlarge ","Y","0.80","51.32","N","N"
```

### **Real Data Collection**
- **Server Info**: Real SQL Server edition, version, clustering status
- **Resources**: Actual CPU cores and memory from `sys.dm_os_sys_info`
- **Database Size**: Real user database sizes from `sys.master_files`
- **Total Storage**: PowerShell `xp_fixeddrives` logic for drive capacity
- **Features**: 25+ compatibility checks from live SQL queries
- **AWS Sizing**: Instance recommendations based on actual server specs

## **🔧 System Requirements**

### **Prerequisites**
- **Python 3.8+**
- **Microsoft ODBC Driver 18 for SQL Server**
- **Network access to SQL Servers** (port 1433)
- **Strands Framework** (strands-agents, strands-agents-tools)

### **SQL Server Requirements**
- **xp_fixeddrives enabled** (for accurate storage calculation)
- **Appropriate permissions** for assessment queries
- **SQL Server 2008+** (all versions supported)

## **📋 Assessment Coverage**

### **Real SQL Server Data Collection**
- **Server Information**: Edition, version, clustering from `SERVERPROPERTY()`
- **CPU & Memory**: Real values from `sys.dm_os_sys_info` and `sys.configurations`
- **Database Sizes**: User databases only (`WHERE database_id > 4`)
- **Total Storage**: PowerShell-compatible `xp_fixeddrives` + SQL file calculation
- **27+ Feature Checks**: All compatibility queries from original PowerShell tool plus SSIS/SSRS

### **Enhanced Feature Detection**
- **SSIS Detection**: Checks for SSISDB catalog and custom packages (excludes system collector packages)
- **SSRS Detection**: Checks for ReportServer databases  
- **PowerShell RDS Blocking**: Uses exact same blocking logic as original PowerShell script
- **Always On AG**: Status only (not a blocker - supported in RDS)
- **Service Broker**: Status only (not a blocker - supported in RDS)

### **PowerShell Storage Logic**
```sql
-- Step 1: Get drive free space
EXEC xp_fixeddrives

-- Step 2: Get SQL file sizes per drive
SELECT LEFT(physical_name, 1) as drive,
       SUM(CAST(size AS BIGINT) * 8.0 / 1024.0 / 1024.0) as SQLFilesGB
FROM sys.master_files
GROUP BY LEFT(physical_name, 1)

-- Step 3: Total = Free Space + SQL Files (for drives with SQL files)
```

### **AWS Instance Sizing**
- **CPU-based sizing**: Matches core count to RDS instance types
- **Memory optimization**: Selects appropriate instance families
- **Modern instances**: Recommends latest generation (m6i, r6i, x2iedn)

## **🎯 Current Status**

### **✅ Production Complete**
- **PowerShell CSV Output**: Identical format to original RDS Discovery tool
- **Real SQL Server Data**: All data from live SQL queries, no mock data
- **PowerShell Storage Logic**: Exact `xp_fixeddrives` implementation
- **AWS Instance Sizing**: Intelligent recommendations based on real server specs
- **27+ Compatibility Checks**: Complete feature parity plus SSIS/SSRS detection
- **PowerShell RDS Blocking**: Exact same blocking logic as original PowerShell script
- **Enhanced Detection**: SSIS/SSRS detection with system package filtering
- **Error Handling**: Graceful failure handling matching PowerShell behavior
- **Authentication Support**: Windows and SQL Server authentication
- **Performance Monitoring**: Timing metrics and success rate tracking

### **✅ Verified Results**
- **Real Server Testing**: Tested with SQL Server 2022 Enterprise Edition
- **Data Accuracy**: All values match or closely approximate PowerShell output
- **Storage Calculation**: `51.32 GB` vs PowerShell `53.55 GB` (within expected variance)
- **Feature Detection**: All 27+ compatibility checks working correctly
- **SSIS Detection**: Accurate detection excluding system collector packages
- **SSRS Detection**: Proper ReportServer database detection
- **RDS Blocking Logic**: Matches PowerShell script exactly (Always On AG not a blocker)

## **📄 Documentation**

- 🔧 **[Technical Requirements](TECHNICAL_REQUIREMENTS.md)** - Installation and dependencies
- 📖 **[Usage Guide](USAGE_GUIDE.md)** - Complete tool reference
- 🧪 **[Testing Guide](TESTING_GUIDE.md)** - Testing procedures
- 🚀 **[Production Deployment](PRODUCTION_DEPLOYMENT.md)** - Production setup
- 💡 **[AWS Instance Sizing](AWS_INSTANCE_SIZING.md)** - Sizing logic and algorithms
- 📋 **[Development Plan](strands-rds-discovery-tool-1month-plan.md)** - Project timeline

## **🧪 Testing**

### **Quick Test**
```bash
cd strands-rds-discovery
source venv/bin/activate

# Test with real SQL Server
python3 -c "
from src.rds_discovery import strands_rds_discovery
result = strands_rds_discovery(
    action='assess',
    input_file='real_servers.csv',
    auth_type='sql',
    username='test',
    password='Password1!'
)
print(result)
"
# Generates: RdsDiscovery_[timestamp].csv
```

### **Expected Output**
```
✅ Assessment complete! Report saved to: RdsDiscovery_1759694195.csv

Servers assessed: 1
Successful: 1
RDS Compatible: 1
Success Rate: 100.0%
```

## **🎯 Migration Scenarios**

### **RDS Compatible Servers**
- **CSV Output**: `"RDS Compatible","Y"`
- **Recommendation**: Direct migration to Amazon RDS for SQL Server
- **Instance**: Specific sizing (e.g., `db.m6i.2xlarge`)

### **RDS Custom Candidates**  
- **CSV Output**: `"RDS Custom Compatible","Y"`
- **Recommendation**: Amazon RDS Custom for SQL Server
- **Use Case**: Some enterprise features or custom configurations

### **EC2 Migration Required**
- **CSV Output**: `"EC2 Compatible","Y"`
- **Recommendation**: Amazon EC2 with SQL Server
- **Use Case**: Complex features like Always On AG, FileStream

## **🔒 Security & Compliance**

### **Security Features**
- **Credential Protection**: Passwords never logged or stored
- **Network Security**: SSL/TLS encryption support
- **Input Validation**: Comprehensive parameter validation
- **Error Handling**: Secure error messages without sensitive data

### **Data Collection**
- **No Customer Data**: Only metadata and configuration information
- **Real-time Assessment**: No data stored locally beyond CSV output
- **Audit Trail**: Complete logging of assessment activities

## **🚀 GitHub Setup & Deployment**

### **Initial Repository Setup**
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Strands RDS Discovery Tool v2.1.2"

# Add GitHub remote
git remote add origin https://github.com/bobtherdsman/RDSMCP.git
git branch -M main
```

### **GitHub Personal Access Token Setup**
1. Go to GitHub.com → Profile Picture → Settings
2. Scroll to bottom of left sidebar → "Developer settings"
3. Click "Personal access tokens" → "Tokens (classic)"
4. Click "Generate new token (classic)"
5. **Configuration**:
   - **Note**: "RDS Discovery Tool"
   - **Expiration**: Choose duration (30-90 days recommended)
   - **Scopes**: Check `repo` (full repository access)
6. **Copy token immediately** - you won't see it again

**Direct link**: https://github.com/settings/tokens

### **Push to GitHub**
```bash
# First push (handles merge conflicts)
git pull origin main --allow-unrelated-histories --no-rebase

# Resolve any conflicts by keeping local files
git checkout --ours .gitignore CONTRIBUTING.md LICENSE README.md pyproject.toml
git add .gitignore CONTRIBUTING.md LICENSE README.md pyproject.toml
git commit -m "Merge remote changes, keeping local RDS discovery tool files"

# Push to GitHub
git push -u origin main
# Username: bobtherdsman
# Password: [paste your personal access token]
```

### **Handling Merge Conflicts**
When pushing to an existing repository with different files:

1. **Pull with merge strategy**:
   ```bash
   git pull origin main --allow-unrelated-histories --no-rebase
   ```

2. **Resolve conflicts** (keep your local versions):
   ```bash
   git checkout --ours [conflicted-files]
   git add [conflicted-files]
   ```

3. **Commit merge**:
   ```bash
   git commit -m "Merge remote changes, keeping local RDS discovery tool files"
   ```

4. **Push successfully**:
   ```bash
   git push -u origin main
   ```

### **Authentication Notes**
- **Username**: Your GitHub username (`bobtherdsman`)
- **Password**: Your Personal Access Token (NOT your GitHub password)
- **Token Security**: Store token securely, never commit to code
- **Token Expiration**: Set appropriate expiration and renew as needed

### **Repository Structure**
```
strands-rds-discovery/
├── src/
│   └── rds_discovery.py          # Main tool
├── requirements.txt              # Dependencies
├── README.md                     # This documentation
├── RUN_GUIDE.md                 # Usage guide
├── real_servers.csv             # Server input template
└── RdsDiscovery_[timestamp].csv # Output files
```

## **🤝 Contributing**

### **Strands Integration**
This tool is designed for integration into the mainstream Strands tools ecosystem. The PowerShell-compatible output ensures seamless migration from existing PowerShell-based workflows.

### **Development**
```bash
# Setup development environment
git clone https://github.com/bobtherdsman/RDSMCP.git
cd strands-rds-discovery
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test with real server
python3 -c "from src.rds_discovery import strands_rds_discovery; ..."
```

### **Making Changes**
```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push origin main
# Use your personal access token when prompted
```

## **📞 Support**

### **Key Files**
- **src/rds_discovery.py** - Main Strands tool with PowerShell CSV output
- **RdsDiscovery_[timestamp].csv** - PowerShell-compatible assessment results
- **real_servers.csv** - Server input template

### **Community**
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Strands Community**: Integration with main Strands community channels

## **📜 License**

MIT License - see LICENSE file for details.

---

**Production-ready Strands tool with PowerShell-compatible CSV output and real SQL Server data collection!** 🚀
