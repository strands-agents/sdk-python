# Strands RDS Discovery Tool - Technical Requirements & Installation

**Last Updated**: 2025-10-03  
**Version**: 2.0  
**Status**: ✅ Ready for Strands Integration  

## **System Requirements**

### **Operating System**
- ✅ **Ubuntu 24.04 LTS** (Verified)
- ✅ **Linux** (General compatibility)
- ⚠️ **Windows/macOS** (Not tested, should work with modifications)

### **Python Environment**
- **Python Version**: 3.12+ (Verified with 3.12.3)
- **Virtual Environment**: Required (externally-managed-environment protection)
- **Package Manager**: pip 24.0+

### **System Dependencies**
```bash
# Required system packages
sudo apt update
sudo apt install -y python3-pip python3.12-venv python3-dev
sudo apt install -y unixodbc unixodbc-dev
sudo apt install -y build-essential  # For package compilation if needed
```

## **Installation Guide**

### **Step 1: Create Project Environment**
```bash
# Create project directory
mkdir strands-rds-discovery
cd strands-rds-discovery

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### **Step 2: Install Python Dependencies**
```bash
# Install core packages
pip install -r requirements.txt
```

**requirements.txt**:
```
strands-agents>=1.10.0
strands-agents-tools>=0.2.9
pyodbc>=5.2.0
sqlalchemy>=2.0.43
boto3>=1.40.41
pandas>=2.3.2
```

### **Step 3: Verify Installation**
```bash
# Test Strands framework
python3 -c "from strands import tool; print('✅ Strands working')"

# Test SQL connectivity framework
python3 -c "import pyodbc; print('✅ ODBC working')"

# Test ODBC driver availability
python3 -c "import pyodbc; print('Available drivers:', pyodbc.drivers())"

# Test AWS integration
python3 -c "import boto3; print('✅ AWS SDK working')"
```

**Expected Output**:
```
✅ Strands working
✅ ODBC working
Available drivers: ['ODBC Driver 18 for SQL Server']
✅ AWS SDK working
```

## **SQL Server Connectivity**

### **ODBC Driver Requirements**
**Status**: ✅ **COMPLETE** - Microsoft ODBC Driver 18 for SQL Server installed and verified

#### **✅ Installed Solution: Microsoft ODBC Driver 18**
```bash
# Add Microsoft repository
curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/24.04/prod noble main" | sudo tee /etc/apt/sources.list.d/msprod.list

# Install ODBC driver
sudo apt update
sudo ACCEPT_EULA=Y apt install msodbcsql18
```

**Verification**: ✅ Driver available as "ODBC Driver 18 for SQL Server"

#### **Option 2: FreeTDS (Alternative)**
```bash
sudo apt install tdsodbc
```

### **Connection String Formats**
```python
# Windows Authentication
"DRIVER={ODBC Driver 18 for SQL Server};SERVER=server_name;Trusted_Connection=yes;"

# SQL Server Authentication  
"DRIVER={ODBC Driver 18 for SQL Server};SERVER=server_name;UID=username;PWD=password;"
```

## **Package Dependencies**

### **Core Strands Framework**
| Package | Version | Purpose |
|---------|---------|---------|
| strands-agents | 1.10.0+ | Core Strands framework with @tool decorators |
| strands-agents-tools | 0.2.9+ | Additional Strands tools and utilities |

### **Database Connectivity**
| Package | Version | Purpose |
|---------|---------|---------|
| pyodbc | 5.2.0+ | SQL Server ODBC connectivity |
| sqlalchemy | 2.0.43+ | Database abstraction layer |

### **AWS Integration**
| Package | Version | Purpose |
|---------|---------|---------|
| boto3 | 1.40.41+ | AWS SDK for Python |
| botocore | 1.40.41+ | AWS core functionality |

### **Data Processing**
| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.2+ | Data analysis and CSV handling |
| numpy | 2.3.3+ | Numerical computing (pandas dependency) |

### **Additional Dependencies**
Automatically installed with main packages:
- pydantic (2.11.9+) - Data validation
- typing-extensions (4.15.0+) - Type hints
- mcp (1.15.0+) - Model Context Protocol
- opentelemetry-* - Observability framework

## **Known Issues & Solutions**

### **Issue 1: Generic `strands` Package Build Failure**
**Problem**: 
```
CMake Error: add_subdirectory given source "lib/matslise" which is not an existing directory
ERROR: Failed building wheel for strands
```

**Solution**: ✅ **RESOLVED**
```bash
# Don't use generic 'strands' package
pip install strands  # ❌ FAILS

# Use specific packages instead
pip install strands-agents strands-agents-tools  # ✅ WORKS
```

### **Issue 2: Externally Managed Environment**
**Problem**:
```
error: externally-managed-environment
× This environment is externally managed
```

**Solution**: ✅ **RESOLVED**
```bash
# Always use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install packages  # Now works
```

### **Issue 3: Missing ODBC Drivers**
**Problem**:
```
ImportError: libodbc.so.2: cannot open shared object file
```

**Solution**: ✅ **RESOLVED**
```bash
# Install ODBC libraries
sudo apt install unixodbc unixodbc-dev

# Install Microsoft SQL Server ODBC Driver 18
sudo ACCEPT_EULA=Y apt install msodbcsql18

# Verify installation
odbcinst -q -d  # Should show "ODBC Driver 18 for SQL Server"
```

## **Development Environment Setup**

### **Project Structure**
```
strands-rds-discovery/
├── venv/                          # Virtual environment (don't commit)
├── src/
│   ├── __init__.py               # Package initialization
│   ├── rds_discovery.py          # Main Strands tools
│   └── sql_queries.py            # SQL Server queries
├── tests/
│   └── test_rds_discovery.py     # Test suite
├── config/                       # Configuration files
├── examples/                     # Usage examples
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
├── TECHNICAL_REQUIREMENTS.md     # This document
└── strands-rds-discovery-tool-1month-plan.md  # Development plan
```

### **Development Workflow**
```bash
# Daily development routine
cd strands-rds-discovery
source venv/bin/activate

# Run tests
python3 test_basic.py

# Test single consolidated Strands tool
python3 src/rds_discovery.py

# Test individual actions in Python
python3 -c "
from src.rds_discovery import strands_rds_discovery
# Create template
strands_rds_discovery(action='template', output_file='servers.csv')
# Assess servers
result = strands_rds_discovery(action='assess', input_file='servers.csv', auth_type='windows')
# Get explanations
explanation = strands_rds_discovery(action='explain', assessment_data=result)
"

# Deactivate when done
deactivate
```

## **Testing & Verification**

### **Basic Functionality Test**
```bash
cd strands-rds-discovery
source venv/bin/activate
python3 test_basic.py
```

**Expected Output**:
```
🚀 Starting RDS Discovery Tool Tests
🧪 Testing JSON structure...
✅ JSON structure test passed!
🧪 Testing connection failure handling...
✅ Connection failure test passed!
🧪 Testing migration blocker explanations...
✅ Explanation function test passed!
🧪 Testing migration recommendations...
✅ Recommendation function test passed!
🎉 All tests passed!
```

### **Strands Integration Test**
```bash
python3 src/rds_discovery.py
```

**Expected Output**:
```
🧪 Testing Consolidated Strands RDS Discovery Tool

1. Testing template creation...
✅ Template creation works

2. Testing assessment...
Assessing server 1/1: test-server.example.com
✅ Assessment works

3. Testing explanations...
✅ Explanations work

4. Testing recommendations...
✅ Recommendations work

🎉 Consolidated Strands RDS Discovery Tool is working!

📋 Usage:
  • Template: strands_rds_discovery(action='template', output_file='servers.csv')
  • Assess: strands_rds_discovery(action='assess', input_file='servers.csv', auth_type='windows')
  • Explain: strands_rds_discovery(action='explain', assessment_data=result)
  • Recommend: strands_rds_discovery(action='recommend', assessment_data=result)
```

## **Performance & Scalability**

### **Resource Requirements**
- **Memory**: 512MB minimum, 2GB recommended
- **CPU**: 1 core minimum, 2+ cores for batch processing
- **Storage**: 100MB for installation, additional for logs/reports
- **Network**: Internet access for AWS API calls

### **Scalability Considerations**
- **Single Server Assessment**: ~5-30 seconds depending on SQL Server response
- **Batch Processing**: Parallel assessment capability (future enhancement)
- **Memory Usage**: ~50MB per assessment, scales linearly

## **Security Considerations**

### **Credential Management**
- ✅ No hardcoded credentials in code
- ✅ Connection strings built dynamically
- ⚠️ Implement secure credential storage (future enhancement)

### **Network Security**
- SQL Server connections use standard TDS protocol
- AWS API calls use HTTPS with IAM authentication
- No sensitive data stored locally

## **Troubleshooting**

### **Common Commands**
```bash
# Check Python version
python3 --version

# Verify virtual environment
which python3  # Should show venv path when activated

# List installed packages
pip list

# Check package versions
pip show strands-agents

# Reinstall if needed
pip uninstall strands-agents strands-agents-tools
pip install strands-agents strands-agents-tools
```

### **Log Locations**
- **Application logs**: Console output (no file logging yet)
- **pip logs**: `~/.cache/pip/log/`
- **System logs**: `/var/log/` (for system package issues)

## **Future Enhancements**

### **Planned Improvements**
- [ ] Complete SQL Server ODBC driver installation
- [ ] Add configuration file support
- [ ] Implement secure credential management
- [ ] Add logging framework
- [ ] Create Docker containerization
- [ ] Add performance monitoring

### **Version Compatibility**
- **Current**: Development version 1.0
- **Target**: Production version 1.0 (Week 4)
- **Backward Compatibility**: Maintained for all 1.x versions

---

**Document Maintenance**: This document should be updated whenever:
- New dependencies are added
- Installation procedures change
- Known issues are resolved
- New issues are discovered
- System requirements change

**Contact**: Update development team when issues arise or improvements are needed.
