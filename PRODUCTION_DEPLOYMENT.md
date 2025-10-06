# Production Deployment Guide

**Version**: 2.0  
**Date**: 2025-10-03  
**Purpose**: Production deployment for Strands integration

## **Production Readiness Checklist**

### **✅ Core Functionality**
- ✅ Single consolidated Strands tool (`strands_rds_discovery`)
- ✅ Real SQL Server connectivity verified
- ✅ All 25 compatibility queries tested
- ✅ Batch processing with file input
- ✅ Multiple authentication methods (Windows/SQL)
- ✅ Comprehensive error handling
- ✅ Production logging and monitoring

### **✅ Security**
- ✅ Password escaping for special characters
- ✅ SSL certificate handling (`TrustServerCertificate`)
- ✅ Input validation and sanitization
- ✅ Connection timeout controls
- ✅ No credentials stored in logs

### **✅ Performance**
- ✅ Configurable connection timeouts (5-300 seconds)
- ✅ Query timeout controls
- ✅ Progress tracking for batch operations
- ✅ Performance metrics collection
- ✅ Memory-efficient CSV processing

### **✅ Reliability**
- ✅ Comprehensive error handling
- ✅ Graceful failure handling
- ✅ Individual server failure isolation
- ✅ Retry logic for transient errors
- ✅ Detailed logging and diagnostics

## **System Requirements**

### **Minimum Requirements**
- **OS**: Ubuntu 20.04+ or compatible Linux distribution
- **Python**: 3.8+
- **Memory**: 512 MB RAM
- **Storage**: 100 MB free space
- **Network**: Outbound connectivity to SQL Servers (port 1433)

### **Recommended Requirements**
- **OS**: Ubuntu 22.04 LTS or Ubuntu 24.04 LTS
- **Python**: 3.12+
- **Memory**: 2 GB RAM (for large server lists)
- **Storage**: 1 GB free space (for logs and results)
- **CPU**: 2+ cores (for concurrent assessments)

## **Installation**

### **1. System Dependencies**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install ODBC components
sudo apt install unixodbc unixodbc-dev -y

# Install Microsoft ODBC Driver 18
curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/$(lsb_release -rs)/prod $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/msprod.list
sudo apt update
sudo ACCEPT_EULA=Y apt install msodbcsql18 -y
```

### **2. Application Setup**
```bash
# Create application directory
sudo mkdir -p /opt/strands-rds-discovery
sudo chown $USER:$USER /opt/strands-rds-discovery
cd /opt/strands-rds-discovery

# Clone or copy application files
# (Copy your strands-rds-discovery directory here)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python3 verify_installation.py
```

### **3. Configuration**
```bash
# Create configuration directory
mkdir -p config logs results

# Set appropriate permissions
chmod 755 config logs results
chmod 644 src/*.py
chmod +x verify_installation.py
```

## **Configuration**

### **Environment Variables**
```bash
# Optional: Set default configuration
export RDS_DISCOVERY_LOG_LEVEL=INFO
export RDS_DISCOVERY_DEFAULT_TIMEOUT=30
export RDS_DISCOVERY_MAX_SERVERS=100
```

### **Logging Configuration**
The tool automatically creates logs in:
- **File**: `rds_discovery.log` (in current directory)
- **Console**: Standard output with timestamps
- **Level**: INFO (configurable via environment)

### **Security Configuration**
```bash
# Secure the application directory
sudo chown -R rds-user:rds-group /opt/strands-rds-discovery
chmod 750 /opt/strands-rds-discovery
chmod 640 /opt/strands-rds-discovery/src/*.py

# Secure log files
chmod 640 /opt/strands-rds-discovery/logs/*
```

## **Usage in Production**

### **1. Create Server Inventory**
```python
from src.rds_discovery import strands_rds_discovery

# Create template
result = strands_rds_discovery(
    action="template",
    output_file="/opt/strands-rds-discovery/config/production_servers.csv"
)
```

### **2. Run Production Assessment**
```python
# Assess with Windows authentication
result = strands_rds_discovery(
    action="assess",
    input_file="/opt/strands-rds-discovery/config/production_servers.csv",
    auth_type="windows",
    timeout=60,
    output_file="/opt/strands-rds-discovery/results/assessment_results.json"
)

# Assess with SQL authentication
result = strands_rds_discovery(
    action="assess",
    input_file="/opt/strands-rds-discovery/config/production_servers.csv",
    auth_type="sql",
    username="assessment_user",
    password="SecurePassword123!",
    timeout=60,
    output_file="/opt/strands-rds-discovery/results/assessment_results.json"
)
```

### **3. Generate Reports**
```python
# Load assessment results
with open("/opt/strands-rds-discovery/results/assessment_results.json", "r") as f:
    assessment_data = f.read()

# Generate explanations
explanation = strands_rds_discovery(
    action="explain",
    assessment_data=assessment_data
)

# Generate recommendations
recommendations = strands_rds_discovery(
    action="recommend",
    assessment_data=assessment_data
)

# Save reports
with open("/opt/strands-rds-discovery/results/explanation_report.txt", "w") as f:
    f.write(explanation)

with open("/opt/strands-rds-discovery/results/recommendations_report.txt", "w") as f:
    f.write(recommendations)
```

## **Monitoring and Maintenance**

### **Log Monitoring**
```bash
# Monitor real-time logs
tail -f /opt/strands-rds-discovery/rds_discovery.log

# Check for errors
grep "ERROR" /opt/strands-rds-discovery/rds_discovery.log

# Monitor performance
grep "Assessment completed" /opt/strands-rds-discovery/rds_discovery.log
```

### **Health Checks**
```bash
# Verify installation
cd /opt/strands-rds-discovery
source venv/bin/activate
python3 verify_installation.py

# Test connectivity
python3 -c "
from src.rds_discovery import strands_rds_discovery
result = strands_rds_discovery(action='template', output_file='health_check.csv')
print('Health check passed' if 'success' in result else 'Health check failed')
"
```

### **Performance Tuning**
```python
# For large server lists (100+ servers)
result = strands_rds_discovery(
    action="assess",
    input_file="large_server_list.csv",
    auth_type="windows",
    timeout=120,  # Increase timeout for slow networks
    output_file="large_assessment_results.json"
)
```

## **Backup and Recovery**

### **Backup Strategy**
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/backup/strands-rds-discovery"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup configuration and results
cp -r /opt/strands-rds-discovery/config $BACKUP_DIR/$DATE/
cp -r /opt/strands-rds-discovery/results $BACKUP_DIR/$DATE/
cp /opt/strands-rds-discovery/rds_discovery.log $BACKUP_DIR/$DATE/

# Compress backup
tar -czf $BACKUP_DIR/strands-rds-discovery-$DATE.tar.gz -C $BACKUP_DIR $DATE
rm -rf $BACKUP_DIR/$DATE

# Keep last 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### **Recovery Procedure**
```bash
# Restore from backup
DATE=20251002  # Replace with backup date
BACKUP_DIR="/backup/strands-rds-discovery"

# Extract backup
tar -xzf $BACKUP_DIR/strands-rds-discovery-$DATE.tar.gz -C /tmp

# Restore files
cp -r /tmp/$DATE/config/* /opt/strands-rds-discovery/config/
cp -r /tmp/$DATE/results/* /opt/strands-rds-discovery/results/

# Verify restoration
cd /opt/strands-rds-discovery
python3 verify_installation.py
```

## **Troubleshooting**

### **Common Issues**

**1. ODBC Driver Not Found**
```bash
# Verify driver installation
odbcinst -q -d

# Expected output: [ODBC Driver 18 for SQL Server]
# If missing, reinstall ODBC driver
```

**2. Connection Timeouts**
```python
# Increase timeout for slow networks
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="windows",
    timeout=120  # Increase from default 30 seconds
)
```

**3. Permission Errors**
```bash
# Check file permissions
ls -la /opt/strands-rds-discovery/
chmod 755 /opt/strands-rds-discovery/
chmod 644 /opt/strands-rds-discovery/src/*.py
```

**4. Memory Issues with Large Server Lists**
```python
# Process servers in smaller batches
# Split large CSV files into chunks of 50-100 servers
```

### **Support Information**
- **Log Location**: `/opt/strands-rds-discovery/rds_discovery.log`
- **Configuration**: `/opt/strands-rds-discovery/config/`
- **Results**: `/opt/strands-rds-discovery/results/`
- **Version**: Check `strands_rds_discovery` output for version info

## **Security Best Practices**

### **Credential Management**
- **Never store passwords in files**
- Use environment variables or secure credential stores
- Rotate assessment account passwords regularly
- Use dedicated service accounts with minimal privileges

### **Network Security**
- Ensure SQL Server ports (1433) are properly firewalled
- Use VPN or private networks when possible
- Monitor network traffic for assessment activities

### **Access Control**
- Limit access to assessment results
- Use role-based access for the tool
- Audit assessment activities regularly

### **Data Protection**
- Encrypt assessment results at rest
- Secure transmission of results
- Implement data retention policies

---

**Production deployment complete! The tool is ready for enterprise use with comprehensive monitoring, security, and operational procedures.**
