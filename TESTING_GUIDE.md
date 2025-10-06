# Testing Guide - Strands RDS Discovery Tool

**Version**: 2.1.2  
**Date**: 2025-10-06  
**Purpose**: Comprehensive testing procedures for Strands tool integration

## **Strands Tool Testing**

### **Test Strands Integration**
```python
# Test @tool decorator functionality
from src.rds_discovery import strands_rds_discovery

# Verify tool is properly decorated
assert hasattr(strands_rds_discovery, '__tool__'), "Tool decorator not found"

# Test basic Strands tool call
result = strands_rds_discovery(
    input_file='test_servers.csv',
    auth_type='windows'
)
print("✅ Strands tool integration working")
```

### **Test Natural Language Integration**
```python
# Test within Strands AI context (if available)
# "Assess SQL Server for RDS migration"
# "Generate RDS discovery report"
# "What AWS instance is recommended?"
```

## **Quick Testing**

### **Basic Functionality Test**
```bash
# Test core functionality with your servers (2 minutes)
python3 test.py --input-file my_servers.csv --auth-type windows --mode basic
```

### **Comprehensive Test Suite**
```bash
# Full test suite with error handling, timeouts, performance (5-10 minutes)
python3 test.py --input-file my_servers.csv --auth-type windows --mode full
```

### **SQL Server Authentication**
```bash
# Test with SQL Server authentication
python3 test.py --input-file my_servers.csv --auth-type sql --username sa --password MyPass123 --mode basic
```

### **Custom Timeout**
```bash
# Test with custom timeout
python3 test.py --input-file my_servers.csv --auth-type windows --timeout 60 --mode full
```

## **Test Modes**

### **Basic Mode** (`--mode basic`)
**Duration**: 1-3 minutes  
**Tests**:
1. Template creation
2. Assessment with your servers
3. AI explanations generation
4. Migration recommendations

**Use when**: Quick validation that tool works with your servers

### **Full Mode** (`--mode full`)
**Duration**: 5-15 minutes  
**Tests**:
1. All basic tests
2. Error handling scenarios
3. Timeout testing (10s, 30s, 60s)
4. Performance monitoring
5. File operations

**Use when**: Comprehensive validation before production use

## **Test Setup**

### **1. Create Server List**
```bash
# Generate template
python3 -c "from src.rds_discovery import strands_rds_discovery; strands_rds_discovery(action='template', output_file='my_servers.csv')"

# Edit my_servers.csv with your servers:
# server_name
# prod-sql01.company.com
# dev-sql02.company.com
```

### **2. Run Tests**
```bash
# Basic test
python3 test.py --input-file my_servers.csv --auth-type windows --mode basic

# Full test
python3 test.py --input-file my_servers.csv --auth-type windows --mode full
```

## **Test Scenarios Explained**

### **Basic Mode Tests**

**1. Template Creation**
- Verifies CSV template generation
- Tests file creation permissions
- Validates template format

**2. Server Assessment**
- Connects to your SQL Servers
- Runs compatibility analysis
- Tests authentication methods
- Measures performance

**3. AI Functions**
- Tests explanation generation
- Tests recommendation generation
- Validates natural language output

### **Full Mode Additional Tests**

**4. Error Handling**
- Invalid action parameters
- Missing input files
- Authentication failures
- SQL Server connection errors

**5. Timeout Scenarios**
- Tests 10-second timeout
- Tests 30-second timeout  
- Tests 60-second timeout
- Validates timeout behavior

**6. Performance Monitoring**
- Measures execution time
- Tracks success rates
- Monitors resource usage
- Validates performance metrics

**7. File Operations**
- Tests output file creation
- Validates file permissions
- Tests different output locations

## **Expected Results**

### **Successful Basic Test Output**
```
🧪 BASIC FUNCTIONALITY TESTS
==================================================

1. Testing template creation...
✅ Template creation: PASS

2. Testing assessment with your servers...
   File: my_servers.csv
   Auth: windows
✅ Assessment: PASS
   Total servers: 3
   Success rate: 100.0%
   RDS compatible: 2
   Total time: 45.2s

3. Testing explanations...
✅ Explanations: PASS (1,234 characters)

4. Testing recommendations...
✅ Recommendations: PASS (2,567 characters)

📊 TEST SUMMARY
==============================
Total tests: 4
Passed: 4
Failed: 0
Success rate: 100.0%

🎉 All tests passed!
```

### **Test Results Files**
- `test_results/basic_assessment.json` - Assessment results
- `test_results/test_results.json` - Detailed test results
- `test_results/test_template.csv` - Generated template
- `test_results/timeout_*.json` - Timeout test results (full mode)

## **Troubleshooting Tests**

### **Common Test Issues**

**1. "Input file not found"**
```bash
# Create server list first
python3 -c "from src.rds_discovery import strands_rds_discovery; strands_rds_discovery(action='template', output_file='servers.csv')"
# Edit servers.csv with your server names
```

**2. "Username and password required for SQL Server authentication"**
```bash
# Add credentials for SQL auth
python3 test.py --input-file servers.csv --auth-type sql --username sa --password MyPass123 --mode basic
```

**3. "Connection timeout" or "Connection failed"**
- Verify server names are correct
- Check network connectivity
- Confirm SQL Server allows remote connections
- Verify firewall settings (port 1433)
- Try increasing timeout: `--timeout 120`

**4. "No module named 'strands'"**
```bash
# Activate virtual environment
source venv/bin/activate
python3 test.py --input-file servers.csv --auth-type windows --mode basic
```

### **Debug Mode**
```bash
# Enable detailed logging
export PYTHONPATH=/home/bacrifai/strands-rds-discovery/src
python3 test.py --input-file servers.csv --auth-type windows --mode full
```

## **Test Data Management**

### **Test File Organization**
```
strands-rds-discovery/
├── test_results/              # Test output files
│   ├── basic_assessment.json  # Basic test results
│   ├── test_results.json      # Detailed test summary
│   ├── timeout_10s.json       # Timeout test results
│   └── performance_test.json  # Performance test results
├── test.py                    # Consolidated test runner
├── my_servers.csv             # Your server list
└── rds_discovery.log          # Execution logs
```

### **Cleanup After Testing**
```bash
# Clean up test files
rm -rf test_results/
rm -f test_template.csv my_servers.csv
```

## **Automated Testing**

### **CI/CD Integration**
```bash
#!/bin/bash
# CI test script

# Basic functionality test
python3 test.py --input-file ci_servers.csv --auth-type windows --mode basic

# Check exit codes
if [ $? -eq 0 ]; then
    echo "✅ All tests passed"
    exit 0
else
    echo "❌ Tests failed"
    exit 1
fi
```

### **Regression Testing**
```bash
# Run before releases
python3 test.py --input-file regression_servers.csv --auth-type windows --mode full > test_results.log 2>&1

# Compare with baseline
diff baseline_results.json test_results/test_results.json
```

## **Performance Benchmarks**

### **Expected Performance**
- **Template creation**: < 1 second
- **Single server assessment**: 2-10 seconds (depending on network)
- **Basic mode (3 servers)**: 30-90 seconds
- **Full mode (3 servers)**: 2-5 minutes

### **Performance Monitoring**
The test runner automatically collects:
- Total execution time
- Average time per server
- Success rates
- Timeout compliance
- Resource usage

## **Test Validation**

### **What Constitutes a Passing Test**

**Basic Mode**: All 4 tests must pass
- Template creation successful
- At least one server assessed successfully
- Explanations generated (>50 characters)
- Recommendations generated (>100 characters)

**Full Mode**: All tests must pass including
- Error handling works correctly
- Timeouts respected
- Performance metrics collected
- File operations successful

### **Acceptable Failure Scenarios**
- Individual server connection failures (network issues)
- Timeout on unreachable servers
- Authentication failures with wrong credentials

### **Unacceptable Failures**
- Tool crashes or exceptions
- Template creation failures
- Complete inability to connect to any server
- Missing performance metrics

## **Test Reporting**

### **Test Results Format**
```json
{
  "test": "assessment",
  "status": "PASS",
  "servers": 3,
  "success_rate": 66.7,
  "rds_compatible": 2,
  "time": 45.2
}
```

### **Generate Test Report**
```bash
# Run tests and view results
python3 test.py --input-file servers.csv --auth-type windows --mode full
cat test_results/test_results.json | python3 -m json.tool
```

---

**Use this consolidated testing approach to validate the RDS Discovery Tool with your specific SQL Server environment.**
