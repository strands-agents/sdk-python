#!/usr/bin/env python3
"""
Installation Verification Script
Verify all components are properly installed and working
"""

def test_strands():
    """Test Strands framework"""
    try:
        from strands import tool
        print("✅ Strands framework: Working")
        return True
    except ImportError as e:
        print(f"❌ Strands framework: Failed - {e}")
        return False

def test_pyodbc():
    """Test pyodbc and ODBC drivers"""
    try:
        import pyodbc
        drivers = pyodbc.drivers()
        print("✅ pyodbc: Working")
        
        sql_drivers = [d for d in drivers if 'SQL Server' in d]
        if sql_drivers:
            print(f"✅ SQL Server ODBC Driver: {sql_drivers[0]}")
            return True
        else:
            print("❌ SQL Server ODBC Driver: Not found")
            return False
    except ImportError as e:
        print(f"❌ pyodbc: Failed - {e}")
        return False

def test_aws_sdk():
    """Test AWS SDK"""
    try:
        import boto3
        print("✅ AWS SDK (boto3): Working")
        return True
    except ImportError as e:
        print(f"❌ AWS SDK (boto3): Failed - {e}")
        return False

def test_data_processing():
    """Test data processing libraries"""
    try:
        import pandas
        import sqlalchemy
        print("✅ Data processing (pandas, sqlalchemy): Working")
        return True
    except ImportError as e:
        print(f"❌ Data processing libraries: Failed - {e}")
        return False

def test_rds_tools():
    """Test RDS Discovery tools"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from rds_discovery import strands_rds_discovery
        from sql_queries import FEATURE_CHECKS
        
        print(f"✅ RDS Discovery tools: Working (1 consolidated Strands tool, {len(FEATURE_CHECKS)} compatibility checks)")
        return True
    except ImportError as e:
        print(f"❌ RDS Discovery tools: Failed - {e}")
        return False

def main():
    """Run all verification tests"""
    print("🚀 Verifying Strands RDS Discovery Tool Installation\n")
    
    tests = [
        test_strands,
        test_pyodbc, 
        test_aws_sdk,
        test_data_processing,
        test_rds_tools
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components verified! Ready for SQL Server connectivity testing.")
        print("\n📋 Available Strands Tool:")
        print("  • strands_rds_discovery() - Single tool with 4 actions:")
        print("    - action='template' - Create server list CSV")
        print("    - action='assess' - Assess SQL Servers from file")
        print("    - action='explain' - Explain migration blockers")
        print("    - action='recommend' - Get migration recommendations")
        return 0
    else:
        print("❌ Some components failed verification. Check installation.")
        return 1

if __name__ == "__main__":
    exit(main())
