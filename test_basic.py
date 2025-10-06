#!/usr/bin/env python3
"""
Basic test script for RDS Discovery Tool
Tests the functionality without requiring actual SQL Server connection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rds_discovery import assess_sql_server, explain_migration_blockers, recommend_migration_path
import json

def test_connection_failure():
    """Test how the tool handles connection failures"""
    print("🧪 Testing connection failure handling...")
    
    result = assess_sql_server("nonexistent-server.example.com", "windows")
    data = json.loads(result)
    
    print(f"Status: {data['status']}")
    print(f"Connection: {data['connection']}")
    print(f"Error present: {'error' in data}")
    
    assert data['status'] == 'error'
    assert data['connection'] == 'failed'
    print("✅ Connection failure test passed!\n")

def test_explanation_function():
    """Test the migration blocker explanation function"""
    print("🧪 Testing migration blocker explanations...")
    
    # Mock assessment result with blocking features
    mock_assessment = {
        "server": "test-server",
        "status": "success",
        "rds_compatible": "N",
        "blocking_features": ["filestream", "linked_servers"]
    }
    
    explanation = explain_migration_blockers(json.dumps(mock_assessment))
    print("Explanation result:")
    print(explanation)
    
    assert "FileStream" in explanation
    assert "Linked servers" in explanation
    print("✅ Explanation function test passed!\n")

def test_recommendation_function():
    """Test the migration recommendation function"""
    print("🧪 Testing migration recommendations...")
    
    # Mock assessment result for RDS-compatible server
    mock_assessment = {
        "server": "test-server",
        "status": "success",
        "rds_compatible": "Y",
        "server_info": {"edition": "Standard Edition"},
        "resources": {"cpu_count": 4, "max_memory_mb": 8192},
        "database_size_gb": 50.5
    }
    
    recommendations = recommend_migration_path(json.dumps(mock_assessment))
    print("Recommendation result:")
    print(recommendations)
    
    assert "Amazon RDS for SQL Server" in recommendations
    assert "db.r5" in recommendations
    print("✅ Recommendation function test passed!\n")

def test_json_structure():
    """Test that our JSON structure is valid"""
    print("🧪 Testing JSON structure...")
    
    result = assess_sql_server("test-server", "windows")
    data = json.loads(result)  # This will fail if JSON is invalid
    
    required_fields = ['server', 'status', 'connection', 'timestamp']
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    print("✅ JSON structure test passed!\n")

def main():
    """Run all tests"""
    print("🚀 Starting RDS Discovery Tool Tests\n")
    
    try:
        test_json_structure()
        test_connection_failure()
        test_explanation_function()
        test_recommendation_function()
        
        print("🎉 All tests passed! The RDS Discovery Tool structure is working correctly.")
        print("\n📋 Next Steps:")
        print("1. Install SQL Server ODBC driver for actual database connections")
        print("2. Test with real SQL Server instance")
        print("3. Add more feature compatibility checks")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
