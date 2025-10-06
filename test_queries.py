#!/usr/bin/env python3
"""
Query Verification Test
Verify that all PowerShell compatibility queries have been ported
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sql_queries import FEATURE_CHECKS, PERFORMANCE_QUERIES, COMPLEX_QUERIES

def test_query_coverage():
    """Test that we have comprehensive query coverage"""
    print("🧪 Testing Query Coverage\n")
    
    # Expected features from PowerShell script
    expected_features = [
        'linked_servers', 'filestream', 'resource_governor', 'log_shipping',
        'service_broker', 'database_count', 'transaction_replication', 
        'extended_procedures', 'tsql_endpoints', 'polybase', 'buffer_pool_extension',
        'file_tables', 'stretch_database', 'trustworthy_databases', 'server_triggers',
        'machine_learning', 'data_quality_services', 'policy_based_management',
        'clr_enabled', 'always_on_ag', 'always_on_fci', 'server_role',
        'read_only_replica', 'enterprise_features', 'online_indexes'
    ]
    
    print(f"📊 Expected features: {len(expected_features)}")
    print(f"📊 Implemented features: {len(FEATURE_CHECKS)}")
    print(f"📊 Performance queries: {len(PERFORMANCE_QUERIES)}")
    print(f"📊 Complex queries: {len(COMPLEX_QUERIES)}")
    
    # Check coverage
    missing_features = []
    for feature in expected_features:
        if feature not in FEATURE_CHECKS:
            missing_features.append(feature)
    
    if missing_features:
        print(f"\n❌ Missing features: {missing_features}")
        return False
    else:
        print(f"\n✅ All expected features implemented!")
    
    # Verify query syntax (basic check)
    print(f"\n🔍 Verifying query syntax...")
    for feature, query in FEATURE_CHECKS.items():
        if not query.strip().upper().startswith('SELECT'):
            print(f"❌ Invalid query for {feature}")
            return False
        if 'CASE WHEN' not in query.upper():
            print(f"⚠️  Query for {feature} might not return Y/N format")
    
    print(f"✅ All queries have valid syntax")
    
    return True

def test_query_examples():
    """Show examples of key queries"""
    print(f"\n📋 Key Query Examples:\n")
    
    key_queries = ['linked_servers', 'always_on_ag', 'enterprise_features', 'polybase']
    
    for feature in key_queries:
        if feature in FEATURE_CHECKS:
            print(f"🔹 **{feature.upper()}**:")
            query = FEATURE_CHECKS[feature].strip()
            # Show first line of query
            first_line = query.split('\n')[0].strip()
            print(f"   {first_line}")
            print()

def main():
    """Run query verification tests"""
    print("🚀 Starting Query Verification Tests\n")
    
    try:
        if test_query_coverage():
            test_query_examples()
            print("🎉 All query verification tests passed!")
            print(f"\n📈 Summary:")
            print(f"   • {len(FEATURE_CHECKS)} compatibility checks implemented")
            print(f"   • {len(PERFORMANCE_QUERIES)} performance queries available")
            print(f"   • {len(COMPLEX_QUERIES)} complex queries for advanced scenarios")
            print(f"   • Ready for SQL Server connectivity testing")
            return 0
        else:
            print("❌ Query verification failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
