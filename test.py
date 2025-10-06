#!/usr/bin/env python3
"""
RDS Discovery Tool - Consolidated Test Runner
Single test file for basic and comprehensive testing
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rds_discovery import strands_rds_discovery


class RDSTestRunner:
    def __init__(self, input_file, auth_type, username=None, password=None, timeout=30):
        self.input_file = input_file
        self.auth_type = auth_type
        self.username = username
        self.password = password
        self.timeout = timeout
        self.results_dir = "test_results"
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Create test results directory"""
        Path(self.results_dir).mkdir(exist_ok=True)
    
    def run_basic_tests(self):
        """Run basic functionality tests"""
        print("🧪 BASIC FUNCTIONALITY TESTS")
        print("=" * 50)
        
        results = []
        
        # Test: Assessment with user servers
        print("\n1. Testing SQL Server assessment...")
        print(f"   File: {self.input_file}")
        print(f"   Auth: {self.auth_type}")
        
        try:
            result = strands_rds_discovery(
                input_file=self.input_file,
                auth_type=self.auth_type,
                username=self.username,
                password=self.password,
                timeout=self.timeout
            )
            
            data = json.loads(result)
            if data.get('status') == 'success':
                summary = data.get('summary', {})
                performance = data.get('performance', {})
                
                results.append({
                    "test": "assessment", 
                    "status": "PASS",
                    "servers": summary.get('total_servers', 0),
                    "success_rate": summary.get('success_rate', 0),
                    "rds_compatible": summary.get('rds_compatible', 0),
                    "time": performance.get('total_time', 0)
                })
                
                print(f"✅ Assessment: PASS")
                print(f"   Total servers: {summary.get('total_servers', 0)}")
                print(f"   Success rate: {summary.get('success_rate', 0)}%")
                print(f"   RDS compatible: {summary.get('rds_compatible', 0)}")
                print(f"   Total time: {performance.get('total_time', 0)}s")
                
                # Assessment completed successfully
                print("✅ Assessment completed successfully")
                
            else:
                results.append({"test": "assessment", "status": "FAIL", "error": data.get('message', 'Unknown error')})
                print(f"❌ Assessment: FAIL - {data.get('message', 'Unknown error')}")
                
        except Exception as e:
            results.append({"test": "assessment", "status": "FAIL", "error": str(e)})
            print(f"❌ Assessment: FAIL - {str(e)}")
        
        return results
    
    def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("🧪 COMPREHENSIVE TEST SUITE")
        print("=" * 50)
        
        results = []
        
        # Test 1: Error handling
        print("\n1. Testing error handling...")
        error_tests = [
            {"name": "missing_file", "params": {"input_file": "nonexistent.csv", "auth_type": "windows"}},
            {"name": "sql_no_creds", "params": {"input_file": self.input_file, "auth_type": "sql"}}
        ]
        
        for test in error_tests:
            try:
                result = strands_rds_discovery(**test["params"])
                data = json.loads(result)
                error_handled = "error" in data.get("status", "").lower()
                results.append({
                    "test": f"error_{test['name']}", 
                    "status": "PASS" if error_handled else "FAIL",
                    "error_handled": error_handled
                })
                print(f"   {test['name']}: {'✅ PASS' if error_handled else '❌ FAIL'}")
            except Exception as e:
                results.append({"test": f"error_{test['name']}", "status": "FAIL", "error": str(e)})
                print(f"   {test['name']}: ❌ FAIL - {str(e)}")
        
        # Test 2: Different timeout values
        print("\n2. Testing timeout scenarios...")
        timeout_tests = [10, 30, 60]
        
        for timeout_val in timeout_tests:
            try:
                start_time = time.time()
                result = strands_rds_discovery(
                    input_file=self.input_file,
                    auth_type=self.auth_type,
                    username=self.username,
                    password=self.password,
                    timeout=timeout_val
                )
                elapsed_time = time.time() - start_time
                
                data = json.loads(result)
                results.append({
                    "test": f"timeout_{timeout_val}s",
                    "status": "PASS",
                    "timeout_setting": timeout_val,
                    "actual_time": round(elapsed_time, 2),
                    "within_range": elapsed_time <= (timeout_val * 2 + 10)
                })
                print(f"   {timeout_val}s timeout: ✅ PASS ({elapsed_time:.2f}s actual)")
                
            except Exception as e:
                results.append({"test": f"timeout_{timeout_val}s", "status": "FAIL", "error": str(e)})
                print(f"   {timeout_val}s timeout: ❌ FAIL - {str(e)}")
        
        # Test 3: Performance monitoring
        print("\n3. Testing performance monitoring...")
        try:
            result = strands_rds_discovery(
                input_file=self.input_file,
                auth_type=self.auth_type,
                username=self.username,
                password=self.password,
                timeout=self.timeout
            )
            
            data = json.loads(result)
            performance = data.get("performance", {})
            has_performance_data = bool(performance)
            
            results.append({
                "test": "performance_monitoring",
                "status": "PASS" if has_performance_data else "FAIL",
                "has_performance_data": has_performance_data,
                "total_time": performance.get("total_time", 0),
                "average_time": performance.get("average_time_per_server", 0)
            })
            print(f"   Performance monitoring: {'✅ PASS' if has_performance_data else '❌ FAIL'}")
            
        except Exception as e:
            results.append({"test": "performance_monitoring", "status": "FAIL", "error": str(e)})
            print(f"   Performance monitoring: ❌ FAIL - {str(e)}")
        
        # Test 3: Output file validation
        print("\n3. Testing output file generation...")
        try:
            # Check if assessment generated the expected output files
            import glob
            csv_files = glob.glob("RDSdiscovery_*.csv")
            json_files = glob.glob("RDSdiscovery_*.json") 
            log_files = glob.glob("RDSdiscovery_*.log")
            
            files_exist = len(csv_files) > 0 and len(json_files) > 0 and len(log_files) > 0
            
            results.append({
                "test": "output_files",
                "status": "PASS" if files_exist else "FAIL",
                "csv_files": len(csv_files),
                "json_files": len(json_files),
                "log_files": len(log_files)
            })
            print(f"   Output files: {'✅ PASS' if files_exist else '❌ FAIL'}")
            print(f"   CSV files: {len(csv_files)}, JSON files: {len(json_files)}, LOG files: {len(log_files)}")
            
        except Exception as e:
            results.append({"test": "output_files", "status": "FAIL", "error": str(e)})
            print(f"   Output files: ❌ FAIL - {str(e)}")
        
        return results
    
    def run_simulation_tests(self):
        """Run simulation tests with random SQL Server configurations"""
        print("\n🎲 SIMULATION TESTS")
        print("=" * 50)
        
        import random
        
        # Define feature sets and their RDS blocking status
        features = {
            # RDS Blockers (from PowerShell script)
            "linked_servers": {"blocks_rds": True, "probability": 0.3},
            "filestream": {"blocks_rds": True, "probability": 0.2},
            "clr_enabled": {"blocks_rds": True, "probability": 0.15},
            "extended_procedures": {"blocks_rds": True, "probability": 0.1},
            "resource_governor": {"blocks_rds": True, "probability": 0.25},
            "polybase": {"blocks_rds": True, "probability": 0.1},
            "stretch_database": {"blocks_rds": True, "probability": 0.05},
            
            # Non-blockers (status only)
            "always_on_ag": {"blocks_rds": False, "probability": 0.4},
            "service_broker": {"blocks_rds": False, "probability": 0.3},
            "ssis": {"blocks_rds": False, "probability": 0.6},
            "ssrs": {"blocks_rds": False, "probability": 0.4},
            "log_shipping": {"blocks_rds": True, "probability": 0.2},
            "transaction_replication": {"blocks_rds": True, "probability": 0.15}
        }
        
        # CPU/Memory configurations
        server_configs = [
            {"cpu": 2, "memory_mb": 4096, "type": "Small"},
            {"cpu": 4, "memory_mb": 8192, "type": "Medium"},
            {"cpu": 8, "memory_mb": 16384, "type": "Large"},
            {"cpu": 16, "memory_mb": 32768, "type": "XLarge"},
            {"cpu": 32, "memory_mb": 65536, "type": "XXLarge"},
            {"cpu": 64, "memory_mb": 131072, "type": "XXXLarge"}
        ]
        
        simulation_results = []
        
        # Generate 20 random server configurations
        for i in range(1, 21):
            server_name = f"sim-server-{i:02d}.test.com"
            config = random.choice(server_configs)
            
            # Randomly enable features based on probability
            enabled_features = {}
            blocking_features = []
            
            for feature, props in features.items():
                if random.random() < props["probability"]:
                    enabled_features[feature] = "Y"
                    if props["blocks_rds"]:
                        blocking_features.append(feature)
                else:
                    enabled_features[feature] = "N"
            
            # Determine RDS compatibility
            rds_compatible = "N" if blocking_features else "Y"
            
            # Get AWS instance recommendation
            from rds_discovery import get_aws_instance_recommendation
            cpu_cores = config["cpu"]
            memory_gb = config["memory_mb"] / 1024
            aws_instance, match_type = get_aws_instance_recommendation(cpu_cores, memory_gb)
            
            simulation_results.append({
                "server": server_name,
                "config_type": config["type"],
                "cpu": config["cpu"],
                "memory_gb": memory_gb,
                "aws_instance": aws_instance,
                "rds_compatible": rds_compatible,
                "blocking_features": blocking_features,
                "enabled_features": [f for f, v in enabled_features.items() if v == "Y"]
            })
            
            # Print simulation result
            status_icon = "✅" if rds_compatible == "Y" else "❌"
            print(f"{i:2d}. {status_icon} {server_name}")
            print(f"    Config: {config['type']} ({config['cpu']} CPU, {memory_gb:.1f}GB RAM)")
            print(f"    AWS Instance: {aws_instance}")
            print(f"    RDS Compatible: {rds_compatible}")
            if blocking_features:
                print(f"    Blockers: {', '.join(blocking_features)}")
            enabled_list = [f for f, v in enabled_features.items() if v == "Y"]
            if enabled_list:
                print(f"    Features: {', '.join(enabled_list[:3])}{'...' if len(enabled_list) > 3 else ''}")
            print()
        
        # Generate summary statistics
        total_servers = len(simulation_results)
        rds_compatible_count = len([r for r in simulation_results if r["rds_compatible"] == "Y"])
        rds_incompatible_count = total_servers - rds_compatible_count
        
        # Feature frequency analysis
        feature_stats = {}
        for result in simulation_results:
            for feature in result["enabled_features"]:
                feature_stats[feature] = feature_stats.get(feature, 0) + 1
        
        # Instance type distribution
        instance_stats = {}
        for result in simulation_results:
            instance = result["aws_instance"]
            instance_stats[instance] = instance_stats.get(instance, 0) + 1
        
        print("📊 SIMULATION SUMMARY")
        print("=" * 30)
        print(f"Total Servers: {total_servers}")
        print(f"RDS Compatible: {rds_compatible_count} ({rds_compatible_count/total_servers*100:.1f}%)")
        print(f"RDS Incompatible: {rds_incompatible_count} ({rds_incompatible_count/total_servers*100:.1f}%)")
        
        print(f"\n🔧 Most Common Features:")
        for feature, count in sorted(feature_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {count}/{total_servers} ({count/total_servers*100:.1f}%)")
        
        print(f"\n💻 AWS Instance Distribution:")
        for instance, count in sorted(instance_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {instance}: {count} servers")
        
        # Save simulation results
        with open(f"{self.results_dir}/simulation_results.json", "w") as f:
            json.dump(simulation_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {self.results_dir}/simulation_results.json")
        
        return simulation_results
    
    def run_tests(self, mode):
        """Run tests based on mode"""
        if mode == "basic":
            return self.run_basic_tests()
        elif mode == "simulation":
            return self.run_simulation_tests()
        elif mode == "full":
            basic_results = self.run_basic_tests()
            comprehensive_results = self.run_comprehensive_tests()
            return basic_results + comprehensive_results
        else:
            raise ValueError(f"Unknown test mode: {mode}. Available: basic, simulation, full")
    
    def print_summary(self, results):
        """Print test summary"""
        passed = len([r for r in results if r.get("status") == "PASS"])
        failed = len([r for r in results if r.get("status") == "FAIL"])
        
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {passed/len(results)*100:.1f}%")
        
        if failed > 0:
            print(f"\n❌ Failed tests:")
            for result in results:
                if result.get("status") == "FAIL":
                    print(f"   - {result.get('test')}: {result.get('error', 'Unknown error')}")
        
        # Save detailed results
        with open(f"{self.results_dir}/test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {self.results_dir}/test_results.json")
        return passed == len(results)


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='RDS Discovery Tool - Consolidated Test Runner')
    parser.add_argument('--input-file', required=True, help='CSV file with server list')
    parser.add_argument('--auth-type', choices=['windows', 'sql'], default='windows', 
                       help='Authentication type (windows or sql)')
    parser.add_argument('--username', help='SQL Server username (required for SQL auth)')
    parser.add_argument('--password', help='SQL Server password (required for SQL auth)')
    parser.add_argument('--timeout', type=int, default=30, help='Connection timeout in seconds')
    parser.add_argument('--mode', choices=['basic', 'simulation', 'full'], default='basic',
                       help='Test mode: basic (quick tests), simulation (random configs), or full (comprehensive)')
    
    args = parser.parse_args()
    
    # Validate SQL authentication parameters
    if args.auth_type == 'sql':
        if not args.username or not args.password:
            print("❌ Error: Username and password required for SQL Server authentication")
            sys.exit(1)
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    print("🚀 RDS Discovery Tool - Consolidated Test Runner")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Authentication: {args.auth_type}")
    if args.auth_type == 'sql':
        print(f"Username: {args.username}")
    print(f"Timeout: {args.timeout}s")
    print(f"Mode: {args.mode}")
    print("")
    
    # Run tests
    tester = RDSTestRunner(args.input_file, args.auth_type, args.username, args.password, args.timeout)
    results = tester.run_tests(args.mode)
    success = tester.print_summary(results)
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
