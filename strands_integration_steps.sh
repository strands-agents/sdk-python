#!/bin/bash
# Strands Integration Steps

echo "🚀 STRANDS RDS DISCOVERY INTEGRATION"
echo "===================================="
echo ""

echo "Step 1: Fork Strands Repository"
echo "1. Go to: https://github.com/strands-ai/strands (or main Strands repo)"
echo "2. Click 'Fork' button"
echo "3. Clone your fork:"
echo "   git clone https://github.com/YOUR-USERNAME/strands.git"
echo ""

echo "Step 2: Setup Integration Branch"
echo "cd strands"
echo "git checkout -b add-rds-discovery-tool"
echo ""

echo "Step 3: Copy RDS Discovery Files"
echo "# Copy main tool"
echo "cp /home/bacrifai/strands-rds-discovery/src/rds_discovery.py strands/tools/"
echo "cp /home/bacrifai/strands-rds-discovery/src/sql_queries.py strands/tools/"
echo ""

echo "Step 4: Update Dependencies"
echo "echo 'pyodbc>=4.0.0' >> requirements.txt"
echo ""

echo "Step 5: Register Tool"
echo "# Edit strands/tools/__init__.py to add:"
echo "# from .rds_discovery import strands_rds_discovery"
echo ""

echo "Step 6: Add Documentation"
echo "mkdir -p docs/tools"
echo "cp /home/bacrifai/strands-rds-discovery/USAGE_GUIDE.md docs/tools/rds_discovery.md"
echo ""

echo "Step 7: Add Tests"
echo "mkdir -p tests/tools"
echo "cp /home/bacrifai/strands-rds-discovery/quick_test.py tests/tools/test_rds_discovery.py"
echo ""

echo "Step 8: Commit and Push"
echo "git add ."
echo "git commit -m 'Add RDS Discovery tool for SQL Server migration assessment'"
echo "git push origin add-rds-discovery-tool"
echo ""

echo "Step 9: Create Pull Request"
echo "Go to GitHub and create PR from your branch to main Strands repo"
echo ""

echo "📋 What do you need to provide:"
echo "1. Strands repository URL"
echo "2. Your GitHub username"
echo ""

