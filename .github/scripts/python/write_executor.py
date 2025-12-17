#!/usr/bin/env python3
"""Write Executor Script for GitHub Operations.

This script reads JSONL artifact files containing deferred GitHub operations
and executes them using functions from github_tools.py. It's designed to run
after the strands-agent-runner to publish any write commands or commits.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from github_tools import GitHubOperation

# Import write only github_tools functions for dynamic execution
from github_tools import (
    create_issue,
    update_issue, 
    add_issue_comment,
    create_pull_request,
    update_pull_request,
    reply_to_review_comment,
)

# Configure structured logging
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)
logger = logging.getLogger("write_executor")


def get_function_mapping() -> Dict[str, Any]:
    """Get mapping of function names to actual functions."""
    return {
        create_issue.tool_name: create_issue,
        update_issue.tool_name: update_issue,
        add_issue_comment.tool_name: add_issue_comment,
        create_pull_request.tool_name: create_pull_request,
        update_pull_request.tool_name: update_pull_request,
        reply_to_review_comment.tool_name: reply_to_review_comment,
    }


def process_jsonl_file(file_path: Path, default_issue_id: int | None = None):
    """Process JSONL file and execute operations.
    
    Args:
        file_path: Path to the JSONL artifact file
        default_issue_id: Default issue ID to use for fallback operations
        
    Returns:
        Tuple of (total_operations, successful_operations, failed_operations)
    """
    function_map = get_function_mapping()
    
    logger.info(f"Starting JSONL processing: {file_path}")
    total_ops = 0
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            total_ops += 1
            logger.info(f"Processing operation {total_ops} (line {line_num})")
            
            try:
                # Parse JSONL entry
                operation: GitHubOperation = json.loads(line)
                func_name = operation.get("function")
                args = operation.get('args', [])
                kwargs = operation.get('kwargs', {})
                
                if not func_name:
                    logger.error(f"Line {line_num}: Missing function name")
                    continue
                
                # Get function from mapping
                if func_name not in function_map:
                    logger.error(f"Line {line_num}: Unknown function '{func_name}'")
                    continue
                
                func = function_map[func_name]
                
                # Set default issue ID for create_pull_request if not already set
                if func_name == "create_pull_request" and default_issue_id and not kwargs.get("fallback_issue_id"):
                    kwargs["fallback_issue_id"] = default_issue_id
                
                # Execute function
                logger.info(f"Executing {func_name} with args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                
                logger.info(f"Line {line_num}: Operation {func_name} completed successfully")
                logger.info(f"Function output: {str(result)}")
                    
            except Exception as e:
                logger.error(f"Line {line_num}: Execution error - {e}")
                    
    
    logger.info(f"JSONL processing completed.")


def main():
    """Main entry point for the write executor script."""
    parser = argparse.ArgumentParser(
        description="Execute deferred GitHub operations from JSONL artifact files"
    )
    parser.add_argument(
        "artifact_file",
        help="Path to JSONL artifact file containing deferred operations"
    )
    parser.add_argument(
        "--issue-id",
        type=int,
        help="Default issue ID to use for fallback operations"
    )
    
    args = parser.parse_args()
    artifact_path = Path(args.artifact_file)
    
    logger.info(f"Write executor started with artifact file: {artifact_path}")
    if args.issue_id:
        logger.info(f"Default issue ID set to: {args.issue_id}")
    
    # Check if file exists
    if not artifact_path.exists():
        logger.warning(f"Artifact file not found: {artifact_path}")
        logger.warning("No deferred operations to execute")
        return
    
    # Check if file is empty
    if artifact_path.stat().st_size == 0:
        logger.info("Artifact file is empty")
        logger.info("No deferred operations to execute")
        return
    
    # Set environment to enable write operations
    os.environ['GITHUB_WRITE'] = 'true'
    logger.info("GitHub write mode enabled")
    
    logger.info(f"Processing deferred operations from: {artifact_path}")
    
    # Process the JSONL file
    process_jsonl_file(artifact_path, args.issue_id)

if __name__ == "__main__":
    main()
