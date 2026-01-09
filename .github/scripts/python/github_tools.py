"""GitHub repository management tool for Strands Agents.

This module provides comprehensive GitHub repository operations including issues,
pull requests, comments, and repository management. Supports full GitHub API
integration with rich console output and error handling.

Key Features:
1. List and manage issues and pull requests
2. Add comments to issues and PRs
3. Create, update, and manage issues
4. Create, update, and manage pull requests
5. Get detailed information for specific issues/PRs
6. Manage PR reviews and review comments
7. Get issue and PR comment threads
8. Check GitHub token permissions for repositories
9. Rich console output with formatted tables
10. Automatic fallback to GITHUB_REPOSITORY environment variable

Usage Examples:
```python
from strands import Agent
from tools.github_tools import list_issues, add_comment, create_issue, _check_token_permissions

agent = Agent(tools=[list_issues, add_comment, create_issue])

# Check token permissions
has_write = _check_token_permissions("ghp_token123", "owner/repo")

# List open issues in repository
result = agent.tool.list_issues(state="open", repo="owner/repo")

# Add comment to an issue
result = agent.tool.add_comment(
    issue_number=42,
    comment_text="Great idea! I'll work on this.",
    repo="owner/repo"
)

# Create a new issue
result = agent.tool.create_issue(
    title="Bug: Application crashes on startup",
    body="Description of the issue with steps to reproduce...",
    repo="owner/repo"
)

# List pull requests
result = agent.tool.list_pull_requests(state="open", repo="owner/repo")

# Get specific issue details
result = agent.tool.get_issue(issue_number=123, repo="owner/repo")

# Update pull request
result = agent.tool.update_pull_request(
    pr_number=456,
    title="Updated PR title",
    body="Updated description",
    repo="owner/repo"
)
```
"""

import os
import traceback
from datetime import datetime
from functools import wraps
import json
from typing import Any, TypedDict
from urllib.parse import urlencode, quote

import requests
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from strands import tool
from strands_tools.utils import console_util

console = console_util.create()


class GitHubOperation(TypedDict):
    """Type definition for GitHub operation records in JSONL files."""
    timestamp: str
    function: str
    args: list[Any]
    kwargs: dict[str, Any]


def log_inputs(func):
    """Decorator to log function inputs in a blue panel."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name and format it nicely
        func_name = func.__name__.replace('_', ' ').title()
        
        # Format parameters
        params = []
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v) > 50:
                params.append(f"{k}='{v[:50]}...'")
            else:
                params.append(f"{k}='{v}'")
        
        console.print(Panel(", ".join(params), title=f"[bold blue]{func_name}", border_style="blue"))
        return func(*args, **kwargs)
    return wrapper


def _github_request(
    method: str, endpoint: str, repo: str | None = None, data: dict | None = None, params: dict | None = None, should_raise: bool = False
) -> dict[str, Any] | str:
    """Make a GitHub API request with common error handling.

    Args:
        method: HTTP method (GET, POST, PATCH, etc.)
        endpoint: API endpoint path (e.g., "pulls", "issues/123")
        repo: Repository in "owner/repo" format
        data: JSON data for request body
        params: Query parameters for the request

    Returns:
        Response JSON or error string
    """
    if repo is None:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return "Error: GITHUB_REPOSITORY environment variable not found"

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found"

    url = f"https://api.github.com/repos/{repo}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, params=params, timeout=30)
        else:
            response = requests.request(method, url, headers=headers, json=data, params=params, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception as e:
        if should_raise:
            raise e
        return f"Error {e!s}"


def check_should_call_write_api_or_record(func):
    """Decorator that checks if a write api should be called, or if the tool should record to JSONL."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if not _should_call_write_api():
                # Record the tool request to JSONL file
                record_entry: GitHubOperation = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }
                
                os.makedirs(".artifact", exist_ok=True)
                with open(".artifact/write_operations.jsonl", "a") as f:
                    f.write(json.dumps(record_entry) + "\n")
                
                # Generate and return deferred message
                params = dict(kwargs)
                if args:
                    # Map positional args to parameter names from function signature
                    import inspect
                    sig = inspect.signature(func)
                    param_names = list(sig.parameters.keys())
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            params[param_names[i]] = arg
                
                deferred_msg = _generate_deferred_message(func.__name__, params)
                console.print(Panel(escape(deferred_msg), title="[bold yellow]Operation Deferred", border_style="yellow"))
                return deferred_msg
        except Exception as e:
            error_msg = f"Error checking permissions: {e!s}"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            return error_msg
        
        return func(*args, **kwargs)
    return wrapper


def _generate_deferred_message(operation_name: str, params: dict[str, Any]) -> str:
    """Generate a consistent deferred message for write operations.
    
    Args:
        operation_name: Name of the operation being deferred
        params: Parameters that would have been used for the operation
        
    Returns:
        Formatted deferred message string
    """
    if not params:
        return f"Operation deferred: {operation_name}"
    
    # Format parameters, truncating long values
    param_strs = []
    for key, value in params.items():
        if isinstance(value, str) and len(value) > 50:
            param_strs.append(f"{key}='{value[:50]}...'")
        elif isinstance(value, str):
            param_strs.append(f"{key}='{value}'")
        else:
            param_strs.append(f"{key}={value}")
    
    return f"Operation deferred: {operation_name} - {', '.join(param_strs)}"


def _should_call_write_api() -> bool:
    """Checks if GITHUB_WRITE environment variable is set to true.
        
    Returns:
        bool: True if GITHUB_WRITE is set to 'true', False otherwise
    """
    return os.environ.get("GITHUB_WRITE", "").lower() == "true"


# =============================================================================
# WRITE FUNCTIONS (Functions that modify GitHub resources)
# =============================================================================

@tool
@log_inputs
@check_should_call_write_api_or_record
def create_issue(title: str, body: str = "", repo: str | None = None) -> str:
    """Creates a new issue in the specified repository.

    Args:
        title: The issue title
        body: The issue body (optional)
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Result of the operation
    """
    result = _github_request("POST", "issues", repo, {"title": title, "body": body})
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    message = f"Issue created: #{result['number']} - {result['html_url']}"
    console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
    return message


@tool
@log_inputs
@check_should_call_write_api_or_record
def update_issue(
    issue_number: int,
    title: str | None = None,
    body: str | None = None,
    state: str | None = None,
    repo: str | None = None,
) -> str:
    """Updates an issue's title, body, or state.

    Args:
        issue_number: The issue number
        title: New title (optional)
        body: New body (optional)
        state: New state - "open" or "closed" (optional)
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Result of the operation
    """
    data = {}
    if title is not None:
        data["title"] = title
    if body is not None:
        data["body"] = body
    if state is not None:
        data["state"] = state

    if not data:
        error_msg = "Error: At least one field (title, body, or state) must be provided"
        console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
        return error_msg

    result = _github_request("PATCH", f"issues/{issue_number}", repo, data)
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    message = f"Issue updated: #{result['number']} - {result['html_url']}"
    console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
    return message


@tool
@log_inputs
@check_should_call_write_api_or_record
def add_issue_comment(issue_number: int, comment_text: str, repo: str | None = None) -> str:
    """Adds a comment to an issue or pull request in the specified repository or GITHUB_REPOSITORY environment variable.

    Args:
        issue_number: The issue or PR number to comment on
        comment_text: The comment text
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Result of the operation
    """
    result = _github_request("POST", f"issues/{issue_number}/comments", repo, {"body": comment_text})
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    message = f"Comment added successfully: {result['html_url']} (created: {result['created_at']})"
    console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
    return message


@tool
@log_inputs
@check_should_call_write_api_or_record
def create_pull_request(title: str, head: str, base: str, body: str = "", repo: str | None = None, fallback_issue_id: int | None = None) -> str:
    """Creates a new pull request, or optionally comments on the fallback_issue_id for a link to create a pull request.

    Args:
        title: The PR title
        head: The branch containing changes
        base: The branch to merge into
        body: The PR body (optional)
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)
        fallback_issue_id: Issue ID to comment on if PR creation fails with an error (optional)

    Returns:
        Result of the operation
    """
    try:
        result = _github_request(
            "POST",
            "pulls",
            repo,
            {"title": title, "head": head, "base": base, "body": body},
            should_raise=True
        )

        if isinstance(result, str):
            console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
            return result


        message = f"Pull request created: #{result['number']} - {result['html_url']}"
        console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
        return message
    
    except Exception as e:
        if fallback_issue_id is not None:
            agent_message = "Failed to create pull request, commenting on issue instead."
            console.print(Panel(escape(agent_message), title="[bold yellow]Fallback", border_style="yellow"))
            repo_name = repo or os.environ.get("GITHUB_REPOSITORY", "")
            query_params = urlencode({
                'quick_pull': '1',
                'title': title,
                'body': body
            }, quote_via=quote)
            pr_link = f"https://github.com/{repo_name}/compare/{base}...{head}?{query_params}"
            fallback_comment = f"Unable to create pull request via API. You can create it manually by clicking [here]({pr_link})."
            add_issue_comment(fallback_issue_id, fallback_comment, repo)
            return f"Unable to create pull request via API - posted a manual creation link as a comment on issue #{fallback_issue_id}"
        else:
            error_msg = f"Error: {e!s}"
            console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
            return error_msg


@tool
@log_inputs
@check_should_call_write_api_or_record
def update_pull_request(
    pr_number: int,
    title: str | None = None,
    body: str | None = None,
    base: str | None = None,
    repo: str | None = None,
) -> str:
    """Updates a pull request's title, body, or base branch.

    Args:
        pr_number: The pull request number
        title: New title (optional)
        body: New body (optional)
        base: New base branch (optional)
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Result of the operation
    """
    data = {}
    if title is not None:
        data["title"] = title
    if body is not None:
        data["body"] = body
    if base is not None:
        data["base"] = base

    if not data:
        error_msg = "Error: At least one field (title, body, or base) must be provided"
        console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
        return error_msg

    result = _github_request("PATCH", f"pulls/{pr_number}", repo, data)
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    message = f"Pull request updated: #{result['number']} - {result['html_url']}"
    console.print(Panel(escape(message), title="[bold green]Success", border_style="green"))
    return message


@tool
@log_inputs
@check_should_call_write_api_or_record
def reply_to_review_comment(pr_number: int, comment_id: int, reply_text: str, repo: str | None = None) -> str:
    """Replies to a pull request review comment.

    Args:
        pr_number: The pull request number
        comment_id: The review comment ID to reply to
        reply_text: The reply text
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Result of the operation
    """
    result = _github_request("POST", f"pulls/{pr_number}/comments/{comment_id}/replies", repo, {"body": reply_text})
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    message = f"Reply added to review comment: {result['html_url']}"
    reply_details = f"Reply: {reply_text}\nURL: {result['html_url']}"
    console.print(Panel(escape(reply_details), title="[bold green]✅ Reply Added", border_style="green"))
    return message


# =============================================================================
# READ FUNCTIONS (Functions that only read GitHub resources)
# =============================================================================

@tool
@log_inputs
def get_issue(issue_number: int, repo: str | None = None) -> str:
    """Gets details of a specific issue.

    Args:
        issue_number: The issue number
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Issue details
    """
    result = _github_request("GET", f"issues/{issue_number}", repo)
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    details = (
        f"#{result['number']} - {result['title']}\n"
        f"State: {result['state']}\n"
        f"Author: {result['user']['login']}\n"
        f"URL: {result['html_url']}\n\n{result['body']}"
    )
    console.print(
        Panel(
            escape(details),
            title=f"[bold green]📋 Issue #{result['number']}",
            border_style="blue",
        )
    )
    return details


@tool
@log_inputs
def list_issues(state: str = "open", repo: str | None = None) -> str:
    """Lists issues from the specified GitHub repository or GITHUB_REPOSITORY environment variable.

    Args:
        state: Filter issues by state: "open", "closed", or "all" (default: "open")
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        String representation of the issues
    """
    result = _github_request("GET", "issues", repo, params={"state": state})
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    # Filter out pull requests from issues list
    issues = [issue for issue in result if "pull_request" not in issue]
    if not issues:
        message = f"No {state} issues found in {repo or os.environ.get('GITHUB_REPOSITORY')}"
        console.print(Panel(escape(message), title="[bold yellow]Info", border_style="yellow"))
        return message

    table = Table(title=f"🐛 Issues ({state})", box=box.DOUBLE)
    table.add_column("Issue #", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Author", style="green")
    table.add_column("URL", style="blue")

    for issue in issues:
        table.add_row(
            f"#{issue['number']}",  # type: ignore[index]
            issue["title"],  # type: ignore[index]
            issue["user"]["login"],  # type: ignore[index]
            issue["html_url"],  # type: ignore[index]
        )

    console.print(table)

    output = f"Issues ({state}) in {repo or os.environ.get('GITHUB_REPOSITORY')}:\n"
    for issue in issues:
        output += f"#{issue['number']} - {issue['title']} by {issue['user']['login']} - {issue['html_url']}\n"  # type: ignore[index]
    return output


@tool
@log_inputs
def get_issue_comments(issue_number: int, repo: str | None = None, since: str | None = None) -> str:
    """Gets all comments for a specific issue.

    Args:
        issue_number: The issue number
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)
        since: ISO 8601 timestamp to filter comments updated after this date (optional)

    Returns:
        List of comments
    """
    params = {"since": since} if since else None
    result = _github_request("GET", f"issues/{issue_number}/comments", repo, params=params)
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    if not result:
        message = f"No comments found for issue #{issue_number}" + (f" updated after {since}" if since else "")
        console.print(Panel(escape(message), title="[bold yellow]Info", border_style="yellow"))
        return message

    output = f"Comments for issue #{issue_number}:\n"
    for comment in result:
        output += f"{comment['user']['login']} - updated: {comment['updated_at']}\n{comment['body']}\n\n"  # type: ignore[index]
    
    console.print(Panel(escape(output), title=f"[bold green]💬 Issue #{issue_number} Comments", border_style="blue"))
    return output


@tool
@log_inputs
def get_pull_request(pr_number: int, repo: str | None = None) -> str:
    """Gets details of a specific pull request.

    Args:
        pr_number: The pull request number
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        Pull request details
    """
    result = _github_request("GET", f"pulls/{pr_number}", repo)
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    details = (
        f"#{result['number']} - {result['title']}\n"
        f"State: {result['state']}\n"
        f"Author: {result['user']['login']}\n"
        f"Head: {result['head']['ref']} -> Base: {result['base']['ref']}\n"
        f"URL: {result['html_url']}\n\n{result['body']}"
    )
    console.print(
        Panel(
            escape(details),
            title=f"[bold green]🔀 PR #{result['number']}",
            border_style="blue",
        )
    )
    return details


@tool
@log_inputs
def list_pull_requests(state: str = "open", repo: str | None = None) -> str:
    """Lists pull requests from the specified GitHub repository or GITHUB_REPOSITORY environment variable.

    Args:
        state: Filter PRs by state: "open", "closed", or "all" (default: "open")
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)

    Returns:
        String representation of the pull requests
    """
    result = _github_request("GET", "pulls", repo, params={"state": state})
    if isinstance(result, str):
        console.print(Panel(escape(result), title="[bold red]Error", border_style="red"))
        return result

    if not result:
        message = f"No {state} pull requests found in {repo or os.environ.get('GITHUB_REPOSITORY')}"
        console.print(Panel(escape(message), title="[bold yellow]Info", border_style="yellow"))
        return message

    table = Table(title=f"🔀 Pull Requests ({state})", box=box.DOUBLE)
    table.add_column("PR #", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Author", style="green")
    table.add_column("URL", style="blue")

    for pr in result:
        table.add_row(f"#{pr['number']}", pr["title"], pr["user"]["login"], pr["html_url"])  # type: ignore[index]

    console.print(table)

    output = f"Pull Requests ({state}) in {repo or os.environ.get('GITHUB_REPOSITORY')}:\n"
    for pr in result:
        output += f"#{pr['number']} - {pr['title']} by {pr['user']['login']} - {pr['html_url']}\n"  # type: ignore[index]
    return output


@tool
@log_inputs
def get_pr_review_and_comments(pr_number: int, show_resolved: bool = False, repo: str | None = None, since: str | None = None) -> str:
    """Gets all review threads and comments for a PR.

    Args:
        pr_number: The pull request number
        repo: GitHub repository in the format "owner/repo" (optional; falls back to env var)
        show_resolved: Whether to include resolved review threads (default: False)
        since: ISO 8601 timestamp to filter comments/threads updated after this date (optional)

    Returns:
        Formatted review threads and comments
    """
    if repo is None:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        return "Error: GITHUB_REPOSITORY environment variable not found"

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not found"

    owner, repo_name = repo.split("/")
    
    query = """
    query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          reviewThreads(first: 100) {
            nodes {
              isResolved
              comments(first: 100) {
                nodes {
                  id
                  fullDatabaseId
                  author { login }
                  body
                  updatedAt
                  path
                  line
                  startLine
                  diffHunk
                  replyTo { id }
                  pullRequestReview { 
                    id 
                    body
                    author { login }
                    updatedAt
                  }
                }
              }
            }
          }
          comments(first: 100) {
            nodes {
              author { login }
              body
              updatedAt
            }
          }
        }
      }
    }
    """
    
    variables = {"owner": owner, "name": repo_name, "number": pr_number}
    
    try:
        response = requests.post(
            "https://api.github.com/graphql",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": query, "variables": variables},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "errors" in data:
            return f"GraphQL Error: {data['errors']}"
            
        pr_data = data["data"]["repository"]["pullRequest"]
        
        # Filter by since if provided
        if since:
            cutoff = datetime.fromisoformat(since.replace('Z', '+00:00'))
            
            # Filter review threads - if any comment in thread is newer, include entire thread
            filtered_threads = []
            for thread in pr_data["reviewThreads"]["nodes"]:
                has_newer_comment = any(datetime.fromisoformat(c['updatedAt'].replace('Z', '+00:00')) > cutoff 
                                      for c in thread["comments"]["nodes"])
                if has_newer_comment:
                    filtered_threads.append(thread)
            pr_data["reviewThreads"]["nodes"] = filtered_threads
            
            # Filter general comments
            pr_data["comments"]["nodes"] = [c for c in pr_data["comments"]["nodes"] 
                                          if datetime.fromisoformat(c['updatedAt'].replace('Z', '+00:00')) > cutoff]
        
        output = f"Review threads and comments for PR #{pr_number}:\n\n"
        
        # Group review threads by review ID
        review_threads = {}
        for thread in pr_data["reviewThreads"]["nodes"]:
            if not show_resolved and thread["isResolved"]:
                continue
                
            if thread["comments"]["nodes"]:
                first_comment = thread["comments"]["nodes"][0]
                review_id = first_comment.get("pullRequestReview", {}).get("id", "N/A")
                
                if review_id not in review_threads:
                    review_threads[review_id] = {
                        "review_data": first_comment.get("pullRequestReview", {}),
                        "threads": []
                    }
                
                review_threads[review_id]["threads"].append(thread)
        
        # Display grouped review threads
        for review_id, review_info in review_threads.items():
            review_data = review_info['review_data']
            output += f"📝 Review [Review ID: {review_id}]\n"
            
            # Always show review author and timestamps
            if review_data.get('author'):
                output += f"   👤 Review by {review_data['author']['login']} (updated: {review_data['updatedAt']})\n"
            
            # Show top-level review comment if it exists
            if review_data.get('body'):
                output += f"   📋 Review Comment:\n"
                output += f"      {review_data['body']}\n"
            output += "\n"
            
            # Show all threads for this review
            for thread in review_info["threads"]:
                first_comment = thread["comments"]["nodes"][0]
                line_info = f":{first_comment['line']}" if first_comment.get('line') else " (Comment on file)"
                status = "✅ RESOLVED" if thread["isResolved"] else "🔄 OPEN"
                
                output += f"   📍 Thread ({status}): {first_comment['path']}{line_info}\n"
                
                # Show code context right after thread header
                if first_comment.get('diffHunk') and first_comment.get('line'):
                    diff_lines = first_comment['diffHunk'].split('\n')
                    current_new_line = 0
                    target_line = first_comment['line']
                    start_line = first_comment.get('startLine') or target_line
                    
                    output += f"      Code context (lines {start_line}-{target_line}):\n"
                    
                    for diff_line in diff_lines:
                        if diff_line.startswith('@@'):
                            parts = diff_line.split(' ')
                            if len(parts) >= 3:
                                new_start = parts[2].split(',')[0][1:]
                                current_new_line = int(new_start) - 1
                        elif diff_line.startswith('+'):
                            current_new_line += 1
                            if start_line <= current_new_line <= target_line:
                                output += f"       +{current_new_line}: {diff_line[1:]}\n"
                        elif diff_line.startswith('-'):
                            pass
                        elif diff_line.startswith(' '):
                            current_new_line += 1
                            if start_line <= current_new_line <= target_line:
                                output += f"        {current_new_line}: {diff_line[1:]}\n"
                    output += "\n"
                
                # Group comments by reply relationships
                comments = thread["comments"]["nodes"]
                root_comments = [c for c in comments if not c.get('replyTo')]
                
                for root_comment in root_comments:
                    output += f"      💬 {root_comment['author']['login']} (updated: {root_comment['updatedAt']}) [Comment ID: {root_comment['fullDatabaseId']}]:\n"
                    output += f"         {root_comment['body']}\n"
                    
                    # Find and show replies to this comment
                    replies = [c for c in comments if c.get('replyTo') and c['replyTo'].get('id') == root_comment['id']]
                    if replies:
                        for reply in replies:
                            output += f"         ↳ {reply['author']['login']} (updated: {reply['updatedAt']}):\n"
                            output += f"           {reply['body']}\n"
                
                output += "\n"
            output += "\n"
        
        # General comments
        if pr_data["comments"]["nodes"]:
            for comment in pr_data["comments"]["nodes"]:
                output += f"💬 Comment\n"
                output += f"   👤 Comment by {comment['author']['login']} (updated: {comment['updatedAt']})\n"
                output += f"   📝 Comment:\n"
                output += f"      {comment['body']}\n\n"
        
        console.print(Panel(escape(output), title=f"[bold green]PR #{pr_number} Review Data", border_style="blue"))
        return output
        
    except Exception as e:
        error_msg = f"Error: {e!s}\n\nStack trace:\n{traceback.format_exc()}"
        console.print(Panel(escape(error_msg), title="[bold red]Error", border_style="red"))
        return error_msg
