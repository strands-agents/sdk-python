# AWS SigV4 Signing Tool

## Description
`AwsSigV4Tool` produces AWS Signature Version 4 headers for HTTP requests.
It can be used with the Strands HTTP tool to securely call any AWS service without external dependencies.

## Usage

```
from aws_sigv4_tool import AwsSigV4Tool

tool = AwsSigV4Tool()

signed_headers = tool(
    method="GET",
    service="s3",
    region="us-east-1",
    host="examplebucket.s3.amazonaws.com",
    uri="/test.txt",
    query="",
    headers={},
    body="",
    access_key="AKIA...",
    secret_key="SECRET..."
)
# Use signed_headers["headers"] with the Strands HTTP tool
```
## Parameters

| Name       | Type | Description                   |
| ---------- | ---- | ----------------------------- |
| method     | str  | HTTP method (GET, POST, etc.) |
| service    | str  | AWS service name (e.g., s3)   |
| region     | str  | AWS region (e.g., us-east-1)  |
| host       | str  | AWS service endpoint host     |
| uri        | str  | Request URI (path)            |
| query      | str  | Query string                  |
| headers    | dict | Existing headers to include   |
| body       | str  | Request body                  |
| access_key | str  | AWS Access Key ID             |
| secret_key | str  | AWS Secret Access Key         |

## Returns

```
{
    "headers": {
        "x-amz-date": "20251130T190000Z",
        "Authorization": "AWS4-HMAC-SHA256 Credential=..."
    }
}
```

## Notes
    Fully compatible with Strands HTTP tool.
    No external dependencies; pure Python.
    Safe for testing — no actual AWS credentials required.