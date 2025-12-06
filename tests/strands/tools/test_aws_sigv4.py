import pytest
from strands.tools.aws_sigv4 import AwsSigV4Tool


def test_sigv4_returns_headers():
    tool = AwsSigV4Tool()
    result = tool(
        method="GET",
        service="service",
        region="us-east-1",
        host="example.com",
        uri="/",
        query="",
        headers={},
        body="",
        access_key="AKIAEXAMPLE",
        secret_key="SECRETEXAMPLE",
    )

    assert "headers" in result
    headers = result["headers"]
    assert "Authorization" in headers
    assert "x-amz-date" in headers
    # basic format checks
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
