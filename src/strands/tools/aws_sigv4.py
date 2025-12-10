from datetime import datetime
import hashlib
import hmac
from typing import Dict


class AwsSigV4Tool:
    """
    Tool: aws_sigv4
    Description: Produces AWS Signature Version 4 headers for an HTTP request.

    Usage:
        tool = AwsSigV4Tool()
        headers = tool(
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
    """

    def __call__(
        self,
        method: str,
        service: str,
        region: str,
        host: str,
        uri: str,
        query: str,
        headers: Dict[str, str],
        body: str,
        access_key: str,
        secret_key: str,
    ) -> Dict[str, Dict[str, str]]:

        def _sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        def _get_signature_key(key, date_stamp, region_name, service_name):
            k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
            k_region = _sign(k_date, region_name)
            k_service = _sign(k_region, service_name)
            k_signing = _sign(k_service, "aws4_request")
            return k_signing

        # Step 1: Prepare timestamps
        t = datetime.utcnow()
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = t.strftime("%Y%m%d")

        # Step 2: Canonical request
        canonical_uri = uri
        canonical_querystring = query
        canonical_headers = f"host:{host}\n" + f"x-amz-date:{amz_date}\n"
        signed_headers = "host;x-amz-date"
        payload_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
        canonical_request = (
            f"{method}\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )

        # Step 3: String to sign
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
        string_to_sign = (
            f"{algorithm}\n{amz_date}\n{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )

        # Step 4: Signing key
        signing_key = _get_signature_key(secret_key, date_stamp, region, service)

        # Step 5: Signature
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Step 6: Authorization header
        authorization_header = (
            f"{algorithm} Credential={access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        headers["x-amz-date"] = amz_date
        headers["Authorization"] = authorization_header

        return {"headers": headers}
