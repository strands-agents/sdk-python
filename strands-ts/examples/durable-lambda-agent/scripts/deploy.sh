#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-${1:-us-west-2}}"
STACK_NAME="${STACK_NAME:-strands-durable-agent}"
FUNCTION_NAME="${FUNCTION_NAME:-strands-durable-agent}"
MODEL_ID="${BEDROCK_MODEL_ID:-global.anthropic.claude-sonnet-4-6}"

cd "$(dirname "$0")/.."

for command in node npm sam; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done

npm run build:sdk
npm run build
npm run bundle
npm run validate:template

sam deploy \
  --template-file template.yml \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --capabilities CAPABILITY_IAM \
  --resolve-s3 \
  --confirm-changeset \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    "BedrockModelId=$MODEL_ID" \
    "FunctionName=$FUNCTION_NAME"

echo "Deployment complete: stack=$STACK_NAME, function=$FUNCTION_NAME, region=$REGION"
