#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
STACK_NAME="${STACK_NAME:-strands-durable-agent}"
PROMPT="${PROMPT:-Plan my trip to Seattle.}"
SIMULATE_RESTART=false
CRASH_AFTER_FIRST_TOOL=false
SCENARIO=baseline

case "${1:-}" in
  --restart)
    SIMULATE_RESTART=true
    SCENARIO=restart
    ;;
  --crash)
    CRASH_AFTER_FIRST_TOOL=true
    SCENARIO=crash-after-tool
    ;;
  "") ;;
  *)
    echo "Usage: $0 [--restart|--crash]" >&2
    exit 2
    ;;
esac

FUNCTION_NAME=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionName'].OutputValue | [0]" \
  --output text)

if [[ -z "$FUNCTION_NAME" || "$FUNCTION_NAME" == "None" ]]; then
  echo "FunctionName output not found for stack $STACK_NAME" >&2
  exit 1
fi

EXECUTION_NAME="strands-${SCENARIO}-$(date +%s)"
SESSION_ID="${SESSION_ID:-$EXECUTION_NAME}"
PAYLOAD=$(printf '{"prompt":"%s","sessionId":"%s","simulateRestart":%s,"crashAfterFirstTool":%s}' \
  "$PROMPT" "$SESSION_ID" "$SIMULATE_RESTART" "$CRASH_AFTER_FIRST_TOOL")

echo "Invoking function=$FUNCTION_NAME, region=$REGION, scenario=$SCENARIO, session=$SESSION_ID"
aws lambda invoke \
  --function-name "${FUNCTION_NAME}:\$LATEST" \
  --invocation-type Event \
  --cli-binary-format raw-in-base64-out \
  --durable-execution-name "$EXECUTION_NAME" \
  --payload "$PAYLOAD" \
  --region "$REGION" \
  /dev/stdout >/dev/null

echo "Invocation queued as $EXECUTION_NAME. Press Ctrl-C to stop following logs."
LOG_GROUP="/aws/lambda/$FUNCTION_NAME"
LOG_GROUP_WAIT_SECONDS="${LOG_GROUP_WAIT_SECONDS:-120}"
POLL_INTERVAL_SECONDS=5
ELAPSED_SECONDS=0
RESOLVED_LOG_GROUP=""

while ((ELAPSED_SECONDS < LOG_GROUP_WAIT_SECONDS)); do
  RESOLVED_LOG_GROUP=$(aws logs describe-log-groups \
    --log-group-name-prefix "$LOG_GROUP" \
    --region "$REGION" \
    --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName | [0]" \
    --output text)
  if [[ "$RESOLVED_LOG_GROUP" == "$LOG_GROUP" ]]; then
    break
  fi

  echo "Waiting for CloudWatch log group $LOG_GROUP..."
  sleep "$POLL_INTERVAL_SECONDS"
  ELAPSED_SECONDS=$((ELAPSED_SECONDS + POLL_INTERVAL_SECONDS))
done

if [[ "$RESOLVED_LOG_GROUP" != "$LOG_GROUP" ]]; then
  echo "CloudWatch log group was not available after ${LOG_GROUP_WAIT_SECONDS}s." >&2
  echo "The invocation may still be queued; rerun the command to follow its logs." >&2
  exit 1
fi

aws logs tail "$LOG_GROUP" \
  --region "$REGION" \
  --since 5m \
  --follow
