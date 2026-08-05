#!/bin/bash
# PostToolUse hook (Write/Edit): runs flake8 whenever a .py file in this
# project is written or edited, and reports the result back to the agent.
set -euo pipefail

file_path=$(jq -r '.tool_input.file_path // .tool_input.TargetFile // empty')
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

cd "$(git rev-parse --show-toplevel)"
output=$(.venv/bin/flake8 src tests 2>&1) && exit_code=0 || exit_code=$?
summary=$(printf '%s' "$output" | tail -10)

context=$(printf 'flake8 linting after editing %s (exit %s):\n%s' "$file_path" "$exit_code" "$summary")
jq -n --arg ctx "$context" '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
