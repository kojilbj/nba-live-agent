#!/bin/bash
# PostToolUse hook (Write|Edit): runs pytest whenever a .py file in this
# project is written or edited, and reports the result back to Claude.
set -euo pipefail

file_path=$(jq -r '.tool_input.file_path // empty')
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;
esac

cd "/Users/watanabekoji/Desktop/nba-live-agent"
output=$(.venv/bin/pytest -q 2>&1) && exit_code=0 || exit_code=$?
summary=$(printf '%s' "$output" | tail -8)

context=$(printf 'pytest after editing %s (exit %s):\n%s' "$file_path" "$exit_code" "$summary")
jq -n --arg ctx "$context" '{hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
