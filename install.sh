#!/usr/bin/env bash
# OpenCode DA-Orchestrator — Unix/macOS Installer
# Copies skills, prompts, and merges agent config into your opencode.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect OpenCode config directory
OCONF="${OP_ENCODE_HOME:-$HOME/.config/opencode}"
OCONF_JSON="$OCONF/opencode.json"
SKILLS_DIR="$OCONF/skills"
PROMPTS_DIR="$OCONF/prompts/da"

echo -e "\033[36m🔧 OpenCode DA-Orchestrator Installer\033[0m"
echo -e "\033[90m   Config dir: $OCONF\033[0m"

# Step 1: Copy skills
echo -e "\n\033[33m📦 Copying DA skills...\033[0m"
DA_SKILLS=("da-eda" "da-cleaning" "da-features" "da-modeling" "da-evaluation" "da-interpreter")
for skill in "${DA_SKILLS[@]}"; do
    src="$SCRIPT_DIR/skills/$skill"
    dst="$SKILLS_DIR/$skill"
    if [[ ! -d "$src" ]]; then
        echo -e "   \033[31m⚠️  Missing: $src — skipping\033[0m"
        continue
    fi
    mkdir -p "$dst"
    cp -r "$src"/* "$dst/"
    echo -e "   \033[32m✅ $skill\033[0m"
done

# Step 2: Copy prompts
echo -e "\n\033[33m📝 Copying DA prompts...\033[0m"
SRC_PROMPTS="$SCRIPT_DIR/prompts/da"
if [[ -d "$SRC_PROMPTS" ]]; then
    mkdir -p "$PROMPTS_DIR"
    cp "$SRC_PROMPTS"/* "$PROMPTS_DIR/"
    echo -e "   \033[32m✅ prompts/da/\033[0m"
else
    echo -e "   \033[31m⚠️  No prompts directory found\033[0m"
fi

# Step 3: Merge agent config into opencode.json
echo -e "\n\033[33m⚙️  Merging agent config into opencode.json...\033[0m"

if [[ ! -f "$OCONF_JSON" ]]; then
    echo -e "   \033[31m❌ opencode.json not found at $OCONF_JSON\033[0m"
    echo -e "   \033[90mCreate one first, then re-run this script.\033[0m"
    exit 1
fi

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo -e "   \033[31m❌ 'jq' is required but not installed.\033[0m"
    echo -e "   Install: brew install jq  |  apt install jq  |  pacman -S jq"
    exit 1
fi

PARTIAL_PATH="$SCRIPT_DIR/opencode.partial.json"
if [[ ! -f "$PARTIAL_PATH" ]]; then
    echo -e "   \033[31m❌ opencode.partial.json not found\033[0m"
    exit 1
fi

# Replace placeholder with actual config home path
TEMP_PARTIAL=$(mktemp)
sed "s|__OP_ENCODE_HOME__|$OCONF|g" "$PARTIAL_PATH" > "$TEMP_PARTIAL"

# Merge: add agents from partial into current, skipping existing ones
ADDED=0
SKIPPED=0
for agent in $(jq -r '.agent | keys[]' "$TEMP_PARTIAL"); do
    if jq -e ".agent.\"$agent\"" "$OCONF_JSON" > /dev/null 2>&1; then
        echo -e "   \033[90m⏭️  $agent already exists — skipping\033[0m"
        ((SKIPPED++)) || true
    else
        jq --arg agent "$agent" --slurpfile partial "$TEMP_PARTIAL" \
           '.agent[$agent] = $partial[0].agent[$agent]' \
           "$OCONF_JSON" > "$OCONF_JSON.tmp" && mv "$OCONF_JSON.tmp" "$OCONF_JSON"
        echo -e "   \033[32m✅ $agent\033[0m"
        ((ADDED++)) || true
    fi
done

rm -f "$TEMP_PARTIAL"

echo -e "\n\033[36m🎉 Done! Added $ADDED agents, skipped $SKIPPED (already existed).\033[0m"
echo -e "\033[90m   Restart OpenCode to load the DA-Orchestrator.\033[0m"
echo -e "\033[90m   Try: /da-start with a CSV file!\033[0m"
