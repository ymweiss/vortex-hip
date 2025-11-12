#!/bin/bash
# Documentation Link Validator
# Validates all internal file links in markdown documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==================================================================="
echo "  Documentation Link Validator"
echo "==================================================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

ERRORS=0
WARNINGS=0
CHECKED=0

# Find all markdown files in our documentation
find "$PROJECT_ROOT" -name "*.md" -type f | while read -r md_file; do
    # Skip third-party directories
    if [[ "$md_file" == *"/chipStar/"* ]] || [[ "$md_file" == *"/hip/"* ]]; then
        continue
    fi

    echo "Checking: ${md_file#$PROJECT_ROOT/}"

    md_dir="$(dirname "$md_file")"

    # Extract all markdown links: [text](link)
    grep -oP '\[.*?\]\(.*?\)' "$md_file" 2>/dev/null | grep -oP '\(.*?\)' | tr -d '()' | while read -r link; do
        CHECKED=$((CHECKED + 1))

        # Skip external links
        if [[ "$link" =~ ^https?:// ]]; then
            continue
        fi

        # Skip anchors
        if [[ "$link" =~ ^# ]]; then
            continue
        fi

        # Skip environment variables
        if [[ "$link" =~ ^\$|^~ ]]; then
            continue
        fi

        # Resolve relative path
        if [[ "$link" == /* ]]; then
            target="$link"
        else
            target="$md_dir/$link"
        fi

        # Normalize path
        target="$(cd "$(dirname "$target")" 2>/dev/null && pwd)/$(basename "$target")" 2>/dev/null || echo "$target"

        # Check if file exists
        if [[ ! -e "$target" ]]; then
            echo "  [ERROR] Broken link: $link"
            echo "          Target: $target"
            ERRORS=$((ERRORS + 1))
        fi
    done
done

echo ""
echo "==================================================================="
echo "  Validation Summary"
echo "==================================================================="
echo "Links checked: $CHECKED"
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [[ $ERRORS -gt 0 ]]; then
    echo "❌ FAILED: Documentation has broken links"
    exit 1
else
    echo "✅ PASSED: All documentation links are valid"
    exit 0
fi
