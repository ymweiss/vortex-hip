# Contributing Guidelines

## Commit Message Format

All commit messages must follow this format:

```
[tag] Brief description of changes

Optional detailed explanation of what was changed and why.
```

### Required Tags

- `[docs]` - Documentation changes
- `[analysis]` - Analysis updates or new analysis
- `[implementation]` - Implementation work
- `[config]` - Configuration changes
- `[refactor]` - Code or documentation restructuring
- `[fix]` - Bug fixes or corrections

### Commit Message Rules

1. **Prefix with tag:** Every commit must start with a relevant tag in square brackets
2. **English only:** All commit messages must be written in English
3. **No AI mentions:** Do not mention AI tools or assistants in commit messages
4. **Track progress:** Commits should represent actual work progress, not tool usage

### Examples

Good:
```
[docs] Add architecture analysis for chipStar OpenCL backend
[implementation] Reorganize documentation into topic-based folders
[fix] Correct SPIR-V translation flow in implementation guide
```

Bad:
```
Initial commit
Update files
Generated with Claude Code
AI-assisted documentation
```

## Documentation Organization

### Folder Structure

```
docs/
├── analysis/           # Analysis of existing implementations
├── implementation/     # Implementation guides and strategies
├── reference/          # Reference documentation (Vortex, etc.)
└── SUMMARY.md          # Project summary
```

### Documentation Rules

1. **No root-level docs:** All documentation must be in appropriate subdirectories
2. **Only README at root:** The only markdown file in the root should be README.md
3. **Topic organization:** Group related documents in their respective folders
4. **Clear naming:** Use descriptive, uppercase filenames with hyphens

### Adding New Documentation

When adding new documentation:

1. Determine the appropriate folder (analysis, implementation, or reference)
2. Use clear, descriptive filenames
3. Update README.md if the document should be listed there
4. Commit with appropriate tag (usually `[docs]`)

## Automated Validation

The repository includes git hooks that automatically validate commits:

### Pre-commit Hook

Runs before each commit to check:

1. **Markdown link validation:** All markdown links `[text](path)` must point to existing files
2. **Root directory check:** No documentation files in root except README.md and CONTRIBUTING.md
3. **File organization:** Files must be in appropriate subdirectories
4. **Naming conventions:** Analysis files should include "ANALYSIS" in filename

### Commit-msg Hook

Validates commit message format:

1. **Tag requirement:** Must start with `[tag]` where tag is lowercase
2. **AI mention check:** Must not mention AI tools or assistants
3. **Language check:** Must be in English
4. **Tag format:** Tags must be lowercase only

### Valid Tags

- `[docs]` - Documentation changes
- `[analysis]` - Analysis updates or new analysis
- `[implementation]` - Implementation work
- `[config]` - Configuration changes
- `[refactor]` - Code or documentation restructuring
- `[fix]` - Bug fixes or corrections

### Hook Behavior

- **Errors:** Will prevent the commit and display issues that must be fixed
- **Warnings:** Will allow the commit but suggest improvements
- The hooks are located in `.git/hooks/` and are already executable
