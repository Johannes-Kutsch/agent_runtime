# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body-file body.md`
- **View an issue with comments**: `gh issue view <number> --comments`
- **List issues by search**: `gh issue list --search "<query>"`
- **Add a label**: `gh issue edit <number> --add-label "..."`
- **Remove a label**: `gh issue edit <number> --remove-label "..."`
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Close**: `gh issue close <number>`
- **Link as sub-issue**: use `gh api`
  ```bash
  gh api repos/OWNER/REPO/issues/N/sub_issues --method POST --field sub_issue_id=M
  ```

Infer the repo from `git remote -v` - `gh` does this automatically when run inside a clone.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

