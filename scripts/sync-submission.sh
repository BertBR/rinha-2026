#!/usr/bin/env bash
# Sync the `submission` branch with a minimal slice of `main`. Run from a clean
# main checkout. Force-pushes submission.
#
# What is included in submission:
#   docker-compose.yml
#   Dockerfile.api, Dockerfile.lb, haproxy.cfg, .dockerignore
#   package.json, package-lock.json, tsconfig.json
#   src/*.ts (no tests)
#
# What is excluded:
#   docs/, bench/, scripts/, data/, README.md, LICENSE, Makefile, .git/, node_modules/
#
# The grader runs `docker compose up` against the submission branch, nothing
# more.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
BRANCH=${BRANCH:-submission}
WORKTREE=${WORKTREE:-/tmp/rinha-submission-$$}

cd "$REPO_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty — commit or stash before syncing"
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git worktree add "$WORKTREE" "$BRANCH"
else
  git worktree add --orphan -b "$BRANCH" "$WORKTREE"
  (cd "$WORKTREE" && git rm -rf --quiet . 2>/dev/null || true)
fi

# Copy in the submission slice.
rsync -a --delete --include-from=- \
  --exclude='*' \
  "$REPO_ROOT/" "$WORKTREE/" <<'EOF'
+ docker-compose.yml
+ Dockerfile.api
+ Dockerfile.lb
+ haproxy.cfg
+ .dockerignore
+ package.json
+ package-lock.json
+ tsconfig.json
+ src/
+ src/*.ts
- src/test/
- *
EOF

# Strip test dir if rsync left it
rm -rf "$WORKTREE/src/test"

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
  echo "no changes to submission"
else
  git commit -m "sync submission: $(date -u +%Y-%m-%dT%H:%M:%SZ) from main@$(cd "$REPO_ROOT" && git rev-parse --short HEAD)"
  git push --force-with-lease origin "$BRANCH" || {
    echo
    echo "push failed — likely no remote configured yet."
    echo "set the remote with:  git remote add origin git@github.com:BertBR/rinha-2026.git"
    echo "then re-run this script."
  }
fi

cd "$REPO_ROOT"
git worktree remove --force "$WORKTREE"

echo "submission branch synced."
