#!/usr/bin/env bash
# Sync the `submission` branch with a minimal slice of `main`. Run from a clean
# main checkout. Force-pushes submission.
#
# What is included in submission:
#   docker-compose.yml
#   Dockerfile.api, Dockerfile.lb, haproxy.cfg, .dockerignore
#   package.json, package-lock.json, tsconfig.json
#   src/*.ts  (no src/test/)
#
# What is excluded:
#   docs/, bench/, scripts/, data/, README.md, LICENSE, Makefile, .git/, node_modules/
#
# The grader runs `docker compose up` against the submission branch.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
BRANCH=${BRANCH:-submission}
WORKTREE=${WORKTREE:-/tmp/rinha-submission-$$}

cd "$REPO_ROOT"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree is dirty; commit or stash before syncing"
  exit 1
fi

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git worktree add "$WORKTREE" "$BRANCH"
else
  git worktree add --orphan -b "$BRANCH" "$WORKTREE"
fi

# Wipe the worktree (preserve .git).
find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# Copy the runtime slice.
cp "$REPO_ROOT/docker-compose.yml" "$WORKTREE/"
cp "$REPO_ROOT/Dockerfile.api"    "$WORKTREE/"
cp "$REPO_ROOT/Dockerfile.lb"     "$WORKTREE/"
cp "$REPO_ROOT/haproxy.cfg"       "$WORKTREE/"
cp "$REPO_ROOT/.dockerignore"     "$WORKTREE/"
cp "$REPO_ROOT/package.json"      "$WORKTREE/"
cp "$REPO_ROOT/package-lock.json" "$WORKTREE/"
cp "$REPO_ROOT/tsconfig.json"     "$WORKTREE/"
mkdir -p "$WORKTREE/src"
cp "$REPO_ROOT/src/"*.ts          "$WORKTREE/src/"

MAIN_SHA=$(cd "$REPO_ROOT" && git rev-parse --short HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
  echo "no changes to submission"
else
  git commit -m "sync submission: $TS from main@$MAIN_SHA"
  git push --force-with-lease origin "$BRANCH" || {
    echo
    echo "push failed; ensure the remote is configured:"
    echo "  git remote add origin git@github.com:BertBR/rinha-2026.git"
    exit 1
  }
fi

cd "$REPO_ROOT"
git worktree remove --force "$WORKTREE"

echo "submission branch synced from main@$MAIN_SHA"
