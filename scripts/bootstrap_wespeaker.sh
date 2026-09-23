#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_URL="https://github.com/wenet-e2e/wespeaker.git"
PINNED_COMMIT="9fecd6cb4f47475d01761d87c826298dff4ef18c"
WESPEAKER_DIR="${WESPEAKER_DIR:-${ROOT_DIR}/.deps/wespeaker}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to bootstrap WeSpeaker" >&2
  exit 2
fi

if [[ -e "${WESPEAKER_DIR}" && ! -d "${WESPEAKER_DIR}/.git" ]]; then
  echo "WESPEAKER_DIR exists but is not a Git repository: ${WESPEAKER_DIR}" >&2
  exit 2
fi

if [[ ! -d "${WESPEAKER_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${WESPEAKER_DIR}")"
  git clone --depth 1 "${REPO_URL}" "${WESPEAKER_DIR}"
fi

if ! git -C "${WESPEAKER_DIR}" diff --quiet || \
   ! git -C "${WESPEAKER_DIR}" diff --cached --quiet; then
  echo "Refusing to update dirty WeSpeaker checkout: ${WESPEAKER_DIR}" >&2
  exit 2
fi

CURRENT_COMMIT="$(git -C "${WESPEAKER_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  git -C "${WESPEAKER_DIR}" fetch --depth 1 origin "${PINNED_COMMIT}"
  git -C "${WESPEAKER_DIR}" checkout --detach "${PINNED_COMMIT}"
fi

ACTUAL_COMMIT="$(git -C "${WESPEAKER_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${PINNED_COMMIT}" ]]; then
  echo "WeSpeaker checkout is ${ACTUAL_COMMIT}, expected ${PINNED_COMMIT}" >&2
  exit 2
fi

echo "${WESPEAKER_DIR}"
