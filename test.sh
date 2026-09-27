#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  replogle) exec "$ROOT/scripts/test_replogle.sh" ;;
  pbmc) exec "$ROOT/scripts/test_pbmc.sh" ;;
  tahoe|tahoe100m) exec "$ROOT/scripts/test_tahoe.sh" ;;
  *) echo "usage: $0 {replogle|pbmc|tahoe}" >&2; exit 2 ;;
esac
