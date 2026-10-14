#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git clone --depth 1 https://github.com/rcbarke/six-times-to-spare.git "$HERE/reference-six-times-to-spare"
