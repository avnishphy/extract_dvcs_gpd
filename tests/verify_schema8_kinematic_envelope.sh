#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 - "${root}/src/extract_dvcs_cff/cli/user.py" \
  "${root}/cpp/partons_bridge/src/main.cpp" <<'PY'
import sys

python_source = open(sys.argv[1], encoding="utf-8").read()
bridge_source = open(sys.argv[2], encoding="utf-8").read()
assert '0.10 <= float(point["x_b"]) <= 0.50' in python_source
assert '-0.90 <= float(point["t_GeV2"]) <= -0.10' in python_source
assert 'kinematics.xB < 0.10 || kinematics.xB > 0.50' in bridge_source
assert 'kinematics.t < -0.90 || kinematics.t > -0.10' in bridge_source
assert 'documented "\n                      "schema-8 multi-Q2 domain' in bridge_source
PY

echo "Schema-8 Python/native kinematic envelope verification: PASS"
