#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 - "${root}/src/extract_dvcs_cff/cli/user.py" \
  "${root}/cpp/partons_bridge/src/main.cpp" <<'PY'
import sys

python_source = open(sys.argv[1], encoding="utf-8").read()
bridge_source = open(sys.argv[2], encoding="utf-8").read()
assert 't_GeV2=float(point["t_GeV2"])' in python_source
assert 'phi_rad=float(point["phi_rad"])' in python_source
assert '0.10 <= float(point["x_b"]) <= 0.50' not in python_source
assert '-0.90 <= float(point["t_GeV2"]) <= -0.10' not in python_source
assert 'kinematics.t < -0.90 || kinematics.t > -0.10' not in bridge_source
assert 'outside exact finite-Q2 DVCS limits' in bridge_source
assert 'kinematics.q2 < q0Squared' in bridge_source
PY

echo "User-controlled physical kinematic contract verification: PASS"
