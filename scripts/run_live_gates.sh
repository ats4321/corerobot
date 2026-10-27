#!/usr/bin/env bash
# Live gates, in order: smoke tests (+ replay check), limited live eval (t4, t7; seed 100;
# conditions A, B, C), planner on the 10 demos, SAM gate. Prints the cost estimate first and
# asks before spending; reports actual spend after every step and stops at the budget.
#
#   scripts/run_live_gates.sh            # asks for confirmation
#   scripts/run_live_gates.sh --yes      # no prompt
#   BUDGET=3 scripts/run_live_gates.sh   # stop once spend since start exceeds $BUDGET (default 3)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
BUDGET="${BUDGET:-3}"
START=$($PY -c 'import time; print(time.time())')

if ! $PY -c 'import sys; from core_robot.config import load_settings; sys.exit(0 if load_settings().api_key else 1)'; then
  echo "MODEL_API_KEY is not set: add it to .env in $(pwd)"; exit 2
fi

echo "== cost estimate (nothing spent yet) =="
echo "smoke tests: 5 calls, about \$0.05"
$PY -m core_robot.eval.run_eval --generator muse --tasks t4,t7 --seeds 100 --dry-run | tail -1
$PY scripts/eval_planner.py --dry-run
echo "SAM gate: reuses the planner's perception (cache hits, \$0)"
echo "budget for this session: \$$BUDGET"

if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "Proceed with live calls? [y/N] " answer
  [[ "$answer" == y* || "$answer" == Y* ]] || { echo "aborted"; exit 1; }
fi

spend() {
  $PY scripts/spend.py --since "$START" --budget "$BUDGET"
  if [[ $? -eq 3 ]]; then echo "stopping: budget \$$BUDGET exceeded"; exit 3; fi
}
step() {
  local title="$1"; shift
  echo; echo "== $title =="
  "$@"; local rc=$?
  [[ $rc -eq 0 ]] || echo "!! $title exited with $rc (continuing; see output above)"
  spend
}

step "1/5 smoke tests (live)" $PY scripts/smoke_models.py
step "2/5 smoke tests replayed from cache (must be OK x5, no network)" env CORE_ROBOT_REPLAY=1 $PY scripts/smoke_models.py
step "3/5 limited live eval: t4, t7; seed 100; A, B, C" \
  $PY -m core_robot.eval.run_eval --generator muse --tasks t4,t7 --seeds 100
cat data/runs/eval/muse_ground_truth/results.csv 2>/dev/null
step "4/5 planner on 10 demos (gate: strict >= 6/10)" $PY scripts/eval_planner.py
step "5/5 SAM gate (gate: >= 8/10 demos)" $PY scripts/sam_gate.py
echo; echo "== total spend this session =="; $PY scripts/spend.py --since "$START"
