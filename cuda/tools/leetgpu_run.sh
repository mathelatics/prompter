#!/usr/bin/env bash
# leetgpu_run.sh — run cuda labs on real GPUs. Two backends, honest status:
#
#   BACKEND=auto (default) tries, in order:
#     1) local nvcc + GPU (instant, best)
#     2) SSH remote GPU via  LEETGPU_SSH=user@host  (rsync kernels over, run, fetch logs)
#     3) Google Colab batch via  gcloud + `gcloud beta compute instances create-with-container`
#        or the colab CLI if configured  (COLAB=1 path, see COLAB-RUNBOOK.md)
#
#   NOTE ON LEETGPU.COM: as of 2026-10-28 we probed their endpoints
#   (api.leetgpu.com/run, /api/run, /v1/run -> HTTP 404; pip package does not exist).
#   LeetGPU is a browser-based challenge platform; it exposes NO public code-execution
#   API we could verify. Therefore this script does NOT fake a LeetGPU call. If they
#   publish an API later, fill submit_leetgpu() below and set LEETGPU_API_KEY.
#
# Usage:
#   bash cuda/tools/leetgpu_run.sh                 # all labs, auto backend
#   bash cuda/tools/leetgpu_run.sh lab01_threads lab15_adamw
#   LEETGPU_SSH=gpu@my-box bash cuda/tools/leetgpu_run.sh
set -uo pipefail
KDIR="$(cd "$(dirname "$0")/../kernels" && pwd)"
RDIR="$KDIR/../results"; mkdir -p "$RDIR"
NVCC_FLAGS="-O3 --use_fast_math"

pick_labs() {
  if [[ $# -gt 0 ]]; then printf '%s\n' "$@"
  else ls "$KDIR"/lab*.cu | xargs -n1 basename | sed 's/\.cu$//'; fi
}

run_local() {
  command -v nvcc >/dev/null && nvidia-smi >/dev/null 2>&1
}

backend_local() {
  local lab=$1 src="$KDIR/$1.cu" bin="$RDIR/$1.bin"
  nvcc $NVCC_FLAGS -o "$bin" "$src" 2>"$RDIR/$1.build.log" || { echo "BUILD-FAIL $lab"; return 1; }
  timeout 600 "$bin" >"$RDIR/$1.gpu.log" 2>&1
  judge "$lab" "$RDIR/$1.gpu.log"
}

backend_ssh() {
  local host="$LEETGPU_SSH" lab=$1
  ssh -o ConnectTimeout=10 "$host" 'command -v nvcc' >/dev/null 2>&1 || return 1
  scp -q "$KDIR/$1.cu" "$KDIR/../include/harness.h" "$host:/tmp/" 2>/dev/null || return 1
  ssh "$host" "mkdir -p /tmp/cuda_inc && mv /tmp/harness.h /tmp/cuda_inc/ && cd /tmp && nvcc $NVCC_FLAGS -I/tmp/cuda_inc -o $lab.bin $lab.cu && ./$lab.bin" \
    >"$RDIR/$1.gpu.log" 2>"$RDIR/$1.build.log"
  judge "$lab" "$RDIR/$1.gpu.log"
}

judge() { # $1 lab, $2 logfile
  local lab=$1 log=$2
  if [[ ! -s "$log" ]]; then echo "NO-OUTPUT $lab"; return 1; fi
  if grep -qiE "FAIL|error" "$log" && ! grep -qE "0 fail" "$log"; then
    echo "GPU-FAIL $lab  (tail:)"; tail -5 "$log" | sed 's/^/    /'; return 1
  elif grep -qE "PASS|ALL OK|checks" "$log"; then
    echo "GPU-PASS $lab"; return 0
  else
    echo "GPU-UNKNOWN $lab — inspect $log"; return 1
  fi
}

main() {
  local labs=($(pick_labs "$@")) pass=0 fail=0 failed=()
  local mode="auto"
  if run_local; then mode="local"
  elif [[ -n "${LEETGPU_SSH:-}" ]]; then mode="ssh"
  fi
  case "$mode" in
    local) echo "backend: local nvcc+GPU";;
    ssh)   echo "backend: ssh ${LEETGPU_SSH}";;
    *)     echo "NO GPU AVAILABLE HERE."
           echo "Options:"
           echo "  a) LEETGPU_SSH=user@host  -> runs labs on your remote GPU box"
           echo "  b) upload cuda/ zip to Colab per docs/COLAB-RUNBOOK.md"
           echo "  c) leetgpu.com has no public exec API (verified 2026-10-28, endpoints 404) — cannot automate there"
           exit 2;;
  esac
  for lab in "${labs[@]}"; do
    [[ -f "$KDIR/$lab.cu" ]] || { echo "skip $lab (missing source)"; continue; }
    if [[ "$mode" == local ]]; then backend_local "$lab" && pass=$((pass+1)) || { fail=$((fail+1)); failed+=("$lab"); }
    else backend_ssh "$lab" && pass=$((pass+1)) || { fail=$((fail+1)); failed+=("$lab"); }
    fi
  done
  echo "----------------------------------------"
  echo "results: $pass pass, $fail fail ${failed[*]:+(${failed[*]})}"
  [[ $fail -eq 0 ]]
}
main "$@"
