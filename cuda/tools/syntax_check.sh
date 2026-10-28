#!/usr/bin/env bash
# Host-side typecheck of every lab: preprocess away <<<grid,block>>> launches,
# compile as C++ against tools/cuda_stub.h. No GPU or nvcc required.
set -u
cd "$(dirname "$0")/.."
fail=0
for f in kernels/*.cu; do
  n=$(basename "$f")
  tmp=$(mktemp /tmp/chk_XXXXXX.cpp)
  # kernel<<<grid,block,args>>>() -> kernel();  (also handle gridDim vars via stub)
  sed -E 's/<<<[^>]*>>>//g' "$f" | \
    sed -E 's/gridDim\.(x|y|z)/1u/g' > "$tmp"
  if g++ -std=c++14 -fsyntax-only -Wall -Wextra -Iinclude -x c++ \
       -Itools/fakecuda "$tmp" 2> "${tmp}.err"; then
    echo "[ok]   $n"
  else
    echo "[FAIL] $n"; head -30 "${tmp}.err"; fail=1
  fi
  rm -f "$tmp" "${tmp}.err"
done
exit $fail
