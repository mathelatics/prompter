# Colab run cells (paste each into a code cell)

## Cell 1 — upload & unpack
from google.colab import files
up = files.upload()   # upload colab_labs.zip (36 KB, sits at cuda/colab_labs.zip in repo)
import zipfile; zipfile.ZipFile(list(up)[0]).extractall('/content/labs')
%cd /content/labs

## Cell 2 — build all 18 labs
!for f in lab*.cu; do nvcc -O3 --use_fast_math -o ${f%.cu}.bin $f || echo "BUILD-FAIL $f"; done

## Cell 3 — run + harvest verdicts
!for b in lab*.bin; do echo "=== $b ==="; timeout 300 ./$b; done 2>&1 | tee gpu_results.txt
# Expect every section to end with its PASS line and no FAIL lines.
# Paste gpu_results.txt back into the repo as cuda/results/gpu_results.txt
