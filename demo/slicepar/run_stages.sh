#!/bin/bash
cd /home/dnz75396/ptyrad-slicepar/demo/slicepar
export CUDA_VISIBLE_DEVICES="GPU-03a8aae1-354c-2d7f-d513-2ace87ccacc8,GPU-5c1cb7cd-b1a6-01b3-ef6e-ba1d766f59cb"
export PYTHONPATH=/home/dnz75396/ptyrad-slicepar/src:.
PY=/home/dnz75396/ptyrad/.venv/bin/python
{
timeout 2100 $PY bench_grid.py --n 1024 --arms main > grid_n1024.log 2>&1;  echo "n1024 rc=$?"
timeout 2100 $PY bench_grid.py --n 2048 --arms main > grid_n2048.log 2>&1;  echo "n2048 rc=$?"
timeout 1800 $PY bench_grid.py --n 256 --arms skipz > grid_skipz.log 2>&1;  echo "skipz rc=$?"
timeout 1800 $PY bench_grid.py --n 256 --arms pmode4 > grid_pmode4.log 2>&1; echo "pmode4 rc=$?"
timeout 1800 $PY bench_grid.py --n 256 --arms compiled > grid_compiled_n256.log 2>&1; echo "compiled rc=$?"
echo DONE_ALL_STAGES
} > stages_status.txt 2>&1
