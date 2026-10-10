#!/bin/bash
cd /home/dnz75396/ptyrad/demo
export CUDA_VISIBLE_DEVICES="GPU-03a8aae1-354c-2d7f-d513-2ace87ccacc8,GPU-5c1cb7cd-b1a6-01b3-ef6e-ba1d766f59cb"
export PYTHONPATH=/home/dnz75396/ptyrad-slicepar/src
export PTYRAD_DISABLE_ISS_PRECOND=1 SLICE_P=2
G=/home/dnz75396/ptyrad-slicepar/demo/slicepar
# wait for the transpose gate to release the GPUs
while pgrep -f "gate_transpose.yml" > /dev/null; do sleep 30; done
CARRY_SCHEME=chain CARRY_TRANSPORT=pinned timeout 2100 /home/dnz75396/ptyrad/.venv/bin/python \
  /home/dnz75396/ptyrad-slicepar/demo/run_cmp_modesplit.py slicedist $G/params_gate/gate_pinned.yml \
  > $G/gate_pinned.log 2>&1
echo "gate_pinned rc=$?" >> $G/gates_status.txt
echo DONE_PINNED >> $G/gates_status.txt
