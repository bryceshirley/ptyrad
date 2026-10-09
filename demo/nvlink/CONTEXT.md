# Multi-GPU Born campaign — context refresh (2026-10-09)

One-page state of the multi-GPU / Born-vs-multislice work. Full data:
`demo/bench_results.md` (SSA-SSK.4). Branch: `nvlink-bench` (NOT yet pushed).
Box: 4x A100 80GB PCIe, KVM guest, **no NVLink/P2P** (all-PHB). Pin GPUs by
UUID; `PTYRAD_DISABLE_ISS_PRECOND=1` mandatory for batch-1 Born runs.

## The headline findings, in order

1. **Three parallel axes.** batch-split cancels vs multislice (MS does it
   too); **slice-split** (depth, carry chain) and **mode-split** (probe
   modes) are DIFFERENTIAL — MS slices are sequential, and MS wall-time is
   pmode-insensitive (launch-bound) while Born is pmode-proportional.
2. **Nz=21 (the PSO sample): no multi-GPU config beats 1 GPU.** Floor
   analysis: born f+b 9.7 of 15.7 ms/batch, 4-GPU zero-transfer floor 8.4 ms
   -> max ~8%/iter even on NVLink. 2-GPU slicedist = parity (67.4 vs 64.5
   s/iter); 4-GPU = disaster (105.6). Use 1 GPU per recon at 21 slices.
3. **MS's weakness is its ADJOINT** (sequential autograd replay, superlinear;
   torch.compile makes it worse). Born's cumsum parallelizes fwd AND adjoint.
4. **PCIe crossover vs multislice: Nz ~ 35-40 with 2-GPU slice split.**
   End-to-end (s/iter, 2-GPU slice born6 vs 1-GPU MS): 0.86x @21, 0.96x @32,
   **1.04x @42**, ~1.5x @64 (projection). 2 GPUs optimal: 4-GPU floor = 2-GPU
   floor at Nz<=32-42 (hard ~8.8 ms serial core); 8 GPUs pointless here.
5. **Adaptive order growth composes with the split** and the refit tracks the
   object's strengthening (order keeps climbing as contrast develops).
6. **Never start growth at M=1**: the first epoch (4096 batch-1 updates)
   under ISS imprints a weak-contrast object; even instant adaptation
   (grow_fast) leaves -11% contrast. **Start at 3.** grow3 matches/exceeds
   fixed-M6 (contrast 1.2664 vs 1.2560, loss 0.2923 vs 0.2930) at less cost.
7. **Cap the growth**: uncapped it reached M=9 (119 s/iter est. — slower than
   MS). M=7 already loses the crossover. **n_limit: 6 VERIFIED (SSK.5)**:
   trajectory 3->4->5->6-pinned, avg **80.8 s/iter = 1.15x vs MS** and 1.11x
   vs fixed6; loss 0.2934 == fixed6; contrast 1.2369 (-1.5% vs fixed6 from
   the cheap warmup epochs; identical M=6 model from iter 4).

## THE RECIPE (deep stacks, this hardware)

```yaml
born_iterations: 3
born_coeffs_refit: { start_iter: 1, step: 1, n_views: 32, pin_first: false,
                     method: detector, target: multislice,
                     grow_tol: 1.0e-2, grow_fast: true, n_limit: 6 }
```
run: `SLICE_P=2 PTYRAD_DISABLE_ISS_PRECOND=1 CUDA_VISIBLE_DEVICES=<2 UUIDs> \
      python run_cmp_modesplit.py distsmoke params/<cfg>.yml`

## Infrastructure (all on nvlink-bench branch)

- `src/ptyrad/forward_models/born_dist.py` — `born_forward_dist`: TWO-LEVEL
  SCAN engine (local scans parallel on all GPUs; carries = transfer+add;
  exact 4e-8). Helps Nz>=64 (1.9x f+b); Nz=21 is hop-latency-bound.
- `src/ptyrad/reconstruction.py` — multi-device optimizer auto-fallback
  (foreach/fused Adam breaks on multi-device param groups); LBFGS REMOVED;
  refit `grow_fast` (basis to n_limit, one refit promotes until residual <
  grow_tol).
- `demo/run_cmp_modesplit.py` — roles: baseline | modesplit | slicesplit |
  slicedist(+distsmoke). slicedist = DISTRIBUTED OBJECT: per-GPU depth-block
  leaves (grads + Adam stay on-device), per-block sparse loss,
  gather-constrain-scatter once/iter. `SLICE_P` env (2|4), optional argv[2]
  params path. Losses match stock EXACTLY (4+ digits every iter).
- Benches: `bench_slice_real.py` (real ckpt), `bench_axes_real.py`
  (slice/mode/batch), `bench_slice_nzsweep.py` (NGPU/BORN_M envs),
  `bench_combo_modeslice.py` (PM x PZ, NZ_LIST env), `bench_ms_vs_born_b1.py`
  (MS head-to-head), `bench_slice_compress.py` (codecs). Runner:
  `nvlink/run_nvlink_bench.sh`; hypotheses: `nvlink/NVLINK_PROMPT.md`.
- Params: `demo/params/cmp*.yml` — cmp_{baseline,modesplit,slicesplit,
  slicedist,distsmoke*}, cmp32*, cmp42* (fixed6/grow/grow3/growfast/cap6).

## Gotchas (hard-won)

- Rerunning an unchanged config SILENTLY OVERWRITES its output folder —
  always change `prefix`.
- CPU-resident small tensors (omode_occu!) force pageable H2D that BLOCKS the
  host at chain end and serializes GPU groups (measured 0.9x).
- Time 4-GPU f+b with PRE-PLACED object blocks (moving blocks per step masks
  ~half the speedup). Time competing arms SOLO (concurrent arms contend).
- quarter/fp8 carry codec is UNSAFE on real objects (global scale); half is
  the only safe compressed carry.
- torch.compile: born eager==compiled; MS backward compiled is WORSE;
  cross-device compiled backward hits a dynamo bug -> run eager.
- grow trajectory prints in the log: `Refit born_coeffs at iter N (..., n=M
  grew)` — watch it to see the order path.

## THE RENTAL (user has loaded $15 credit — budget ~4 h max, aim for 2-3 h)

**Pod**: RunPod PyTorch 2.8.0 image, **2x A100 SXM (NVLink)**, 160 GB VRAM,
250 GB RAM, 32 vCPU, 30 GB disk, **$3.58/h billed per ms**, CUDA 12.8/13.0.
Piloted FROM THIS VM + this chat via the RunPod MCP server
(`npx @runpod/mcp-server@latest add`). 2 GPUs is deliberate: SSK showed 2-GPU
slice-split captures the full floor at Nz<=42; do NOT upsize.

**Pilot sequence on the pod (budget-conscious, stop pod when idle!):**
1. Get code onto the pod: clone `nvlink-bench` if pushed, else rsync/scp the
   repo from this VM (branch committed locally). `pip install -e .` (torch is
   in the image; add h5py, torchvision if missing).
2. scp `demo/data/PSO/sample_data_PrScO3.mat` (287 MB) from this VM; the
   21-slice checkpoint (69 MB) only if running bench_slice_real/axes.
3. Preflight: `nvidia-smi topo -m` must show NV* between the 2 GPUs;
   `torch.cuda.can_device_access_peer(0,1)` must be True.
4. Benches (~25 min): `bench_combo_modeslice.py` NZ_LIST=21,32,42,64
   PM=1 PZ=2 (slice-2 real should approach the floor: 8.86 ms f+b @32);
   `bench_slice_nzsweep.py` NGPU=2 for M=6/2/1 (hypothesis: ISS/double
   deep-stack unlock); `bench_ms_vs_born_b1.py` (MS column on this GPU).
5. End-to-end (~30 min): THE RECIPE arm (cmp42grow3cap6_10) + MS arm
   (cmp42_ms1 at NITER 10) — expect crossover to widen from 1.04x toward
   ~1.15-1.2x; verify losses/contrast match this box's values.
6. **COPY ALL RESULTS BACK TO THIS VM** (hard requirement): logs, the
   results_* dir, bench tables, and the two iter-10 checkpoints ->
   `~/ptyrad/demo/nvlink/runpod_results/`. Then append the NVLink column to
   `bench_results.md` and STOP THE POD.

## Open / next

- PUSH the branch (needs user's git credentials):
  `git -C ~/ptyrad push -u origin nvlink-bench`
- NVLink rental (2x A100 SXM ~ $3/h is enough; 8 GPUs pointless <=Nz 42):
  verify slice-2 real -> floor (8.86 ms @32), crossover shift to Nz~30,
  ISS/double deep-stack unlock (SSG.2), born_forward_dist at Nz>=64.
- 100-iter production run of THE RECIPE vs fixed-6: time-to-contrast curve,
  where growth plateaus under the cap.
- Data files NOT in git (sha256 in NVLINK_PROMPT.md): sample_data_PrScO3.mat
  (287 MB), model_iter0100.hdf5 (69 MB).
