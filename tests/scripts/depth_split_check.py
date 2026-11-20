"""PtyRAD's Born model split over depth across processes.

Each rank ALLOCATES AND KEEPS a contiguous chunk of the object slices
and nothing of the others'. ptychomodels's gauge is that chunk, so the
forward and the adjoint run on it unchanged; what crosses a process
boundary during a solve is the boundary plane of the cumulative sum,
one per rank per order, and the probe, which is replicated. No slice
and no slice gradient ever moves.

Both gradient routes are exercised, since the hosts use both: autograd
through the split forward, and ptychomodels's hand-written adjoint. The
loss curve of a 1-rank run is the reference — it holds every slice —
and a split run has to reproduce it.

Run under torch.distributed:

    torchrun --nproc_per_node=2 depth_split_check.py --params <yaml>
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptychomodels import (  # noqa: E402
    amplitude_seed,
    born_backward,
    born_forward,
    centre_chunk,
    chunk_planes,
    cumulative_propagators,
)

from ptyrad.core.models.ptycho import PtychoModel  # noqa: E402
from ptyrad.init import Initializer  # noqa: E402
from ptyrad.params import load_params  # noqa: E402


def keep_own_slices(model, planes):
    """Drop every slice this rank does not own, canvas and all."""
    model.opt_obja = nn.Parameter(model.opt_obja.data[:, planes].clone())
    model.opt_objp = nn.Parameter(model.opt_objp.data[:, planes].clone())


def inputs(model, indices, planes, Nz, background):
    """This rank's patches and the gauge of its own slices."""
    obja, objp = model.get_obj_patches(indices)
    probes = model.get_probes(indices)
    H = cumulative_propagators(model.get_propagators(indices)[:, None, None], Nz)
    O = torch.polar(obja, objp)
    return O, centre_chunk(O, H, planes.start, background), probes


def autograd_step(model, optimizer, indices, planes, Nz, order, background, forward):
    optimizer.zero_grad(set_to_none=True)
    _, gauge, probes = inputs(model, indices, planes, Nz, background)
    measured = model.get_measurements(indices)
    pattern, _ = amplitude_seed(forward(gauge, probes, order), measured)
    loss = (pattern.sqrt() - measured.sqrt()).square().sum() / measured.sum()
    loss.backward()
    optimizer.step()
    return float(loss)


def adjoint_step(model, optimizer, indices, planes, Nz, order, background, forward):
    """The same step from the hand-written adjoint, which writes the
    gradient of the chunk straight into PtyRAD's amplitude and phase
    through one tape over torch.polar."""
    optimizer.zero_grad(set_to_none=True)
    O, gauge, probes = inputs(model, indices, planes, Nz, background)
    measured = model.get_measurements(indices)
    with torch.no_grad():
        pattern, seed = amplitude_seed(forward(gauge, probes.detach(), order), measured)
        grad_O, grad_probe, _ = born_backward(gauge, probes.detach(), order, seed)
    scale = 1.0 / float(measured.sum())  # the data term the autograd route takes
    grad_probe = grad_probe.sum(dim=2)
    if probes.shape[0] == 1 and grad_probe.shape[0] != 1:
        # PtyRAD shares one probe across the batch when the position
        # shifts are fixed, so its gradient is the batch's sum
        grad_probe = grad_probe.sum(dim=0, keepdim=True)
    torch.autograd.backward([O, probes], [grad_O * scale, grad_probe * scale])
    optimizer.step()
    return float((pattern.sqrt() - measured.sqrt()).square().sum() / measured.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--background", default="mean")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--route", choices=["autograd", "adjoint"], default="adjoint")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    dist.init_process_group(backend="nccl")
    rank, num_devices = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    params = load_params(args.params, validate=True)
    init = Initializer(params["init_params"], seed=7).init_all()
    model = PtychoModel(init.init_variables, params["model_params"], device=device)

    Nz = model.opt_objp.shape[1]
    planes = chunk_planes(Nz)
    keep_own_slices(model, planes)
    torch.cuda.reset_peak_memory_stats()
    order = min(args.order, Nz)
    forward = torch.compile(born_forward, fullgraph=True) if args.compile else born_forward

    optimizer = torch.optim.Adam(
        [
            {"params": [model.opt_obja, model.opt_objp], "lr": 5e-4},
            {"params": [model.opt_probe], "lr": 1e-4},
        ]
    )
    step = autograd_step if args.route == "autograd" else adjoint_step
    batches = [
        torch.arange(start, start + args.batch, device=device)
        for start in range(0, args.positions, args.batch)
    ]

    losses, times = [], []
    for _ in range(args.iters):
        torch.cuda.synchronize()
        start = time.time()
        total = sum(
            step(model, optimizer, indices, planes, Nz, order, args.background, forward)
            for indices in batches
        )
        torch.cuda.synchronize()
        times.append(time.time() - start)
        losses.append(total / len(batches))

    held = torch.zeros(num_devices, dtype=torch.int64, device=device)
    held[rank] = model.opt_objp.shape[1]
    dist.all_reduce(held)
    if rank == 0:
        Path(args.out).write_text(
            json.dumps(
                {
                    "num_devices": num_devices,
                    "route": args.route,
                    "compiled": args.compile,
                    "slices": Nz,
                    "slices_per_rank": held.tolist(),
                    "order": order,
                    "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
                    "loss": losses,
                    "seconds_per_iteration": times,
                }
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
