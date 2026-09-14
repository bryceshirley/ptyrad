"""N-sweep on tBL-WSe2: born vs multislice at fixed total thickness (12 A).

Measures, by reconstruction, the paper's claim that the ISS inter-slice
error GROWS with N (eps1 ~ (Phi^2/2)(1-1/N)) while the multislice operator only
improves with finer slicing. N=1 is the anchor: the two models are identical
there. 20 iterations each, all other hyperparameters exactly the paper runs'.
"""
import json, re, subprocess, time, sys, os

DEMO = "/home/dnz75396/ptyrad/demo"
OUT = f"{DEMO}/output/paper_extra_tests"
NS = [1, 2, 3, 4, 6, 12]
SOLVERS = ["born", "multislice"]
results = {}

os.chdir(DEMO)
for solver in SOLVERS:
    src = open(f"params/tBL_WSe2_reconstruct_minimal_{solver}.yml").read()
    for N in NS:
        tag = f"nsweep_{solver}_N{N}"
        y = src
        y = re.sub(r"('obj_Nlayer'\s*:\s*)12", rf"\g<1>{N}", y)
        y = re.sub(r"('obj_slice_thickness'\s*:\s*)1\b", rf"\g<1>{12.0/N}", y)
        y = re.sub(r"('NITER'\s*:\s*)100", r"\g<1>20", y)
        y = re.sub(r"('SAVE_ITERS'\s*:\s*)10", r"\g<1>20", y)
        lines = []
        for ln in y.splitlines(keepends=True):
            if ln.lstrip().startswith("'output_dir'"):
                indent = ln[:len(ln) - len(ln.lstrip())]
                ln = (indent + f"'output_dir': "
                      f"'output/paper_extra_tests/{tag}',\n")
            lines.append(ln)
        y = "".join(lines)
        pp = f"params/{tag}.yml"
        open(pp, "w").write(y)
        t0 = time.time()
        log = f"{OUT}/{tag}.log"
        rc = subprocess.call(
            f"python -m ptyrad run --params_path {pp} > {log} 2>&1",
            shell=True)
        dt = time.time() - t0
        # final loss: last 'loss' number in the log
        loss = None
        for line in reversed(open(log, errors="ignore").readlines()):
            m = re.search(r"[Ll]oss[^0-9\-]*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", line)
            if m:
                loss = float(m.group(1)); break
        results[tag] = dict(solver=solver, N=N, rc=rc, wall_s=round(dt, 1),
                            s_per_iter=round(dt / 20.0, 2), final_loss=loss)
        print(f"[{tag}] rc={rc} wall={dt:.0f}s loss={loss}", flush=True)
        json.dump(results, open(f"{OUT}/nsweep_results.json", "w"), indent=1)
print("SWEEP DONE", flush=True)
