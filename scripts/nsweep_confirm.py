"""Confirmation runs: is the born-ms gap at N=12 model error or unconverged?
60 iterations at the two extreme N for both solvers."""
import json, re, subprocess, time, os
DEMO="/home/dnz75396/ptyrad/demo"; OUT=f"{DEMO}/output/paper_extra_tests"
os.chdir(DEMO)
results={}
for solver in ("born","multislice"):
    for N in (2,12):
        tag=f"confirm_{solver}_N{N}"
        y=open(f"params/nsweep_{solver}_N{N}.yml").read()
        y=re.sub(r"('NITER'\s*:\s*)20\b", r"\g<1>60", y)
        y=re.sub(r"('SAVE_ITERS'\s*:\s*)20\b", r"\g<1>60", y)
        lines=[]
        for ln in y.splitlines(keepends=True):
            if ln.lstrip().startswith("'output_dir'"):
                ind=ln[:len(ln)-len(ln.lstrip())]
                ln=ind+f"'output_dir': 'output/paper_extra_tests/{tag}',\n"
            lines.append(ln)
        y="".join(lines)
        pp=f"params/{tag}.yml"; open(pp,"w").write(y)
        t0=time.time()
        rc=subprocess.call(f"python -m ptyrad run --params_path {pp} > {OUT}/{tag}.log 2>&1", shell=True)
        dt=time.time()-t0
        loss=None
        for line in reversed(open(f"{OUT}/{tag}.log",errors="ignore").readlines()):
            m=re.search(r"[Ll]oss[^0-9\-]*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)",line)
            if m: loss=float(m.group(1)); break
        results[tag]=dict(rc=rc,wall_s=round(dt,1),final_loss=loss)
        print(f"[{tag}] rc={rc} wall={dt:.0f}s loss={loss}",flush=True)
        json.dump(results,open(f"{OUT}/confirm_results.json","w"),indent=1)
print("CONFIRM DONE",flush=True)
