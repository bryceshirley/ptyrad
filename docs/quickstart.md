# Quickstart Demo

Here, we provide a quick walkthough of our demo data and params files for *PtyRAD*.

Before running the demo, please check the following:

**Checklist**
1. Install a modern code editor (e.g. [VS Code](https://code.visualstudio.com/))
2. Install a Python environment / package manager (e.g. [miniforge](https://conda-forge.org/download/))
3. Download or clone the *PtyRAD* source code from the [GitHub repo](https://github.com/chiahao3/ptyrad)
4. Create a dedicated python environemnt `(ptyrad)` for *PtyRAD* and activate it in the terminal
5. Install [*PtyRAD*](https://ptyrad.readthedocs.io/en/latest/installation.html) into the `(ptyrad)` environment
6. Download and place the demo datasets to the correct location under `demo/data/`

> 💡 **Note:** You can find the two example datasets (PSO, tBL_WSe2) in the `demo/data/` folder in our [Zenodo record](https://doi.org/10.5281/zenodo.15273176), which contains a full copy of the PtyRAD codebase (v0.1.0b4) as well.

Once you complete the checklist, you're now ready to run a quick demo using one of two interfaces:

**1. Interactive Jupyter interface (Recommended)**

Use `run_ptyrad_quick_example.ipynb` to quickly reconstruct the demo dataset in a Jupyter notebook

**2. Command-line interface**

```bash
# Here we assume working directory is set at `demo/`
ptyrad run --params_path "params/tBL_WSe2_reconstruct.yml" --gpuid 0
```
