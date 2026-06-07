# 12. JIT Compile

Enables PyTorch's JIT (just-in-time) compiler via `compiler_configs: {enable: true}`, which fuses and optimizes GPU kernels at runtime for a measured 1.3–1.9× speedup.

**When to use:** Production runs on Linux or macOS where the one-time compilation warmup is acceptable and maximum throughput is desired. Particularly beneficial for large reconstructions run over many iterations. If the hardware permits, it's almost always better to run in JIT mode for significant speedup.

**Tradeoffs & limitations:** The first epoch incurs a compilation overhead before the speedup takes effect. On Windows, requires the `triton-windows` package. Speedup follows a complicated scaling law with problem size (`Npix`, probe modes, slice count, batch sizes) — small problems see less benefit.

```{literalinclude} 12_jit_compile.yaml
:language: yaml
:linenos:
```
