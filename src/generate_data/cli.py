"""Command-line interface. Installed as `generate-data` (see pyproject) and
also runnable as `python -m generate_data`."""

import argparse

from generate_data.datasets import export_gate, list_datasets, produce_named


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="generate-data",
        description="Simulate 4D-STEM datasets in the PtyRAD h5 contract.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list producible datasets")

    pp = sub.add_parser("produce", help="simulate a dataset into out/<name>.h5")
    pp.add_argument("name", choices=sorted(list_datasets()))
    pp.add_argument("--device", choices=("cpu", "gpu"), default="gpu")

    pc = sub.add_parser("cbed", help="single-position CBED pre-scan check")
    pc.add_argument("sample", nargs="?", choices=("smoke", "a"), default="smoke")
    pc.add_argument("--device", choices=("cpu", "gpu"), default="cpu")

    pg = sub.add_parser("gate", help="export a GT-only gate file for design iteration")
    pg.add_argument("name")
    pg.add_argument("slab_cells", nargs="?", type=int, default=2)
    pg.add_argument("marker_cells", nargs="?", type=int, default=2)
    pg.add_argument("si_depths", nargs="*", type=float, default=[])

    args = p.parse_args(argv)
    if args.cmd == "list":
        for name, desc in list_datasets().items():
            print(f"{name:12s} {desc}")
    elif args.cmd == "produce":
        attrs = produce_named(args.name, device=args.device)
        for k, v in attrs.items():
            print(f"  {k} = {v}")
    elif args.cmd == "cbed":
        from generate_data.check_cbed import run_check

        run_check(args.sample, device=args.device)
    elif args.cmd == "gate":
        export_gate(
            args.name,
            slab_cells=args.slab_cells,
            marker_cells=args.marker_cells,
            si_depths=tuple(args.si_depths),
        )


if __name__ == "__main__":
    main()
