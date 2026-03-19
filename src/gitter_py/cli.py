from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from .core import gitter, gitter_batch
from .io import gitter_read, summary_gitter
from .plotting import plot_gitter


def _plate_format(value: str):
    value = value.strip()
    if "," in value:
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("plate-format must be density (1536) or rows,cols")
        return int(parts[0]), int(parts[1])
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gitter")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Process a single plate image")
    run.add_argument("image_file")
    run.add_argument("--plate-format", type=_plate_format, default="1536")
    run.add_argument("--remove-noise", action="store_true")
    run.add_argument("--inverse", action="store_true")
    run.add_argument("--contrast", type=int, default=None)
    run.add_argument("--fast", type=int, default=None)
    run.add_argument("--grid-save", default=".")
    run.add_argument("--dat-save", default=".")
    run.add_argument("--verbose", choices=["l", "p", "n"], default="l")

    batch = sub.add_parser("batch", help="Process a batch of images")
    batch.add_argument("image_files", nargs="+")
    batch.add_argument("--ref-image-file", default=None)
    batch.add_argument("--plate-format", type=_plate_format, default="1536")
    batch.add_argument("--remove-noise", action="store_true")
    batch.add_argument("--inverse", action="store_true")
    batch.add_argument("--contrast", type=int, default=None)
    batch.add_argument("--fast", type=int, default=None)
    batch.add_argument("--grid-save", default=".")
    batch.add_argument("--dat-save", default=".")
    batch.add_argument("--verbose", choices=["l", "p", "n"], default="l")

    read = sub.add_parser("read", help="Read and summarize a DAT file")
    read.add_argument("dat_file")

    plot = sub.add_parser("plot", help="Plot a DAT file")
    plot.add_argument("dat_file")
    plot.add_argument("--plot-type", choices=["heatmap", "bubble"], default="heatmap")
    plot.add_argument("--title", default="")
    plot.add_argument("--out", default=None)
    plot.add_argument("--show", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        result = gitter(
            image_file=args.image_file,
            plate_format=args.plate_format,
            remove_noise=args.remove_noise,
            inverse=args.inverse,
            contrast=args.contrast,
            fast=args.fast,
            grid_save=args.grid_save,
            dat_save=args.dat_save,
            verbose=args.verbose,
        )
        if isinstance(result, list):
            for idx, plate_df in enumerate(result, start=1):
                print(f"Plate {idx}")
                print(summary_gitter(plate_df))
                print()
        else:
            print(summary_gitter(result))
        return 0

    if args.command == "batch":
        images = args.image_files[0] if len(args.image_files) == 1 else args.image_files
        gitter_batch(
            image_files=images,
            ref_image_file=args.ref_image_file,
            plate_format=args.plate_format,
            remove_noise=args.remove_noise,
            inverse=args.inverse,
            contrast=args.contrast,
            fast=args.fast,
            grid_save=args.grid_save,
            dat_save=args.dat_save,
            verbose=args.verbose,
        )
        return 0

    if args.command == "read":
        dat = gitter_read(args.dat_file)
        print(summary_gitter(dat))
        return 0

    if args.command == "plot":
        dat = gitter_read(args.dat_file)
        fig = plot_gitter(dat, title=args.title, plot_type=args.plot_type)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out, dpi=200, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close(fig)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
