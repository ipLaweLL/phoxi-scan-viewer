"""Launch the PhoXi scan viewer."""
import argparse
import logging
from pathlib import Path

from .app import ScanViewer, gui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", nargs="?", type=Path, help="Open a saved PLY or PCD on startup")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = gui.Application.instance
    app.initialize()
    window = ScanViewer(app)
    if args.cloud is not None:
        window.open_file(args.cloud)
    app.run()


if __name__ == "__main__":
    main()
