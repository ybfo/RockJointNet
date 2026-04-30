from __future__ import annotations

from scripts.build_figures import main as build_figures
from scripts.build_tables import main as build_tables


def main() -> None:
    build_figures()
    build_tables()


if __name__ == "__main__":
    main()
