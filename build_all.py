from __future__ import annotations

from scripts.build_checkpoint_manifest import main as build_checkpoint_manifest
from scripts.build_figures import main as build_figures
from scripts.build_tables import main as build_tables
from scripts.build_title_page_docx import main as build_title_page
from scripts.draw_framework_clean import draw as draw_framework_clean
from scripts.draw_framework_compact import draw as draw_framework_compact
from scripts.draw_framework_main import draw as draw_framework_main
from scripts.draw_framework_pinn import draw as draw_framework_pinn
from scripts.draw_framework_sketch import draw as draw_framework_sketch


def main() -> None:
    draw_framework_main()
    draw_framework_compact()
    draw_framework_clean()
    draw_framework_pinn()
    draw_framework_sketch()
    build_figures()
    build_tables()
    build_title_page()
    build_checkpoint_manifest()


if __name__ == "__main__":
    main()
