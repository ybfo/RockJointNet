from __future__ import annotations

from scripts.draw_geospin_clean_v3_framework import draw as draw_clean_framework
from scripts.draw_geospin_framework_reviewed import draw as draw_reviewed_framework
from scripts.draw_geospin_pinn_style_v2_framework import draw as draw_final_framework
from scripts.draw_geospin_pinn_style_framework import draw as draw_pinn_framework
from scripts.draw_geospin_framework_simple import draw as draw_framework
from scripts.make_paper_figures import main as make_figures
from scripts.make_paper_tables import main as make_tables
from scripts.make_tnnls_title_page_docx import main as make_title_page_docx
from scripts.write_checkpoint_manifest import main as write_checkpoints


def main() -> None:
    draw_reviewed_framework()
    draw_final_framework()
    draw_clean_framework()
    draw_pinn_framework()
    draw_framework()
    make_figures()
    make_tables()
    make_title_page_docx()
    write_checkpoints()


if __name__ == "__main__":
    main()
