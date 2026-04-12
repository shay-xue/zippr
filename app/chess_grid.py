"""
8×8 chess grid: coordinate mapping, target selection, and HTML renderer.

Grid coordinate system:
    Columns A–H  →  x = 0–7
    Rows    1–8  →  y = 0–7

Grep for ``# DIMENSION`` to find hardcoded spatial constants.

Color palette: ZIPPR (#FDFBD4, #D9D7B6, #878672, #545333, #161510, #030302)
"""

from __future__ import annotations

import random
from typing import Optional

from config import GRID_COLS, GRID_ROWS, RANDOM_SEED


def optimal_direction(
    piece_col: int, piece_row: int, target_col: int, target_row: int
) -> Optional[str]:
    """Return the best cardinal direction from piece toward target.

    Picks the axis with the larger distance.  Ties broken: column first.
    Returns None if piece is already on the target.
    """
    dc = target_col - piece_col
    dr = target_row - piece_row

    if dc == 0 and dr == 0:
        return None

    if abs(dc) >= abs(dr):
        return "RIGHT" if dc > 0 else "LEFT"
    return "UP" if dr > 0 else "DOWN"


def pick_random_target(rng: random.Random) -> tuple[int, int]:
    """Return a uniformly random (col, row) on the grid. i.i.d., no suppression."""
    col = rng.randint(0, GRID_COLS - 1)   # DIMENSION
    row = rng.randint(0, GRID_ROWS - 1)   # DIMENSION
    return col, row


def col_label(col: int) -> str:
    """0 → 'A', 7 → 'H'."""
    return chr(ord("A") + col)


def square_label(col: int, row: int) -> str:
    """e.g. (0, 0) → 'A1', (7, 7) → 'H8'."""
    return f"{col_label(col)}{row + 1}"


# ── HTML renderer ─────────────────────────────────────────────────────────────

def render_grid_html(
    piece_col: int,
    piece_row: int,
    target_col: int,
    target_row: int,
    flash: Optional[str] = None,
    grid_size_px: int = 380,         # DIMENSION — total grid width in pixels
) -> str:
    """Return self-contained HTML/CSS for the 8×8 grid.

    Color palette:
        Light squares  — #FDFBD4 (cream)
        Dark squares   — #D9D7B6 (warm grey)
        Target square  — #545333 (olive gold)
        Correct flash  — #030302 (near-black)
        Incorrect flash— #878672 (muted)

    Parameters
    ----------
    flash:
        ``"green"`` for correct, ``"red"`` for incorrect, ``None`` for no flash.
    """
    cell = grid_size_px // GRID_COLS  # DIMENSION — pixel size per square

    rows_html: list[str] = []

    # Render rows top-to-bottom (row 7 at top, row 0 at bottom)
    for row in range(GRID_ROWS - 1, -1, -1):        # DIMENSION
        cells: list[str] = []
        for col in range(GRID_COLS):                  # DIMENSION
            is_target = (col == target_col and row == target_row)
            is_piece = (col == piece_col and row == piece_row)

            # Base colour: light/dark checkerboard (ZIPPR palette)
            if (col + row) % 2 == 0:
                bg = "#FDFBD4"   # cream light square
            else:
                bg = "#D9D7B6"   # warm grey dark square

            # Target highlight
            if is_target:
                if flash == "green":
                    bg = "#030302"   # near-black correct
                elif flash == "red":
                    bg = "#878672"   # muted incorrect
                else:
                    bg = "#545333"   # olive target

            piece_content = ""
            if is_piece:
                piece_color = "#FDFBD4" if is_target else "#161510"
                piece_content = (
                    f'<span style="font-size:26px;color:{piece_color};">♟</span>'
                )

            sq_label = square_label(col, row)
            label_color = "#878672" if (col + row) % 2 == 0 else "#545333"
            if is_target and flash is None:
                label_color = "#D9D7B6"

            cells.append(
                f'<td style="width:{cell}px;height:{cell}px;'
                f"background:{bg};text-align:center;vertical-align:middle;"
                f'position:relative;border:1px solid #D9D7B6;'
                f'transition:background 0.25s ease;">'
                f'<span style="position:absolute;top:2px;left:3px;'
                f'font-size:8px;color:{label_color};font-family:monospace;">'
                f"{sq_label}</span>"
                f"{piece_content}</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # Column labels (A–H)
    col_labels_html = "".join(
        f'<td style="width:{cell}px;text-align:center;'
        f'font-size:11px;font-family:monospace;color:#878672;padding:2px 0;">'
        f"{col_label(c)}</td>"
        for c in range(GRID_COLS)
    )

    style = """
    <style>
    @keyframes target-pulse {
        0%   { box-shadow: inset 0 0 8px rgba(84,83,51,0.4); }
        50%  { box-shadow: inset 0 0 20px rgba(84,83,51,0.8); }
        100% { box-shadow: inset 0 0 8px rgba(84,83,51,0.4); }
    }
    table.chess-grid td { transition: background 0.25s; }
    </style>
    """

    # Wrap in a centering flex div so st.html() renders it centred without
    # needing an outer st.markdown wrapper (which can mangle table HTML).
    html = f"""
    {style}
    <div style="display:flex;justify-content:center;padding:4px 0;">
      <div style="display:inline-block;border:3px solid #545333;border-radius:3px;overflow:hidden;">
        <table class="chess-grid" style="border-collapse:collapse;">
          {"".join(rows_html)}
          <tr style="background:#D9D7B6;">{col_labels_html}</tr>
        </table>
      </div>
    </div>
    """
    return html
