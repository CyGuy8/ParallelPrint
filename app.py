from __future__ import annotations

import json
import math
import tempfile
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

# css/head are intentionally passed to gr.Blocks rather than launch(): we run
# via `gradio app.py` (hot-reload), which ignores the __main__ launch() call,
# so moving them there would drop our CSS and head scripts. Silence the Gradio
# 6.0 deprecation notice about that.
warnings.filterwarnings(
    "ignore",
    message=r".*parameters have been moved from the Blocks constructor.*",
    category=UserWarning,
)

import gradio as gr
import numpy as np
import trimesh

from gcode_viewer import (
    build_nozzle_spacing_figure,
    build_parallel_figure,
    build_parallel_gif,
    build_toolpath_figure,
    parse_gcode_path,
)
from stl_slicer import (
    LayerStack,
    calculate_z_levels,
    load_mesh,
    scale_factors_for_target_extents,
    scale_mesh,
    slice_stl_to_layers,
)
from vector_gcode import generate_vector_gcode
from vector_toolpath import (
    LEAD_IN_DIRECTION_CHOICES,
    LEAD_IN_DIRECTION_LEFT,
    LEAD_IN_LINE_AUTO,
    LEAD_IN_LINE_CHOICES,
    RASTER_PATTERN_CHOICES,
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
    ContourSource,
    build_reference_stack,
    group_contour_paths,
    split_layer_stack_grid,
)


SAMPLE_STL_SETS = {
    "Standard Shapes": ("Hollow_Pyramid.stl", "Rounded_Cube_Through_Holes.stl", "halfsphere.stl"),
    "Simple Shapes": ("Simple_Circle.stl", "Simple_Square.stl", "Simple_Triangle.stl"),
    "Multi-Material Demo": (
        "Checkerboard_Cube_1.stl",
        "Checkerboard_Cube_2.stl",
        "Wrapped_Egg_Inside.stl",
        "Wrapped_Egg_Outside.stl",
        "Space_Helmet_Glass.stl",
        "Space_Helmet_Shell.stl",
    ),
}
# Default nozzle per sample file: parts of one multi-material model share a
# nozzle, so loading the demo set forms the assemblies automatically.
SAMPLE_SET_NOZZLES = {
    "Multi-Material Demo": {
        "Checkerboard_Cube_1.stl": 1,
        "Checkerboard_Cube_2.stl": 1,
        "Wrapped_Egg_Inside.stl": 2,
        "Wrapped_Egg_Outside.stl": 2,
        "Space_Helmet_Glass.stl": 3,
        "Space_Helmet_Shell.stl": 3,
    },
}
DEFAULT_SAMPLE_STL_SET = "Standard Shapes"
SAMPLE_STL_DIR = Path(__file__).resolve().parent / "sample_stls"
DEFAULT_TARGET_EXTENTS = (20.0, 20.0, 20.0)
DELETE_SHAPE_COOLDOWN_SECONDS = 1.0
UNIFORM_TARGET_AXES = ("X", "Y", "Z")
SCALE_MODE_TARGET_DIMENSIONS = "Independent X/Y/Z"
SCALE_MODE_UNIFORM_FACTOR = "Keep Proportions"
TARGET_DIMENSION_KEYS = ("target_x", "target_y", "target_z")
FRONT_CAMERA = (90, 80, None)
NOZZLE_LAYOUT_PRESETS = [
    "Custom",
    "One row",
    "One column",
    "2 x 1",
    "1 x 2",
    "2 x 2",
    "3 x 3",
    "2 x 5",
    "5 x 2",
]
APP_CSS = """
.pp-visually-hidden {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    min-width: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: -1px !important;
    border: 0 !important;
    overflow: hidden !important;
    clip-path: inset(50%);
}
.pp-color-cell {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 92px;
}
.pp-color-select {
    width: 100%;
    min-width: 0;
}
.pp-color-current {
    display: inline-block;
    min-width: 46px;
    padding: 1px 5px;
    border: 1px solid rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    font-size: 0.75em;
    text-align: center;
    white-space: nowrap;
}
.pp-swatches {
    display: inline-flex;
    gap: 3px;
    align-items: center;
}
.pp-swatch {
    display: inline-block;
    width: 13px;
    height: 13px;
    border: 1px solid rgba(0, 0, 0, 0.35);
    border-radius: 3px;
    cursor: pointer;
    box-sizing: border-box;
}
.pp-swatch:hover {
    transform: scale(1.3);
}
.pp-swatch.pp-current {
    outline: 2px solid var(--color-accent, #f97316);
    outline-offset: 1px;
}

.gradio-container {
    font-size: 90%;
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
}

.gradio-container .gr-row {
    gap: 0.5rem !important;
}

.gradio-container .gr-form,
.gradio-container .gr-box,
.gradio-container .block {
    padding: 0.4rem !important;
}

.gradio-container .prose {
    margin-bottom: 0.4rem !important;
}

.gcode-shape-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 0.5rem;
    padding: 0.5rem !important;
    min-height: 220px;
}

.gcode-shape-card .prose {
    margin-bottom: 0.25rem !important;
}

.gcode-param-label {
    font-size: 0.8rem;
    font-weight: 600;
    line-height: 1.15;
    margin-bottom: 0.2rem !important;
}

.model3D button[aria-label="Undo"] {
    color: var(--block-label-text-color) !important;
    cursor: pointer !important;
    opacity: 1 !important;
}

.settings-accordion summary,
.settings-accordion button[aria-expanded],
.settings-accordion .label-wrap span {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}

.settings-accordion summary {
    padding-top: 0.55rem !important;
    padding-bottom: 0.55rem !important;
}

#load-sample-stls-button,
#load-sample-stls-button button,
#visualize-nozzle-spacing-button,
#visualize-nozzle-spacing-button button {
    background: #f97316 !important;
    border-color: #ea580c !important;
    color: #ffffff !important;
}

#load-sample-stls-button:hover,
#load-sample-stls-button button:hover,
#visualize-nozzle-spacing-button:hover,
#visualize-nozzle-spacing-button button:hover {
    background: #ea580c !important;
    border-color: #c2410c !important;
}

#load-sample-stls-button:focus-visible,
#load-sample-stls-button button:focus-visible,
#visualize-nozzle-spacing-button:focus-visible,
#visualize-nozzle-spacing-button button:focus-visible {
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.35) !important;
}

#nozzle-spacing-table table tbody tr td:nth-child(-n+2),
#nozzle-grid-spacing-table table tbody tr td:nth-child(-n+2),
#nozzle-spacing-table table thead th,
#nozzle-grid-spacing-table table thead th,
#nozzle-spacing-table [role="columnheader"],
#nozzle-grid-spacing-table [role="columnheader"],
#nozzle-spacing-table [role="gridcell"]:nth-child(4n + 1),
#nozzle-spacing-table [role="gridcell"]:nth-child(4n + 2),
#nozzle-grid-spacing-table [role="gridcell"]:nth-child(4n + 1),
#nozzle-grid-spacing-table [role="gridcell"]:nth-child(4n + 2) {
    background: rgba(243, 244, 246, 0.82) !important;
    color: var(--body-text-color-subdued) !important;
    pointer-events: none;
    user-select: none;
}

#shape-settings-table table tbody tr td:last-child,
#shape-settings-table [role="gridcell"]:nth-child(12n) {
    color: #ffffff !important;
    cursor: pointer;
    font-size: 0 !important;
    text-align: center !important;
    user-select: none;
}

#shape-settings-table table tbody tr td:last-child::after,
#shape-settings-table [role="gridcell"]:nth-child(12n)::after {
    content: "X";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.45rem;
    height: 1.45rem;
    border-radius: 999px;
    background: #dc2626;
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 700;
    line-height: 1;
}

#shape-settings-table table tbody tr td:last-child input,
#shape-settings-table [role="gridcell"]:nth-child(12n) input,
#shape-settings-table table tbody tr td:last-child textarea,
#shape-settings-table [role="gridcell"]:nth-child(12n) textarea {
    display: none !important;
}

/* Nozzle-group summary under the Shape Settings table (filled client-side). */
#pp-group-note {
    padding: 0.4rem 0.3rem 0.1rem;
    font-size: 0.85rem;
    line-height: 1.55;
    color: var(--body-text-color);
}
.pp-group-line {
    display: block;
}
.pp-group-chip {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 6px;
}
.pp-valve-warning {
    color: #dc2626;
}
.pp-port-line {
    color: var(--body-text-color-subdued);
}

/* Stale-G-code warning above the Generate button. */
#gcode-stale-banner p {
    color: #b45309;
    font-weight: 500;
}

/* Raster Pattern + Sweep Buffer: bottom-align so the input boxes line up
   even when their label/hint lines wrap differently. */
#gcode-raster-row {
    align-items: end;
}

/* Narrower columns: header labels ("Pressure (psi)", "Contour Tracing", ...)
   wrap to two lines instead of forcing single-line column widths. Wrapping
   happens ONLY at spaces — words never break mid-word (columns grow to fit
   their longest word instead), whatever font the host renders with. */
#shape-settings-table thead th,
#shape-settings-table thead th button,
#shape-settings-table thead th span {
    white-space: normal !important;
    overflow-wrap: normal !important;
    word-break: keep-all !important;
    hyphens: none;
    line-height: 1.15;
}
#shape-settings-table thead th {
    text-align: center;
    vertical-align: middle;
    padding-left: 0.2rem !important;
    padding-right: 0.2rem !important;
}

#toolpath-anim-controls {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    padding: 0.4rem 0.2rem;
}

#toolpath-anim-controls button,
#toolpath-anim-controls select {
    background: var(--button-secondary-background-fill);
    color: var(--button-secondary-text-color);
    border: 1px solid var(--border-color-primary);
    border-radius: 0.4rem;
    padding: 0.25rem 0.7rem;
    font-size: 0.85rem;
    cursor: pointer;
}

#toolpath-anim-controls button:hover,
#toolpath-anim-controls select:hover {
    border-color: #f97316;
}

#tp-scrub {
    flex: 1 1 140px;
    min-width: 120px;
    accent-color: #f97316;
}

#tp-readout {
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
    flex-basis: 100%;
}

#tp-hint {
    font-size: 0.78rem;
    color: var(--body-text-color-subdued);
    padding: 0 0.2rem 0.3rem;
}

#tube-render-warning {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin-top: -0.3rem !important;
}

/* Keep the per-shape G-code previews at a fixed height with internal scroll
   (gr.Code's max_lines does not constrain the editor in this Gradio build). */
.gcode-view .cm-editor {
    max-height: 320px;
}
.gcode-view .cm-scroller {
    overflow: auto;
}

/* Shrink the empty-state placeholder of the G-code download boxes so they are
   the same compact height before and after Generate G-Code is pressed. */
.gcode-download .empty {
    min-height: 0 !important;
    height: 35px !important;
    padding: 0 !important;
    overflow: hidden !important;
}
.gcode-download .empty svg {
    width: 22px !important;
    height: 22px !important;
}

/* The G-code upload box starts hidden; it is shown client-side only when
   "Upload G-Code file" is selected (see gcode_source.change). */
#gcode-upload-col {
    display: none;
}

/* Parallel-printing animation controls share the look of the single-plot bar. */
#parallel-anim-controls {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    padding: 0.4rem 0.2rem;
}

#parallel-anim-controls button,
#parallel-anim-controls select {
    background: var(--button-secondary-background-fill);
    color: var(--button-secondary-text-color);
    border: 1px solid var(--border-color-primary);
    border-radius: 0.4rem;
    padding: 0.25rem 0.7rem;
    font-size: 0.85rem;
    cursor: pointer;
}

#parallel-anim-controls button:hover,
#parallel-anim-controls select:hover {
    border-color: #f97316;
}

#pp-scrub {
    flex: 1 1 140px;
    min-width: 120px;
    accent-color: #f97316;
}

#pp-readout {
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
    flex-basis: 100%;
}
"""

# Gradio 6.10's gr.Model3D leaves the Undo (reset view) button permanently
# disabled when the value is supplied programmatically — its `has_change_history`
# state only flips on uploads through Model3D's own upload widget. This script
# strips the disabled attribute so clicks reach Svelte's handle_undo, which
# calls reset_camera_position on the underlying canvas.
APP_HEAD = """
<script>
(function () {
    function relayColorChoice(rowIdx, hex) {
        var sink = document.querySelector('#pp-color-sink textarea, #pp-color-sink input');
        var apply = document.querySelector('#pp-color-apply button, button#pp-color-apply, #pp-color-apply');
        if (!sink || !apply) return;
        sink.value = rowIdx + '|' + hex;
        sink.dispatchEvent(new Event('input', { bubbles: true }));
        if (apply.tagName !== 'BUTTON') { apply = apply.querySelector('button') || apply; }
        apply.click();
    }
    // In-table color dropdowns: relay each select's pick to the hidden sink
    // textbox + apply button so the backend updates the shape record. The
    // native popup survives because Color cells' pointer events are
    // swallowed before the dataframe can open its cell editor (see
    // isolateColorCell below).
    document.addEventListener('change', function (event) {
        var el = event.target;
        if (!el || !el.classList || !el.classList.contains('pp-color-select')) return;
        var match = (el.className || '').match(/pp-idx-([0-9]+)/);
        if (!match) return;
        relayColorChoice(match[1], el.value);
    }, true);
    // In-table color swatches (if a cell renders palette chips instead):
    // clicking a chip relays "rowIdx|#hex" the same way.
    document.addEventListener('click', function (event) {
        var el = event.target;
        if (!el || !el.classList || !el.classList.contains('pp-swatch')) return;
        var idxMatch = (el.className || '').match(/pp-idx-([0-9]+)/);
        var hexMatch = (el.className || '').match(/pp-hex-([0-9a-fA-F]{6})/);
        if (!idxMatch || !hexMatch) return;
        event.preventDefault();
        event.stopPropagation();
        relayColorChoice(idxMatch[1], '#' + hexMatch[1].toLowerCase());
    }, true);
    // Header "select all" checkboxes: Gradio's own implementation is buggy
    // (sometimes toggles nothing server-side, sometimes bleeds a stray value
    // into the neighbouring column's first row). Hijack the click and set
    // the whole column through the backend instead.
    function relayBulkBool(columnIndex, value) {
        var sink = document.querySelector('#pp-bulk-sink textarea, #pp-bulk-sink input');
        var apply = document.querySelector('#pp-bulk-apply button, button#pp-bulk-apply, #pp-bulk-apply');
        if (!sink || !apply) return;
        sink.value = columnIndex + '|' + value;
        sink.dispatchEvent(new Event('input', { bubbles: true }));
        if (apply.tagName !== 'BUTTON') { apply = apply.querySelector('button') || apply; }
        apply.click();
    }
    document.addEventListener('click', function (event) {
        var el = event.target;
        if (!el || !el.closest) return;
        if (!el.closest('#shape-settings-table thead')) return;
        var wrap = el.closest('label') || el;
        var cb = (wrap.matches && wrap.matches('input[type=checkbox]'))
            ? wrap
            : (wrap.querySelector ? wrap.querySelector('input[type=checkbox]') : null);
        if (!cb) return;
        event.preventDefault();
        event.stopPropagation();
        var row = cb.closest('tr');
        var cell = cb.closest('th, td');
        if (!row || !cell) return;
        var col = Array.prototype.indexOf.call(row.children, cell);
        // Checkbox pre-click activation: by the time click handlers run the
        // box has ALREADY toggled (preventDefault rolls it back visually),
        // so cb.checked is the state the user is asking for.
        relayBulkBool(col, cb.checked ? 1 : 0);
    }, true);
    function enableUndoButtons(root) {
        (root || document).querySelectorAll('.model3D button[aria-label="Undo"]').forEach(function (btn) {
            if (btn.disabled) {
                btn.disabled = false;
            }
        });
    }
    function suppressDeleteCellEditor(event) {
        var cell = event.target && event.target.closest ? event.target.closest('#shape-settings-table td:last-child, #shape-settings-table [role="gridcell"]:nth-child(12n)') : null;
        if (!cell) return;
        setTimeout(function () {
            if (document.activeElement && cell.contains(document.activeElement)) {
                document.activeElement.blur();
            }
        }, 0);
    }
    // Color cells host clickable palette chips: swallow pointer events before
    // the dataframe's own handlers run, or a click would select the cell and
    // open the raw-HTML cell editor on top of the chips.
    function isolateColorCell(event) {
        var el = event.target;
        if (!el || !el.closest) return;
        var cell = el.closest('td, [role="gridcell"]');
        if (!cell || !cell.querySelector('.pp-color-cell')) return;
        event.stopPropagation();
        if (event.type === 'dblclick') {
            event.preventDefault();
        }
    }
    function start() {
        enableUndoButtons();
        document.addEventListener('focusin', suppressDeleteCellEditor);
        document.addEventListener('pointerdown', isolateColorCell, true);
        document.addEventListener('mousedown', isolateColorCell, true);
        document.addEventListener('touchstart', isolateColorCell, true);
        document.addEventListener('dblclick', isolateColorCell, true);
        // 'click' too: registered AFTER the swatch relay above, so the chip
        // click is applied first, then the dataframe never sees the event.
        document.addEventListener('click', isolateColorCell, true);
        var observer = new MutationObserver(function (mutations) {
            for (var i = 0; i < mutations.length; i++) {
                var m = mutations[i];
                if (m.type === 'attributes' && m.target && m.target.matches && m.target.matches('.model3D button[aria-label="Undo"]')) {
                    if (m.target.disabled) m.target.disabled = false;
                } else if (m.type === 'childList') {
                    enableUndoButtons(m.target);
                }
            }
        });
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['disabled']
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
</script>
<script>
// Nozzle-group highlighting: shapes sharing a nozzle number print as ONE
// multi-material assembly, so their rows get a shared tint + accent stripe,
// and a summary under the table names each assembly. Valve cells used by
// more than one shape turn red (they would dispense simultaneously).
// Pure presentation, recomputed from the rendered table on every re-render.
(function () {
    var NAME_COL = 1, PRESSURE_COL = 5, VALVE_COL = 6, NOZZLE_COL = 7, PORT_COL = 8, COLUMNS = 14;
    var TINTS = ['rgba(31,119,180,0.10)', 'rgba(255,127,14,0.12)', 'rgba(44,160,44,0.10)', 'rgba(148,103,189,0.12)', 'rgba(23,190,207,0.12)'];
    var EDGES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#17becf'];
    var PORT_EDGES = ['#64748b', '#0ea5e9', '#f59e0b', '#10b981'];
    function cellNumber(td) {
        var text = ((td && td.textContent) || '').replace(/[^0-9.\\-]/g, '');
        if (!text) return null;
        var value = parseFloat(text);
        return isNaN(value) ? null : Math.round(value);
    }
    function cellName(td) {
        return ((td && td.textContent) || '').replace(/\\u22ee/g, '').trim();
    }
    function refresh() {
        var container = document.getElementById('shape-settings-table');
        if (!container) return;
        // The component renders rows in TWO tables (one wrapped in the
        // drag-drop <button>, one outside), splitting or DUPLICATING rows
        // between them — and the outer table can hold a stale leftover of a
        // previous render. Counting raw rows duplicated group names and
        // flagged phantom valve conflicts, so rows are deduplicated by
        // their Shape number, preferring the button-wrapped (live) copy;
        // every copy still gets styled.
        var byIdx = {};
        var entries = [];
        Array.prototype.slice.call(container.querySelectorAll('table tbody tr')).forEach(function (tr) {
            var tds = tr.querySelectorAll('td');
            for (var i = 0; i < tds.length; i++) {
                tds[i].style.background = '';
                if (i === 0 || i === PRESSURE_COL || i === PORT_COL) tds[i].style.boxShadow = '';
                if (i === VALVE_COL || i === PRESSURE_COL || i === PORT_COL) tds[i].removeAttribute('title');
            }
            tr.removeAttribute('title');
            if (tds.length < COLUMNS) return;
            var idx = cellNumber(tds[0]);
            if (idx === null) return;
            var fromLive = !!tr.closest('button');
            var entry = byIdx[idx];
            if (!entry) {
                entry = byIdx[idx] = {
                    name: cellName(tds[NAME_COL]),
                    nozzle: cellNumber(tds[NOZZLE_COL]),
                    valve: cellNumber(tds[VALVE_COL]),
                    port: cellNumber(tds[PORT_COL]),
                    pressure: cellName(tds[PRESSURE_COL]),
                    fromLive: fromLive,
                    copies: []
                };
                entries.push(entry);
            } else if (fromLive && !entry.fromLive) {
                entry.name = cellName(tds[NAME_COL]);
                entry.nozzle = cellNumber(tds[NOZZLE_COL]);
                entry.valve = cellNumber(tds[VALVE_COL]);
                entry.port = cellNumber(tds[PORT_COL]);
                entry.pressure = cellName(tds[PRESSURE_COL]);
                entry.fromLive = true;
            }
            entry.copies.push({tr: tr, tds: tds});
        });
        var byNozzle = {}, byValve = {}, byPort = {};
        entries.forEach(function (entry) {
            if (entry.nozzle !== null) (byNozzle[entry.nozzle] = byNozzle[entry.nozzle] || []).push(entry);
            if (entry.valve !== null) (byValve[entry.valve] = byValve[entry.valve] || []).push(entry);
            if (entry.port !== null) (byPort[entry.port] = byPort[entry.port] || []).push(entry);
        });
        var summary = [];
        var groupNozzles = Object.keys(byNozzle).filter(function (n) { return byNozzle[n].length > 1; });
        groupNozzles.sort(function (a, b) { return a - b; });
        groupNozzles.forEach(function (nozzle, gi) {
            var tint = TINTS[gi % TINTS.length];
            var edge = EDGES[gi % EDGES.length];
            var names = [];
            byNozzle[nozzle].forEach(function (entry) {
                entry.copies.forEach(function (copy) {
                    for (var i = 0; i < copy.tds.length; i++) copy.tds[i].style.background = tint;
                    copy.tds[0].style.boxShadow = 'inset 3px 0 0 ' + edge;
                    copy.tr.title = 'Nozzle ' + nozzle + ': these shapes print as one multi-material assembly';
                });
                if (entry.name) names.push(entry.name);
            });
            summary.push(
                '<span class="pp-group-line"><span class="pp-group-chip" style="background:' + edge + '"></span>' +
                'Nozzle ' + nozzle + ': <b>' + names.join(' + ') + '</b> print as one assembly</span>'
            );
        });
        // Port groups: pressure is a PORT property (one regulator per serial
        // port), so shapes sharing a Port always share one pressure. Marked
        // subtly - an underline on the Pressure + Port cells - since the row
        // backgrounds belong to the nozzle assemblies.
        Object.keys(byPort).filter(function (p) { return byPort[p].length > 1; }).sort(function (a, b) { return a - b; }).forEach(function (port, pi) {
            var edge = PORT_EDGES[pi % PORT_EDGES.length];
            var names = [];
            var pressure = null;
            byPort[port].forEach(function (entry) {
                entry.copies.forEach(function (copy) {
                    [PRESSURE_COL, PORT_COL].forEach(function (col) {
                        copy.tds[col].style.boxShadow = 'inset 0 -2px 0 ' + edge;
                        copy.tds[col].title = 'Port ' + port + ': one pressure regulator - these shapes share one pressure';
                    });
                });
                if (entry.name) names.push(entry.name);
                if (pressure === null && entry.pressure) pressure = entry.pressure;
            });
            summary.push(
                '<span class="pp-group-line pp-port-line"><span class="pp-group-chip" style="background:' + edge + '"></span>' +
                'Port ' + port + ': <b>' + names.join(' + ') + '</b> share one pressure regulator' +
                (pressure ? ' (' + pressure + ' psi)' : '') + '</span>'
            );
        });
        Object.keys(byValve).filter(function (v) { return byValve[v].length > 1; }).sort(function (a, b) { return a - b; }).forEach(function (valve) {
            var names = [];
            byValve[valve].forEach(function (entry) {
                entry.copies.forEach(function (copy) {
                    var td = copy.tds[VALVE_COL];
                    td.style.background = 'rgba(220,38,38,0.22)';
                    td.title = 'Valve ' + valve + ' is used by more than one shape - they would dispense together';
                });
                if (entry.name) names.push(entry.name);
            });
            summary.push(
                '<span class="pp-group-line pp-valve-warning">&#9888;&#65039; Valve ' + valve + ' is shared by <b>' +
                names.join('</b> and <b>') + '</b> - they would dispense at the same time. Give each shape its own valve.</span>'
            );
        });
        var note = document.getElementById('pp-group-note');
        if (!note) {
            note = document.createElement('div');
            note.id = 'pp-group-note';
            container.appendChild(note);
        }
        var html = summary.join('');
        // Only touch the DOM when the content changed: the observer watches
        // childList, so an unconditional innerHTML write would loop forever.
        if (note.__ppHtml !== html) {
            note.__ppHtml = html;
            note.innerHTML = html;
        }
        note.style.display = html ? '' : 'none';
    }
    var scheduled = false;
    function schedule() {
        if (scheduled) return;
        scheduled = true;
        // setTimeout, not requestAnimationFrame: rAF never fires in hidden
        // tabs, which would leave the table unstyled after a background
        // refresh (and breaks headless testing).
        setTimeout(function () { scheduled = false; refresh(); }, 50);
    }
    function arm() {
        // Observe the BODY, not the table container: Gradio replaces the
        // container node when it hydrates, which would orphan the observer.
        // Style writes are attribute mutations and the note rewrite is
        // guarded, so observing childList/characterData cannot loop.
        new MutationObserver(schedule).observe(document.body, {childList: true, subtree: true, characterData: true});
        schedule();
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', arm);
    } else {
        arm();
    }
})();
</script>
"""

# Client-side build animation for the G-Code Visualization tab. The rendered
# Plotly figure carries per-point timestamps (cumulative path length) in
# layout.meta.animation; this script reveals the print/travel traces up to a
# moving time cutoff via Plotly.restyle, entirely in the browser. Event
# listeners are delegated from document so they survive Gradio re-renders.
TOOLPATH_ANIM_HEAD = """
<script>
(function () {
    var anim = {
        gd: null, meta: null, cache: null,
        playing: false, scrubbing: false,
        cutoff: 0, speed: 1,
        lastTick: null, lastDraw: 0, raf: null
    };

    function findPlot() {
        var container = document.getElementById('toolpath_plot');
        return container ? container.querySelector('.js-plotly-plot') : null;
    }

    function ensureInit() {
        var gd = findPlot();
        if (!gd || !gd.data || !gd.layout || !gd.layout.meta || !gd.layout.meta.animation) {
            return false;
        }
        var meta = gd.layout.meta.animation;
        if (gd === anim.gd && meta === anim.meta && anim.cache) return true;

        var cache = { printIdx: -1, travelIdx: -1, nozzleIdx: -1 };
        for (var i = 0; i < gd.data.length; i++) {
            var name = gd.data[i].name;
            if (gd.data[i].type === 'mesh3d' && name === 'Print (G1)') cache.printIdx = i;
            else if (gd.data[i].type === 'mesh3d' && name === 'Travel (G0)') cache.travelIdx = i;
            else if (name === 'Nozzle') cache.nozzleIdx = i;
        }
        function snapMesh(idx, t) {
            if (idx < 0 || !t || !t.length) return null;
            var tr = gd.data[idx];
            return {
                i: Array.from(tr.i),
                j: Array.from(tr.j),
                k: Array.from(tr.k),
                t: t
            };
        }
        cache.printMesh = snapMesh(cache.printIdx, meta.print_face_t);
        cache.travelMesh = snapMesh(cache.travelIdx, meta.travel_face_t);
        cache.path = (meta.path_t && meta.path_t.length)
            ? { x: meta.path_x, y: meta.path_y, z: meta.path_z, t: meta.path_t }
            : null;

        // Deduplicated move-boundary timestamps for frame stepping.
        var times = cache.path ? cache.path.t : [];
        cache.times = times.filter(function (v, j) { return j === 0 || v !== times[j - 1]; });

        anim.gd = gd;
        anim.meta = meta;
        anim.cache = cache;
        anim.cutoff = meta.total_length;
        anim.playing = false;
        return true;
    }

    function upperBound(arr, v) {
        var lo = 0, hi = arr.length;
        while (lo < hi) {
            var mid = (lo + hi) >> 1;
            if (arr[mid] <= v) lo = mid + 1; else hi = mid;
        }
        return lo;
    }

    function nozzlePos(path, cutoff) {
        var n = upperBound(path.t, cutoff);
        if (n <= 0) return [path.x[0], path.y[0], path.z[0]];
        if (n >= path.t.length) {
            var m = path.t.length - 1;
            return [path.x[m], path.y[m], path.z[m]];
        }
        var t0 = path.t[n - 1], t1 = path.t[n];
        var f = t1 > t0 ? (cutoff - t0) / (t1 - t0) : 1;
        return [
            path.x[n - 1] + (path.x[n] - path.x[n - 1]) * f,
            path.y[n - 1] + (path.y[n] - path.y[n - 1]) * f,
            path.z[n - 1] + (path.z[n] - path.z[n - 1]) * f
        ];
    }

    function applyCutoff() {
        if (!ensureInit()) return;
        var c = anim.cache, cutoff = anim.cutoff;

        if (c.nozzleIdx >= 0 && c.path) {
            var pos = nozzlePos(c.path, cutoff);
            Plotly.restyle(anim.gd, { x: [[pos[0]]], y: [[pos[1]]], z: [[pos[2]]] }, [c.nozzleIdx]);
        }

        var idxs = [], fis = [], fjs = [], fks = [];
        [['travelMesh', c.travelIdx], ['printMesh', c.printIdx]].forEach(function (pair) {
            var mesh = c[pair[0]], idx = pair[1];
            if (!mesh || idx < 0) return;
            var nf = upperBound(mesh.t, cutoff);
            idxs.push(idx);
            fis.push(mesh.i.slice(0, nf));
            fjs.push(mesh.j.slice(0, nf));
            fks.push(mesh.k.slice(0, nf));
        });
        if (idxs.length) Plotly.restyle(anim.gd, { i: fis, j: fjs, k: fks }, idxs);
        syncUI();
    }

    function syncUI() {
        if (!anim.meta) return;
        var total = anim.meta.total_length || 1;
        var frac = Math.max(0, Math.min(1, anim.cutoff / total));

        var scrub = document.getElementById('tp-scrub');
        if (scrub && !anim.scrubbing) scrub.value = String(Math.round(frac * 1000));

        var readout = document.getElementById('tp-readout');
        if (readout) {
            var text = Math.round(frac * 100) + '% of path';
            var ends = anim.meta.layer_t_end || [];
            if (ends.length) {
                var k = upperBound(ends, anim.cutoff);
                if (k >= ends.length) k = ends.length - 1;
                var start = k > 0 ? ends[k - 1] : 0;
                var span = ends[k] - start;
                var lp = span > 0 ? Math.round(((anim.cutoff - start) / span) * 100) : 100;
                lp = Math.max(0, Math.min(100, lp));
                text = 'Layer ' + (k + 1) + '/' + ends.length + ' \\u00b7 ' + lp + '% \\u2014 ' + text;
            }
            readout.textContent = text;
        }
    }

    function setPlaying(on) {
        anim.playing = on;
        var btn = document.getElementById('tp-play');
        if (btn) btn.innerHTML = on ? '&#9208; Pause' : '&#9654; Play';
        if (on) {
            anim.lastTick = null;
            anim.raf = requestAnimationFrame(tick);
        } else if (anim.raf) {
            cancelAnimationFrame(anim.raf);
            anim.raf = null;
        }
    }

    function tick(ts) {
        if (!anim.playing) return;
        if (!ensureInit()) { setPlaying(false); return; }
        if (anim.lastTick == null) anim.lastTick = ts;
        var dt = (ts - anim.lastTick) / 1000;
        anim.lastTick = ts;

        // 1x speed plays the full build in 60 seconds.
        var rate = (anim.meta.total_length / 60) * anim.speed;
        anim.cutoff = Math.min(anim.meta.total_length, anim.cutoff + dt * rate);

        if (anim.cutoff >= anim.meta.total_length) {
            applyCutoff();
            setPlaying(false);
            return;
        }
        if (ts - anim.lastDraw >= 33) {  // cap redraws at ~30 fps
            anim.lastDraw = ts;
            applyCutoff();
        }
        anim.raf = requestAnimationFrame(tick);
    }

    document.addEventListener('click', function (e) {
        var target = e.target.closest
            ? e.target.closest('#tp-play, #tp-restart, #tp-step-back, #tp-step-fwd')
            : null;
        if (!target) return;
        if (!ensureInit()) return;
        var times = anim.cache.times || [];
        if (target.id === 'tp-play') {
            if (!anim.playing && anim.cutoff >= anim.meta.total_length) anim.cutoff = 0;
            setPlaying(!anim.playing);
        } else if (target.id === 'tp-step-fwd') {
            setPlaying(false);
            var i = upperBound(times, anim.cutoff);
            if (i < times.length) { anim.cutoff = times[i]; applyCutoff(); }
        } else if (target.id === 'tp-step-back') {
            setPlaying(false);
            // First index with times[i] >= cutoff, then step to the boundary before it.
            var lo = 0, hi = times.length;
            while (lo < hi) {
                var mid = (lo + hi) >> 1;
                if (times[mid] < anim.cutoff) lo = mid + 1; else hi = mid;
            }
            anim.cutoff = lo > 0 ? times[lo - 1] : 0;
            applyCutoff();
        } else {
            setPlaying(false);
            anim.cutoff = 0;
            applyCutoff();
        }
    });

    document.addEventListener('input', function (e) {
        if (e.target && e.target.id === 'tp-scrub') {
            if (!ensureInit()) return;
            setPlaying(false);
            anim.scrubbing = true;
            anim.cutoff = (parseFloat(e.target.value) / 1000) * anim.meta.total_length;
            applyCutoff();
            anim.scrubbing = false;
        }
    });

    document.addEventListener('change', function (e) {
        if (e.target && e.target.id === 'tp-speed') {
            anim.speed = parseFloat(e.target.value) || 1;
        }
    });
})();
</script>
"""

TOOLPATH_CONTROLS_HTML = """
<div id="toolpath-anim-controls">
    <button id="tp-restart" type="button" title="Back to start">&#9198;</button>
    <button id="tp-step-back" type="button" title="Step back one move">&#9204;</button>
    <button id="tp-play" type="button">&#9654; Play</button>
    <button id="tp-step-fwd" type="button" title="Step forward one move">&#9205;</button>
    <select id="tp-speed" title="Playback speed">
        <option value="0.25">0.25&times;</option>
        <option value="0.5">0.5&times;</option>
        <option value="1" selected>1&times;</option>
        <option value="2">2&times;</option>
        <option value="5">5&times;</option>
        <option value="10">10&times;</option>
    </select>
    <input id="tp-scrub" type="range" min="0" max="1000" value="1000" step="1"
           title="Build progress">
    <span id="tp-readout">Render a tool path, then press Play.</span>
</div>
<div id="tp-hint">
    Tip: drag the plot to rotate and scroll to zoom &mdash; easiest while the
    animation is paused. The &#9204; / &#9205; buttons step through the path
    one move at a time.
</div>
"""


# Parallel-printing animation engine: drives multiple parts in one plot off a
# shared cumulative-length time axis. Independent of the single-plot engine so
# the G-Code Visualization tab is unaffected. Targets #parallel_plot / #pp-*.
PARALLEL_ANIM_HEAD = """
<script>
(function () {
    var anim = { gd:null, meta:null, cache:null, playing:false, scrubbing:false,
                 cutoff:0, speed:1, lastTick:null, lastDraw:0, raf:null };

    function findPlot() {
        var c = document.getElementById('parallel_plot');
        return c ? c.querySelector('.js-plotly-plot') : null;
    }

    function ensureInit() {
        var gd = findPlot();
        if (!gd || !gd.data || !gd.layout || !gd.layout.meta || !gd.layout.meta.animation) return false;
        var meta = gd.layout.meta.animation;
        if (gd === anim.gd && meta === anim.meta && anim.cache) return true;

        var nameToIdx = {};
        for (var i = 0; i < gd.data.length; i++) nameToIdx[gd.data[i].name] = i;
        function snapMesh(idx, t) {
            if (idx == null || idx < 0 || !t || !t.length) return null;
            var tr = gd.data[idx];
            return { i: Array.from(tr.i), j: Array.from(tr.j), k: Array.from(tr.k), t: t };
        }
        var parts = meta.parts.map(function (p) {
            var printIdx = nameToIdx[p.printName]; if (printIdx == null) printIdx = -1;
            var travelIdx = nameToIdx[p.travelName]; if (travelIdx == null) travelIdx = -1;
            var nozzleIdx = nameToIdx[p.nozzleName]; if (nozzleIdx == null) nozzleIdx = -1;
            return {
                printIdx: printIdx, travelIdx: travelIdx, nozzleIdx: nozzleIdx,
                printMesh: snapMesh(printIdx, p.print_face_t),
                travelMesh: snapMesh(travelIdx, p.travel_face_t),
                path: (p.path_t && p.path_t.length) ? {x:p.path_x, y:p.path_y, z:p.path_z, t:p.path_t} : null
            };
        });
        var times = [];
        meta.parts.forEach(function (p) { if (p.path_t) for (var j=0;j<p.path_t.length;j++) times.push(p.path_t[j]); });
        times.sort(function (a, b) { return a - b; });
        anim.cache = { parts: parts, times: times.filter(function (v, j) { return j === 0 || v !== times[j-1]; }) };
        anim.gd = gd; anim.meta = meta; anim.cutoff = meta.total_length; anim.playing = false;
        return true;
    }

    function upperBound(arr, v) {
        var lo = 0, hi = arr.length;
        while (lo < hi) { var m = (lo + hi) >> 1; if (arr[m] <= v) lo = m + 1; else hi = m; }
        return lo;
    }
    function nozzlePos(path, cutoff) {
        var n = upperBound(path.t, cutoff);
        if (n <= 0) return [path.x[0], path.y[0], path.z[0]];
        if (n >= path.t.length) { var m = path.t.length - 1; return [path.x[m], path.y[m], path.z[m]]; }
        var t0 = path.t[n-1], t1 = path.t[n], f = t1 > t0 ? (cutoff - t0) / (t1 - t0) : 1;
        return [path.x[n-1]+(path.x[n]-path.x[n-1])*f, path.y[n-1]+(path.y[n]-path.y[n-1])*f, path.z[n-1]+(path.z[n]-path.z[n-1])*f];
    }

    function applyCutoff() {
        if (!ensureInit()) return;
        var cutoff = anim.cutoff;
        var meshIdxs=[], fis=[], fjs=[], fks=[], nozIdxs=[], nx=[], ny=[], nz=[];
        anim.cache.parts.forEach(function (pt) {
            [['travelMesh','travelIdx'],['printMesh','printIdx']].forEach(function (pair) {
                var mesh = pt[pair[0]], idx = pt[pair[1]];
                if (!mesh || idx < 0) return;
                var nf = upperBound(mesh.t, cutoff);
                meshIdxs.push(idx); fis.push(mesh.i.slice(0,nf)); fjs.push(mesh.j.slice(0,nf)); fks.push(mesh.k.slice(0,nf));
            });
            if (pt.nozzleIdx >= 0 && pt.path) {
                var pos = nozzlePos(pt.path, cutoff);
                nozIdxs.push(pt.nozzleIdx); nx.push([pos[0]]); ny.push([pos[1]]); nz.push([pos[2]]);
            }
        });
        if (nozIdxs.length) Plotly.restyle(anim.gd, { x:nx, y:ny, z:nz }, nozIdxs);
        if (meshIdxs.length) Plotly.restyle(anim.gd, { i:fis, j:fjs, k:fks }, meshIdxs);
        syncUI();
    }

    function syncUI() {
        if (!anim.meta) return;
        var total = anim.meta.total_length || 1;
        var frac = Math.max(0, Math.min(1, anim.cutoff / total));
        var scrub = document.getElementById('pp-scrub');
        if (scrub && !anim.scrubbing) scrub.value = String(Math.round(frac * 1000));
        var readout = document.getElementById('pp-readout');
        if (readout) readout.textContent = Math.round(frac * 100) + '% of build';
    }

    function setPlaying(on) {
        anim.playing = on;
        var btn = document.getElementById('pp-play');
        if (btn) btn.innerHTML = on ? '&#9208; Pause' : '&#9654; Play';
        if (on) { anim.lastTick = null; anim.raf = requestAnimationFrame(tick); }
        else if (anim.raf) { cancelAnimationFrame(anim.raf); anim.raf = null; }
    }

    function tick(ts) {
        if (!anim.playing) return;
        if (!ensureInit()) { setPlaying(false); return; }
        if (anim.lastTick == null) anim.lastTick = ts;
        var dt = (ts - anim.lastTick) / 1000; anim.lastTick = ts;
        var rate = (anim.meta.total_length / 60) * anim.speed;
        anim.cutoff = Math.min(anim.meta.total_length, anim.cutoff + dt * rate);
        if (anim.cutoff >= anim.meta.total_length) { applyCutoff(); setPlaying(false); return; }
        if (ts - anim.lastDraw >= 33) { anim.lastDraw = ts; applyCutoff(); }
        anim.raf = requestAnimationFrame(tick);
    }

    document.addEventListener('click', function (e) {
        var target = e.target.closest ? e.target.closest('#pp-play, #pp-restart, #pp-step-back, #pp-step-fwd') : null;
        if (!target) return;
        if (!ensureInit()) return;
        var times = anim.cache.times || [];
        if (target.id === 'pp-play') {
            if (!anim.playing && anim.cutoff >= anim.meta.total_length) anim.cutoff = 0;
            setPlaying(!anim.playing);
        } else if (target.id === 'pp-step-fwd') {
            setPlaying(false);
            var i = upperBound(times, anim.cutoff);
            if (i < times.length) { anim.cutoff = times[i]; applyCutoff(); }
        } else if (target.id === 'pp-step-back') {
            setPlaying(false);
            var lo = 0, hi = times.length;
            while (lo < hi) { var mid = (lo + hi) >> 1; if (times[mid] < anim.cutoff) lo = mid + 1; else hi = mid; }
            anim.cutoff = lo > 0 ? times[lo - 1] : 0;
            applyCutoff();
        } else {
            setPlaying(false); anim.cutoff = 0; applyCutoff();
        }
    });
    document.addEventListener('input', function (e) {
        if (e.target && e.target.id === 'pp-scrub') {
            if (!ensureInit()) return;
            setPlaying(false); anim.scrubbing = true;
            anim.cutoff = (parseFloat(e.target.value) / 1000) * anim.meta.total_length;
            applyCutoff(); anim.scrubbing = false;
        }
    });
    document.addEventListener('change', function (e) {
        if (e.target && e.target.id === 'pp-speed') anim.speed = parseFloat(e.target.value) || 1;
    });
})();
</script>
"""

PARALLEL_CONTROLS_HTML = """
<div id="parallel-anim-controls">
    <button id="pp-restart" type="button" title="Back to start">&#9198;</button>
    <button id="pp-step-back" type="button" title="Step back one move">&#9204;</button>
    <button id="pp-play" type="button">&#9654; Play</button>
    <button id="pp-step-fwd" type="button" title="Step forward one move">&#9205;</button>
    <select id="pp-speed" title="Playback speed">
        <option value="0.25">0.25&times;</option>
        <option value="0.5">0.5&times;</option>
        <option value="1" selected>1&times;</option>
        <option value="2">2&times;</option>
        <option value="5">5&times;</option>
        <option value="10">10&times;</option>
    </select>
    <input id="pp-scrub" type="range" min="0" max="1000" value="1000" step="1" title="Build progress">
    <span id="pp-readout">Render the parallel print, then press Play.</span>
</div>
"""


def _format_model_details(
    source_name: str,
    mesh,
    scale_factors: tuple[float, float, float] | None = None,
) -> str:
    extents = mesh.extents
    watertight_status = "yes" if mesh.is_watertight else "no"
    watertight_explanation = (
        "closed solid with no holes or open edges"
        if mesh.is_watertight
        else "mesh has holes or open edges"
    )
    lines = [
        "### Model Details",
        f"- Source: `{source_name}`",
        f"- Dimensions (mm): `X {extents[0]:.3f}, Y {extents[1]:.3f}, Z {extents[2]:.3f}`",
        f"- Footprint (mm): `X {extents[0]:.3f} x Y {extents[1]:.3f}`",
    ]

    if scale_factors and not all(math.isclose(value, 1.0) for value in scale_factors):
        lines.append(
            f"- Scale Factors: `X {scale_factors[0]:.4g}, Y {scale_factors[1]:.4g}, Z {scale_factors[2]:.4g}`"
        )

    lines.extend(
        [
            f"- Faces: `{len(mesh.faces)}`",
            f"- Vertices: `{len(mesh.vertices)}`",
            f"- Watertight ({watertight_explanation}): `{watertight_status}`",
        ]
    )
    return "\n".join(lines)


def _opacity_to_alpha(opacity: float) -> int:
    bounded = max(0.05, min(float(opacity), 1.0))
    return int(round(255 * bounded))


def _resolve_model_opacity(setting: float | bool | None) -> float:
    if isinstance(setting, bool):
        return 0.75 if setting else 1.0
    if setting is None:
        return 1.0
    return max(0.05, min(float(setting), 1.0))


def _resolve_target_extents(
    scale_to_target: bool | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
) -> tuple[float, float, float] | None:
    if not scale_to_target:
        return None

    values = (target_x, target_y, target_z)
    if any(value is None for value in values):
        raise ValueError("Target X, Y, and Z dimensions are required when STL scaling is enabled.")

    target = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in target):
        raise ValueError("Target X, Y, and Z dimensions must be greater than zero.")

    return (target[0], target[1], target[2])


def _axis_index(axis: str | None) -> int:
    normalized = (axis or "X").upper()
    if normalized not in UNIFORM_TARGET_AXES:
        raise ValueError("Uniform target side must be X, Y, or Z.")
    return UNIFORM_TARGET_AXES.index(normalized)


def _resolve_uniform_scale_from_targets(
    mesh: trimesh.Trimesh,
    scale_to_target: bool | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
    anchor_axis: str | None = "X",
) -> float | None:
    if not scale_to_target:
        return None

    targets = (target_x, target_y, target_z)
    anchor_index = _axis_index(anchor_axis)
    target_size = targets[anchor_index]
    if target_size is None:
        raise ValueError("Target side length is required when uniform STL scaling is enabled.")

    target_size = float(target_size)
    if not math.isfinite(target_size) or target_size <= 0:
        raise ValueError("Target side length must be greater than zero.")

    current_size = float(mesh.extents[anchor_index])
    if current_size <= 0:
        axis = UNIFORM_TARGET_AXES[anchor_index]
        raise ValueError(f"Cannot scale uniformly from a zero-sized {axis} extent.")

    return target_size / current_size


def _normalize_scale_mode(scale_mode: str | None) -> str:
    if scale_mode == SCALE_MODE_UNIFORM_FACTOR:
        return SCALE_MODE_UNIFORM_FACTOR
    return SCALE_MODE_TARGET_DIMENSIONS


def _resolve_mesh_scale_factors(
    mesh: trimesh.Trimesh,
    scale_to_target: bool | None,
    scale_mode: str | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
) -> tuple[float, float, float] | None:
    if not scale_to_target:
        return None

    target_extents = _resolve_target_extents(True, target_x, target_y, target_z)
    if target_extents is None:
        return None

    if _normalize_scale_mode(scale_mode) == SCALE_MODE_UNIFORM_FACTOR:
        extents = np.asarray(mesh.extents, dtype=float)
        ratios = np.asarray(target_extents, dtype=float) / extents
        anchor_index = int(np.argmax(np.abs(np.log(ratios))))
        scale = float(ratios[anchor_index])
        return (scale, scale, scale)

    return scale_factors_for_target_extents(mesh, target_extents)


def _uniform_target_extents_from_anchor(
    mesh: trimesh.Trimesh,
    anchor_axis: str | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
) -> tuple[float, float, float]:
    scale = _resolve_uniform_scale_from_targets(
        mesh,
        True,
        target_x,
        target_y,
        target_z,
        anchor_axis=anchor_axis,
    )
    extents = np.asarray(mesh.extents, dtype=float)
    return (
        float(extents[0] * scale),
        float(extents[1] * scale),
        float(extents[2] * scale),
    )


def _dimension_update(current: float | None, target: float) -> dict[str, Any]:
    rounded = round(float(target), 6)
    try:
        if current is not None and math.isclose(float(current), rounded, rel_tol=1e-9, abs_tol=1e-6):
            return gr.update()
    except (TypeError, ValueError):
        pass
    return gr.update(value=rounded)


def _load_model_mesh(
    stl_file: str | Path,
    scale_to_target: bool | None = False,
    scale_mode: str | None = SCALE_MODE_TARGET_DIMENSIONS,
    target_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target_z: float | None = DEFAULT_TARGET_EXTENTS[2],
) -> tuple[trimesh.Trimesh, tuple[float, float, float]]:
    mesh = load_mesh(stl_file)
    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        scale_to_target,
        scale_mode,
        target_x,
        target_y,
        target_z,
    )
    if scale_factors is None:
        return mesh, (1.0, 1.0, 1.0)
    return scale_mesh(mesh, scale_factors), scale_factors


def _viewer_update(model_path: str | None) -> dict[str, Any]:
    return gr.update(value=model_path, camera_position=FRONT_CAMERA)


def sync_uniform_target_dimensions(
    stl_file: str | None,
    scale_to_target: bool | None,
    scale_mode: str | None,
    changed_axis: str,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        not stl_file
        or not scale_to_target
        or _normalize_scale_mode(scale_mode) != SCALE_MODE_UNIFORM_FACTOR
    ):
        return gr.update(), gr.update(), gr.update()

    try:
        mesh = load_mesh(stl_file)
        x_value, y_value, z_value = _uniform_target_extents_from_anchor(
            mesh,
            changed_axis,
            target_x,
            target_y,
            target_z,
        )
    except Exception:
        return gr.update(), gr.update(), gr.update()

    return (
        _dimension_update(target_x, x_value),
        _dimension_update(target_y, y_value),
        _dimension_update(target_z, z_value),
    )


def _build_annotated_scene(mesh: trimesh.Trimesh, opacity: float = 1.0) -> str:
    """Export a GLB containing the mesh, origin axes, and a Z=0 grid plane."""
    scene = trimesh.Scene()
    display_transform = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

    # --- Model (muted orange to match the Gradio theme accent) ---
    model_copy = mesh.copy()
    model_copy.apply_transform(display_transform)
    bounded_opacity = _resolve_model_opacity(opacity)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[230, 150, 90, _opacity_to_alpha(bounded_opacity)],
        alphaMode="OPAQUE" if bounded_opacity >= 0.999 else "BLEND",
        metallicFactor=0.0,
        roughnessFactor=0.6,
    )
    model_copy.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(model_copy, geom_name="model")

    bounds = mesh.bounds
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    extent = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # --- Origin axes (coloured cylinders + cones) ---
    axis_len = extent * 0.4
    axis_radius = extent * 0.008
    cone_radius = axis_radius * 3.5
    cone_height = axis_len * 0.12

    axis_defs = [
        ("X", [1, 0, 0], [255, 50, 50, 255]),
        ("Y", [0, 1, 0], [50, 200, 50, 255]),
        ("Z", [0, 0, 1], [50, 120, 255, 255]),
    ]

    for name, direction, color in axis_defs:
        d = np.array(direction, dtype=float)

        # Cylinder from origin along axis
        cyl = trimesh.creation.cylinder(
            radius=axis_radius, height=axis_len, sections=12
        )
        # Default cylinder is along Z; rotate to desired axis
        midpoint = d * axis_len / 2
        if name == "X":
            cyl.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [0, 1, 0]
            ))
        elif name == "Y":
            cyl.apply_transform(trimesh.transformations.rotation_matrix(
                -np.pi / 2, [1, 0, 0]
            ))
        cyl.apply_translation(midpoint)
        cyl.apply_transform(display_transform)
        cyl.visual = trimesh.visual.ColorVisuals(
            mesh=cyl,
            face_colors=np.tile(color, (len(cyl.faces), 1)),
        )
        scene.add_geometry(cyl, geom_name=f"axis_{name}")

        # Cone arrowhead at tip
        cone = trimesh.creation.cone(
            radius=cone_radius, height=cone_height, sections=12
        )
        if name == "X":
            cone.apply_transform(trimesh.transformations.rotation_matrix(
                np.pi / 2, [0, 1, 0]
            ))
        elif name == "Y":
            cone.apply_transform(trimesh.transformations.rotation_matrix(
                -np.pi / 2, [1, 0, 0]
            ))
        cone.apply_translation(d * (axis_len + cone_height / 2))
        cone.apply_transform(display_transform)
        cone.visual = trimesh.visual.ColorVisuals(
            mesh=cone,
            face_colors=np.tile(color, (len(cone.faces), 1)),
        )
        scene.add_geometry(cone, geom_name=f"cone_{name}")

    # --- Grid plane at z=0 ---
    nice_spacings = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    target_spacing = extent * 0.1
    grid_spacing = min(nice_spacings, key=lambda v: abs(v - target_spacing))

    # Grid extends to cover model footprint plus some margin
    margin = grid_spacing * 2
    gx_min = math.floor((x_min - margin) / grid_spacing) * grid_spacing
    gx_max = math.ceil((x_max + margin) / grid_spacing) * grid_spacing
    gy_min = math.floor((y_min - margin) / grid_spacing) * grid_spacing
    gy_max = math.ceil((y_max + margin) / grid_spacing) * grid_spacing

    grid_color = [160, 160, 160, 100]
    grid_segments: list[list[list[float]]] = []

    # Lines parallel to Y
    x = gx_min
    while x <= gx_max:
        grid_segments.append([[x, gy_min, 0], [x, gy_max, 0]])
        x += grid_spacing
    # Lines parallel to X
    y = gy_min
    while y <= gy_max:
        grid_segments.append([[gx_min, y, 0], [gx_max, y, 0]])
        y += grid_spacing

    if grid_segments:
        grid_path = trimesh.load_path(grid_segments)
        grid_path.apply_transform(display_transform)
        grid_path.colors = np.tile(grid_color, (len(grid_path.entities), 1))
        scene.add_geometry(grid_path, geom_name="grid")

    # Export to GLB (camera angle is set via gr.Model3D camera_position)
    out_path = Path(tempfile.mkdtemp(prefix="model3d_")) / "scene.glb"
    scene.export(str(out_path), file_type="glb")
    return str(out_path)


def load_single_model(
    stl_file: str | None,
    opacity: float = 1.0,
    scale_to_target: bool | None = False,
    scale_mode: str | None = SCALE_MODE_TARGET_DIMENSIONS,
    target_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target_z: float | None = DEFAULT_TARGET_EXTENTS[2],
) -> tuple[str | None, str]:
    if not stl_file:
        return _viewer_update(None), "No model loaded."
    mesh, scale_factors = _load_model_mesh(
        stl_file,
        scale_to_target=scale_to_target,
        scale_mode=scale_mode,
        target_x=target_x,
        target_y=target_y,
        target_z=target_z,
    )
    glb_path = _build_annotated_scene(mesh, opacity=_resolve_model_opacity(opacity))
    return _viewer_update(glb_path), _format_model_details(Path(stl_file).name, mesh, scale_factors)


GCODE_SOURCE_UPLOAD = "Upload G-Code file"
GCODE_SOURCE_PARALLEL = "Parallel print (all shapes)"

# Fixed toolpath-view colors: print color comes from the shape's table color
# (uploads default to orange); travel is always grey.
TOOLPATH_UPLOAD_PRINT_COLOR = "#ff7f0e"
TOOLPATH_TRAVEL_COLOR = "#969696"


PARALLEL_COLOR_CHOICES = [
    ("Orange", "#ff7f0e"), ("Blue", "#1f77b4"), ("Green", "#2ca02c"),
    ("Red", "#d62728"), ("Purple", "#9467bd"), ("Pink", "#e377c2"),
    ("Teal", "#17becf"), ("Yellow", "#ffe119"), ("White", "#ffffff"),
    ("Black", "#000000"),
]
DEFAULT_PARALLEL_COLORS = ("#ff7f0e", "#1f77b4", "#2ca02c")
SHAPE_COLOR_NAMES = [name for name, _hex in PARALLEL_COLOR_CHOICES]
_COLOR_NAME_BY_HEX = {hex_value.lower(): name for name, hex_value in PARALLEL_COLOR_CHOICES}
_COLOR_HEX_BY_NAME = {name.lower(): hex_value for name, hex_value in PARALLEL_COLOR_CHOICES}


def _color_display(value: str | None) -> str:
    """Palette name for a stored color; unknown values show as-is."""
    text = str(value or "").strip()
    return _COLOR_NAME_BY_HEX.get(text.lower(), text)


def apply_color_selection(
    records: list[dict] | None,
    settings_table: Any,
    payload: str | None,
) -> tuple[list[dict], list[list[Any]]]:
    """Apply an in-table color dropdown change ("idx|#hex" from the sink)."""
    records = _apply_shape_settings(records or [], settings_table)
    parts = str(payload or "").split("|")
    if len(parts) >= 2:
        try:
            idx = int(parts[0])
        except (TypeError, ValueError):
            idx = None
        hex_value = parts[1].strip().lower()
        palette = {value.lower() for _name, value in PARALLEL_COLOR_CHOICES}
        if idx is not None and hex_value in palette:
            records = [
                dict(record, color=hex_value)
                if int(record.get("idx", 0) or 0) == idx
                else record
                for record in records
            ]
    return records, _shape_settings_rows(records)


def _color_select_cell(record: dict) -> str:
    """HTML dropdown for a shape's Color cell (markdown-rendered).

    The select carries the record idx as a sanitizer-safe class token; a
    delegated head-script change listener relays picks to the hidden
    sink/apply pair, which updates the record and re-renders the table. The
    closed select shows the current color as its background. Works because
    the head script also swallows pointer events on Color cells before the
    dataframe's handlers run — otherwise the cell editor would open on
    mousedown and kill the native dropdown popup.
    """
    idx = int(record.get("idx", 0) or 0)
    current = str(record.get("color", _default_color(idx))).strip().lower()
    options = "".join(
        '<option value="{hex}"{sel}>{name}</option>'.format(
            hex=hex_value,
            sel=" selected" if hex_value.lower() == current else "",
            name=name,
        )
        for name, hex_value in PARALLEL_COLOR_CHOICES
    )
    text_color = "#ffffff" if current in ("#000000",) else "#000000"
    return (
        '<span class="pp-color-cell">'
        '<select class="pp-color-select pp-idx-{idx}" '
        'style="background-color:{bg};color:{fg}">{options}</select>'
        "</span>"
    ).format(idx=idx, bg=current if current.startswith("#") else "#ffffff",
             fg=text_color, options=options)


def _color_from_cell(cell, fallback: str) -> str:
    """Parse a Color cell: palette name (case-insensitive) or a hex value.

    Anything unrecognized keeps the previous color, so a typo never breaks
    the visualization.
    """
    text = str(cell or "").strip()
    if not text:
        return fallback
    hex_value = _COLOR_HEX_BY_NAME.get(text.lower())
    if hex_value:
        return hex_value
    if text.startswith("#") and len(text) in (4, 7):
        return text
    return fallback


def _group_parts_by_nozzle(parts: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for part in parts:
        grouped.setdefault(_record_nozzle_number(part, int(part.get("idx", 1) or 1)), []).append(part)
    return grouped


def _nozzle_group_bounds(grouped: dict[int, list[dict]], nozzle: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins: list[tuple[float, float, float]] = []
    maxs: list[tuple[float, float, float]] = []
    for part in grouped[nozzle]:
        part_min, part_max = part["parsed"]["bounds"]
        mins.append(part_min)
        maxs.append(part_max)
    return (
        tuple(min(values) for values in zip(*mins)),
        tuple(max(values) for values in zip(*maxs)),
    )


def _grid_default_gap_for_pair(pair_index: int, column_count: int, x_gap: float, y_gap: float) -> tuple[float, float]:
    current_col = pair_index % column_count
    next_col = (pair_index + 1) % column_count
    if next_col > current_col:
        return x_gap, 0.0
    return 0.0, y_gap


def _resolve_nozzle_grid_layout(
    parts: list[dict],
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_individual_spacing: bool | None = False,
    spacing_table: Any | None = None,
) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    offsets: dict[int, tuple[float, float]] = {}
    spacings: list[dict] = []
    if not parts:
        return offsets, spacings

    grouped = _group_parts_by_nozzle(parts)
    ordered_nozzles = sorted(grouped)
    column_count = max(1, _coerce_int(columns, 1))
    requested_rows = max(1, _coerce_int(rows, 1))
    row_count = max(requested_rows, math.ceil(len(ordered_nozzles) / column_count))
    x_gap = _coerce_float(column_spacing, 0.0)
    y_gap = _coerce_float(row_spacing, 0.0)

    placements = {
        nozzle: (index % column_count, index // column_count)
        for index, nozzle in enumerate(ordered_nozzles)
    }
    column_widths = [0.0 for _ in range(column_count)]
    row_heights = [0.0 for _ in range(row_count)]
    bounds_by_nozzle: dict[int, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    for nozzle, (column, row) in placements.items():
        bounds = _nozzle_group_bounds(grouped, nozzle)
        bounds_by_nozzle[nozzle] = bounds
        (xmin, ymin, _), (xmax, ymax, _) = bounds
        column_widths[column] = max(column_widths[column], xmax - xmin)
        row_heights[row] = max(row_heights[row], ymax - ymin)

    if use_individual_spacing:
        gap_pairs = _spacing_pairs_from_table(spacing_table)

        def pair_gap(pair_index: int) -> tuple[float, float]:
            if pair_index < len(gap_pairs):
                return gap_pairs[pair_index]
            return _grid_default_gap_for_pair(pair_index, column_count, x_gap, y_gap)

        row_start_x = 0.0
        row_min_y = 0.0
        for row in range(row_count):
            row_start_index = row * column_count
            row_nozzles = ordered_nozzles[row_start_index:row_start_index + column_count]
            if not row_nozzles:
                break
            for col, nozzle in enumerate(row_nozzles):
                (xmin, ymin, _), _ = bounds_by_nozzle[nozzle]
                if col == 0:
                    target_x = row_start_x
                    target_y = row_min_y
                else:
                    prev = row_nozzles[col - 1]
                    gap_x, gap_y = pair_gap(row_start_index + col - 1)
                    prev_offset_x, _prev_offset_y = offsets[prev]
                    (_, _, _), (prev_xmax, _prev_ymax, _) = bounds_by_nozzle[prev]
                    target_x = prev_offset_x + prev_xmax + gap_x
                    target_y = row_min_y + gap_y
                offsets[nozzle] = (target_x - xmin, target_y - ymin)

            row_bottom = max(offsets[nozzle][1] + bounds_by_nozzle[nozzle][1][1] for nozzle in row_nozzles)
            next_row_start_index = row_start_index + len(row_nozzles)
            if next_row_start_index < len(ordered_nozzles):
                row_shift_x, row_gap = pair_gap(next_row_start_index - 1)
                row_start_x += row_shift_x
                row_min_y = row_bottom + row_gap

        for first, second in zip(ordered_nozzles, ordered_nozzles[1:]):
            first_x, first_y = offsets[first]
            second_x, second_y = offsets[second]
            spacings.append({
                "from": first,
                "to": second,
                "dx": second_x - first_x,
                "dy": second_y - first_y,
            })
        return offsets, spacings

    column_positions: list[float] = []
    x_pos = 0.0
    for width in column_widths:
        column_positions.append(x_pos)
        x_pos += width + x_gap

    row_positions: list[float] = []
    y_pos = 0.0
    for height in row_heights:
        row_positions.append(y_pos)
        y_pos += height + y_gap

    for nozzle, (column, row) in placements.items():
        (xmin, ymin, _), _ = bounds_by_nozzle[nozzle]
        offsets[nozzle] = (column_positions[column] - xmin, row_positions[row] - ymin)

    for first, second in zip(ordered_nozzles, ordered_nozzles[1:]):
        first_x, first_y = offsets[first]
        second_x, second_y = offsets[second]
        spacings.append({
            "from": first,
            "to": second,
            "dx": second_x - first_x,
            "dy": second_y - first_y,
        })
    return offsets, spacings


def _format_shape_dimensions(parts: list[dict]) -> list[str]:
    lines = ["**Shape dimensions from generated G-code:**"]
    for part in sorted(parts, key=lambda item: item["idx"]):
        (xmin, ymin, zmin), (xmax, ymax, zmax) = part["parsed"]["bounds"]
        nozzle = _record_nozzle_number(part, int(part.get("idx", 1) or 1))
        lines.append(
            f"Shape {part['idx']} (Nozzle {nozzle}): X {xmax - xmin:.2f} mm, "
            f"Y {ymax - ymin:.2f} mm, Z {zmax - zmin:.2f} mm."
        )
    return lines


def _format_nozzle_spacing_status(
    parts: list[dict],
    offsets: dict[int, tuple[float, float]],
    spacings: list[dict],
) -> str:
    lines = _format_shape_dimensions(parts)
    ordered_nozzles = sorted(offsets)
    if ordered_nozzles:
        lines.append("**Nozzle coordinates:**")
        for idx in ordered_nozzles:
            x, y = offsets[idx]
            lines.append(f"Nozzle {idx}: X {x:.2f} mm, Y {y:.2f} mm.")

    if not spacings:
        lines.append("Generate G-code for at least two nozzles to calculate nozzle spacing.")
        return "  \n".join(lines)

    lines.append("**Nozzle-to-nozzle distances:**")
    for first_pos, first_idx in enumerate(ordered_nozzles):
        for second_idx in ordered_nozzles[first_pos + 1:]:
            x0, y0 = offsets[first_idx]
            x1, y1 = offsets[second_idx]
            dx = x1 - x0
            dy = y1 - y0
            distance = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx)) if distance else 0.0
            lines.append(
                f"Nozzle {first_idx} -> {second_idx}: "
                f"Delta X {dx:.2f} mm, Delta Y {dy:.2f} mm; "
                f"distance {distance:.2f} mm at {angle:.1f} deg."
            )

    lines.append("**Adjacent spacing inputs:**")
    for spacing in spacings:
        lines.append(
            f"Nozzle {spacing['from']} -> {spacing['to']}: "
            f"X {spacing['dx']:.2f} mm, Y {spacing['dy']:.2f} mm."
        )
    return "  \n".join(lines)


SHAPE_SETTINGS_HEADERS = [
    "Shape",
    "STL",
    "X (mm)",
    "Y (mm)",
    "Z (mm)",
    "Pressure (psi)",
    "Valve",
    "Nozzle",
    "Port",
    "Color",
    "Infill %",
    "Contour Tracing",
    "Lead In",
    "Delete",
]
SHAPE_SETTINGS_DATATYPES = [
    "number",
    "str",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "markdown",
    "number",
    "bool",
    "bool",
    "str",
]
ADVANCED_NOZZLE_SPACING_HEADERS = [
    "From Nozzle",
    "To Nozzle",
    "X edge spacing (mm)",
    "Y nozzle spacing (mm)",
]


def _normalise_rows(table: Any) -> list[list[Any]]:
    if table is None:
        return []
    if hasattr(table, "values") and hasattr(table, "columns"):
        return table.values.tolist()
    if isinstance(table, dict) and "data" in table:
        return table.get("data") or []
    return list(table or [])


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on", "checked"}:
        return True
    if text in {"0", "false", "no", "n", "off", "unchecked"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        try:
            return int(float(default))
        except (TypeError, ValueError):
            return 0


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _record_nozzle_number(record: dict, fallback: int | None = None) -> int:
    default = fallback if fallback is not None else int(record.get("idx", 1) or 1)
    nozzle = _coerce_int(record.get("nozzle", default), default)
    return nozzle if nozzle > 0 else default


def _file_path_value(file_value: Any) -> str | None:
    if not file_value:
        return None
    if isinstance(file_value, (str, Path)):
        return str(file_value)
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name") or file_value.get("orig_name")
    return getattr(file_value, "name", None) or getattr(file_value, "path", None)


def _uploaded_file_paths(files: Any) -> list[str]:
    if not files:
        return []
    values = files if isinstance(files, list) else [files]
    paths = [_file_path_value(value) for value in values]
    return [str(path) for path in paths if path]


def _default_color(index: int) -> str:
    return DEFAULT_PARALLEL_COLORS[(index - 1) % len(DEFAULT_PARALLEL_COLORS)]


def _default_target_extents_for_stl(path: str) -> tuple[float, float, float]:
    try:
        extents = load_mesh(path).extents
        values = tuple(float(value) for value in extents)
        if len(values) == 3 and all(math.isfinite(value) and value > 0 for value in values):
            return values
    except Exception:
        pass
    return DEFAULT_TARGET_EXTENTS


def _shape_choice(record: dict) -> str:
    return f"{record['idx']}: {record['name']}"


def _selected_record_index(records: list[dict], selected: str | None) -> int:
    if not records:
        return -1
    try:
        idx = int(str(selected or "").split(":", 1)[0])
    except ValueError:
        idx = records[0].get("idx", 1)
    for pos, record in enumerate(records):
        if record.get("idx") == idx:
            return pos
    return 0


def _next_unused_nozzle(used_nozzles: set[int]) -> int:
    nozzle = 1
    while nozzle in used_nozzles:
        nozzle += 1
    return nozzle


def _next_unused_valve(used_valves: set[int]) -> int:
    """New shapes default to their own valve (numbering starts at the
    historical default, 4): shapes sharing a valve dispense simultaneously,
    which is almost never intended."""
    valve = 4
    while valve in used_valves:
        valve += 1
    return valve


def _records_from_files(files: Any, previous_records: list[dict] | None = None) -> list[dict]:
    previous_by_path: dict[str | None, list[dict]] = {}
    previous_by_name: dict[str, list[dict]] = {}
    for record in previous_records or []:
        previous_by_path.setdefault(record.get("stl_path"), []).append(record)
        if record.get("stl_path"):
            previous_by_name.setdefault(Path(str(record["stl_path"])).name, []).append(record)
    matched_ids: set[int] = set()

    def _take_previous(path: str) -> dict:
        """Match by exact path first, then by FILENAME: Gradio copies
        uploads into its temp cache, so the same file re-arrives under a new
        path — without the name fallback every re-sync treated all shapes
        as brand-new, wiping nozzles/valves (and duplicating rows)."""
        for queue in (previous_by_path.get(path), previous_by_name.get(Path(path).name)):
            while queue:
                candidate = queue.pop(0)
                if id(candidate) not in matched_ids:
                    matched_ids.add(id(candidate))
                    return candidate
        return {}

    used_nozzles: set[int] = set()
    used_valves: set[int] = {
        _coerce_int(record.get("valve"), 0) for record in (previous_records or [])
    }
    records: list[dict] = []
    for index, path in enumerate(_uploaded_file_paths(files), start=1):
        previous = _take_previous(str(path))
        name = previous.get("name") or Path(path).stem or f"Shape {index}"
        default_x, default_y, default_z = _default_target_extents_for_stl(path)
        nozzle = _record_nozzle_number(previous, index) if previous else _next_unused_nozzle(used_nozzles)
        used_nozzles.add(nozzle)
        valve = _coerce_int(previous.get("valve"), 0) if previous else 0
        if valve <= 0:
            valve = _next_unused_valve(used_valves)
        used_valves.add(valve)
        # Pressure is a port property (one regulator per serial port): a new
        # shape adopts the pressure other shapes already use on its port.
        pressure = previous.get("pressure")
        if pressure is None:
            port = _coerce_int(previous.get("port"), 1)
            port_mates = [
                record
                for record in (previous_records or [])
                if _coerce_int(record.get("port"), 1) == port
            ]
            pressure = port_mates[0].get("pressure", 25.0) if port_mates else 25.0
        records.append({
            "idx": index,
            "name": name,
            "stl_path": path,
            # Dimensions live on a 0.1 mm grid (originals included, so a
            # pristine row's target/original ratios are exactly 1).
            "original_x": round(_coerce_float(previous.get("original_x"), default_x), 1),
            "original_y": round(_coerce_float(previous.get("original_y"), default_y), 1),
            "original_z": round(_coerce_float(previous.get("original_z"), default_z), 1),
            "target_x": round(_coerce_float(previous.get("target_x"), default_x), 1),
            "target_y": round(_coerce_float(previous.get("target_y"), default_y), 1),
            "target_z": round(_coerce_float(previous.get("target_z"), default_z), 1),
            "last_scaled_axis": previous.get("last_scaled_axis", "target_x"),
            "pressure": pressure,
            "valve": valve,
            "nozzle": nozzle,
            "port": previous.get("port", 1),
            "color": previous.get("color", _default_color(index)),
            "infill": previous.get("infill", 100.0),
            "contour_tracing": previous.get("contour_tracing", False),
            "lead_in": previous.get("lead_in", False),
            "layer_stack": previous.get("layer_stack"),
            "slice_params": previous.get("slice_params"),
            "gcode_path": previous.get("gcode_path"),
            # Carried through a re-sync: the toolpath's world anchor (Auto
            # Align and the visualizations read it) and the generation
            # fingerprint (the stale-G-code banner compares against it).
            "path_origin": previous.get("path_origin"),
            "gcode_snapshot": previous.get("gcode_snapshot"),
        })
    return records


def _reindex_shape_records(records: list[dict]) -> list[dict]:
    reindexed: list[dict] = []
    for index, record in enumerate(records, start=1):
        copy = dict(record)
        copy["idx"] = index
        reindexed.append(copy)
    return reindexed


def _round_targets_to_tenths(record: dict) -> dict:
    """Snap a record's dimensions (targets AND originals) to the 0.1 mm grid.

    Table dimensions are DISPLAYED to the tenths place, and the table's
    .change echo re-applies whatever is displayed — so the stored values
    must round identically or every echo would look like a fresh edit and
    the convergence guards would ping-pong. Originals round too: a pristine
    row's target/original ratios must be exactly 1, or legacy noisy extents
    (e.g. 32.99557) would break the odd-one-out edit anchoring."""
    for key in (*TARGET_DIMENSION_KEYS, "original_x", "original_y", "original_z"):
        try:
            value = float(record.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            record[key] = round(value, 1)
    return record


def _shape_settings_rows(records: list[dict]) -> list[list[Any]]:
    return [
        [
            record["idx"],
            record["name"],
            round(_coerce_float(record.get("target_x"), DEFAULT_TARGET_EXTENTS[0]), 1),
            round(_coerce_float(record.get("target_y"), DEFAULT_TARGET_EXTENTS[1]), 1),
            round(_coerce_float(record.get("target_z"), DEFAULT_TARGET_EXTENTS[2]), 1),
            record.get("pressure", 25.0),
            record.get("valve", 4),
            _record_nozzle_number(record, int(record["idx"])),
            record.get("port", 1),
            _color_select_cell(record),
            _coerce_float(record.get("infill", 100.0), 100.0),
            bool(record.get("contour_tracing", False)),
            bool(record.get("lead_in", False)),
            "Delete",
        ]
        for record in records
    ]


def _apply_shape_settings(records: list[dict], settings_table: Any) -> list[dict]:
    rows = _normalise_rows(settings_table)
    by_idx: dict[int, list[Any]] = {}
    for row in rows:
        if not row:
            continue
        try:
            by_idx[int(float(row[0]))] = row
        except (TypeError, ValueError):
            continue
    updated: list[dict] = []
    for record in records or []:
        copy = dict(record)
        row = by_idx.get(int(copy.get("idx", 0)))
        if row:
            copy["name"] = str(row[1] or copy["name"])
            if "original_x" not in copy or "original_y" not in copy or "original_z" not in copy:
                original_x, original_y, original_z = _default_target_extents_for_stl(str(copy.get("stl_path", "")))
                copy.setdefault("original_x", original_x)
                copy.setdefault("original_y", original_y)
                copy.setdefault("original_z", original_z)
            for key, pos, default in (
                ("target_x", 2, DEFAULT_TARGET_EXTENTS[0]),
                ("target_y", 3, DEFAULT_TARGET_EXTENTS[1]),
                ("target_z", 4, DEFAULT_TARGET_EXTENTS[2]),
                ("pressure", 5, 25.0),
                ("valve", 6, 4),
            ):
                try:
                    copy[key] = float(row[pos])
                except (IndexError, TypeError, ValueError):
                    copy[key] = copy.get(key, default)
            has_nozzle_column = len(row) >= len(SHAPE_SETTINGS_HEADERS)
            nozzle_pos = 7 if has_nozzle_column else None
            port_pos = 8 if has_nozzle_column else 7
            color_pos = 9 if has_nozzle_column else 8
            infill_pos = 10 if has_nozzle_column else 9
            contour_pos = 11 if has_nozzle_column else 10
            copy["valve"] = _coerce_int(copy.get("valve", 4), 4)
            copy["nozzle"] = _coerce_int(
                row[nozzle_pos] if nozzle_pos is not None else copy.get("nozzle", copy.get("idx", 1)),
                _record_nozzle_number(copy),
            )
            copy["port"] = _coerce_int(
                row[port_pos] if len(row) > port_pos else copy.get("port", 1),
                _coerce_int(copy.get("port", 1), 1),
            )
            if copy["nozzle"] <= 0:
                copy["nozzle"] = _record_nozzle_number(copy)
            copy["color"] = _color_from_cell(
                row[color_pos] if len(row) > color_pos else None,
                str(copy.get("color") or _default_color(int(copy.get("idx", 1) or 1))),
            )
            try:
                copy["infill"] = max(
                    0.0,
                    min(100.0, float(row[infill_pos])),
                )
            except (IndexError, TypeError, ValueError):
                copy["infill"] = _coerce_float(copy.get("infill", 100.0), 100.0)
            try:
                copy["contour_tracing"] = _coerce_bool(row[contour_pos], bool(copy.get("contour_tracing", False)))
            except IndexError:
                copy["contour_tracing"] = bool(copy.get("contour_tracing", False))
            lead_in_pos = contour_pos + 1
            try:
                copy["lead_in"] = _coerce_bool(row[lead_in_pos], bool(copy.get("lead_in", False)))
            except IndexError:
                copy["lead_in"] = bool(copy.get("lead_in", False))
        updated.append(copy)
    return updated


def _last_edited_target_axes(records: list[dict] | None, settings_table: Any) -> dict[int, str]:
    rows = _normalise_rows(settings_table)
    previous_by_idx: dict[int, dict] = {}
    for record in records or []:
        try:
            previous_by_idx[int(record.get("idx", 0))] = record
        except (TypeError, ValueError):
            continue

    edited_axes: dict[int, str] = {}
    for row in rows:
        try:
            idx = int(float(row[0]))
        except (IndexError, TypeError, ValueError):
            continue
        previous = previous_by_idx.get(idx)
        if not previous:
            continue
        changed_axes: list[str] = []
        for key, pos in zip(TARGET_DIMENSION_KEYS, (2, 3, 4)):
            try:
                new_value = float(row[pos])
                # The table displays tenths, so an edit is a deviation from
                # what was DISPLAYED — comparing against an unrounded stored
                # value would flag phantom edits on legacy noisy records.
                old_value = round(float(previous.get(key)), 1)
            except (IndexError, TypeError, ValueError):
                continue
            if not math.isclose(new_value, old_value, rel_tol=1e-9, abs_tol=1e-9):
                changed_axes.append(key)
        if changed_axes:
            edited_axes[idx] = changed_axes[-1]
    return edited_axes


def _nozzle_spacing_label(nozzle: int, records: list[dict]) -> str:
    shapes = [
        f"Shape {record.get('idx', '?')}"
        for record in records
        if _record_nozzle_number(record, int(record.get("idx", 1) or 1)) == nozzle
    ]
    if shapes:
        return f"Nozzle {nozzle}: {', '.join(shapes)}"
    return f"Nozzle {nozzle}"


def _ordered_nozzle_numbers(records: list[dict]) -> list[int]:
    nozzles = {
        _record_nozzle_number(record, int(record.get("idx", position) or position))
        for position, record in enumerate(records, start=1)
    }
    return sorted(nozzles)


def _spacing_pairs_from_table(spacing_table: Any) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in _normalise_rows(spacing_table):
        if not row:
            continue
        try:
            if len(row) >= 4:
                pairs.append((float(row[2]), float(row[3])))
            elif len(row) >= 3:
                pairs.append((float(row[1]), float(row[2])))
        except (TypeError, ValueError):
            continue
    return pairs


def _grid_spacing_rows(
    records: list[dict],
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    existing_table: Any | None = None,
) -> tuple[list[list[Any]], int, int]:
    ordered_nozzles = _ordered_nozzle_numbers(records)
    column_count = max(1, _coerce_int(columns, 2))
    requested_rows = max(1, _coerce_int(rows, 1))
    row_count = max(requested_rows, math.ceil(len(ordered_nozzles) / column_count) if ordered_nozzles else requested_rows)
    x_spacing = _coerce_float(column_spacing, 5.0)
    y_spacing = _coerce_float(row_spacing, 5.0)
    existing_pairs = _spacing_pairs_from_table(existing_table)

    spacing_rows: list[list[Any]] = []
    for index, (first, second) in enumerate(zip(ordered_nozzles, ordered_nozzles[1:])):
        default_gap_x, default_gap_y = _grid_default_gap_for_pair(index, column_count, x_spacing, y_spacing)
        gap_x, gap_y = existing_pairs[index] if index < len(existing_pairs) else (default_gap_x, default_gap_y)
        spacing_rows.append([
            _nozzle_spacing_label(first, records),
            _nozzle_spacing_label(second, records),
            gap_x,
            gap_y,
        ])
    return spacing_rows, column_count, row_count


def _grid_spacing_table_update(
    records: list[dict] | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    existing_table: Any | None = None,
    use_individual_grid_spacing: bool | None = False,
) -> dict[str, Any]:
    spacing_rows, _column_count, _row_count = _grid_spacing_rows(
        records or [],
        columns,
        rows,
        column_spacing,
        row_spacing,
        existing_table if use_individual_grid_spacing else None,
    )
    return gr.update(
        headers=ADVANCED_NOZZLE_SPACING_HEADERS,
        value=spacing_rows,
        row_count=(len(spacing_rows), "fixed"),
        column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
    )


def _records_by_nozzle(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for index, record in enumerate(records, start=1):
        grouped.setdefault(_record_nozzle_number(record, int(record.get("idx", index) or index)), []).append(record)
    return grouped


def _split_pair_was_created_together(records_by_nozzle: dict[int, list[dict]], first: int, second: int) -> bool:
    for first_record in records_by_nozzle.get(first, []):
        first_group = first_record.get("split_group_id")
        if not first_group:
            continue
        first_index = _coerce_int(first_record.get("split_index"), -1)
        if first_index < 0:
            continue
        for second_record in records_by_nozzle.get(second, []):
            if second_record.get("split_group_id") != first_group:
                continue
            if _coerce_int(second_record.get("split_index"), -1) == first_index + 1:
                return True
    return False


def _part_world_bounds(part: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """XY bounding box of a part's toolpath in the shape's own world frame.

    Requires the `; PathOrigin` header the G-code generator writes; parts
    generated before that header exists return None.
    """
    parsed = part.get("parsed") or {}
    origin = parsed.get("path_origin")
    if origin is None:
        return None
    (x_min, y_min, _z_min), (x_max, y_max, _z_max) = parsed["bounds"]
    return (
        (x_min + origin[0], y_min + origin[1]),
        (x_max + origin[0], y_max + origin[1]),
    )


def _nozzle_world_bounds(
    grouped: dict[int, list[dict]],
) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
    world: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
    for nozzle, parts in grouped.items():
        boxes = [_part_world_bounds(part) for part in parts]
        if not boxes or any(box is None for box in boxes):
            continue
        world[nozzle] = (
            (min(box[0][0] for box in boxes), min(box[0][1] for box in boxes)),
            (max(box[1][0] for box in boxes), max(box[1][1] for box in boxes)),
        )
    return world


def _split_grid_shape(records: list[dict]) -> tuple[int, int] | None:
    """The split grid (columns, rows) when records hold exactly one split group."""
    groups = {
        record.get("split_group_id")
        for record in records
        if record.get("split_group_id")
    }
    if len(groups) != 1:
        return None
    group_id = next(iter(groups))
    for record in records:
        if record.get("split_group_id") == group_id:
            columns = _coerce_int(record.get("split_columns"), 0)
            rows = _coerce_int(record.get("split_rows"), 0)
            if columns >= 1 and rows >= 1:
                return columns, rows
    return None


def _auto_align_grid_spacing_rows(
    records: list[dict],
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
) -> tuple[list[list[Any]], int, int, int, int]:
    """Exact per-pair gaps that reassemble split pieces in the parallel view.

    Every generated G-code file records its PathOrigin: the world position of
    the relative toolpath's start. Anchoring each piece back into its world
    frame turns the required pair gaps into the actual world-frame gaps
    between the parts' toolpath bounding boxes — which automatically accounts
    for the raster pattern, filament width, travel buffers, reference-stack
    motion, and overlapping-layer splits. No hardcoded offsets.
    """
    spacing_rows, column_count, row_count = _grid_spacing_rows(
        records,
        columns,
        rows,
        column_spacing,
        row_spacing,
    )
    parts, _messages = _parts_from_records(records)
    world = _nozzle_world_bounds(_group_parts_by_nozzle(parts))
    records_by_nozzle = _records_by_nozzle(records)
    ordered_nozzles = _ordered_nozzle_numbers(records)

    aligned_count = 0
    missing_count = 0
    for index, (first, second) in enumerate(zip(ordered_nozzles, ordered_nozzles[1:])):
        if not _split_pair_was_created_together(records_by_nozzle, first, second):
            continue

        second_column = (index + 1) % column_count
        if second_column == 0:
            # Row transition: `second` opens a new grid row. The layout places
            # it relative to the previous row's first nozzle (x) and the
            # previous row's lowest edge (y).
            previous_row = ordered_nozzles[index + 1 - column_count : index + 1]
            anchors = [second, *previous_row]
        else:
            row_first = ordered_nozzles[index + 1 - second_column]
            anchors = [first, second, row_first]
        if any(nozzle not in world for nozzle in anchors):
            missing_count += 1
            continue

        (second_min_x, second_min_y), _second_max = world[second]
        if second_column == 0:
            gap_x = second_min_x - world[previous_row[0]][0][0]
            gap_y = second_min_y - max(world[nozzle][1][1] for nozzle in previous_row)
        else:
            gap_x = second_min_x - world[first][1][0]
            gap_y = second_min_y - world[row_first][0][1]
        spacing_rows[index][2] = round(gap_x, 4)
        spacing_rows[index][3] = round(gap_y, 4)
        aligned_count += 1
    return spacing_rows, column_count, row_count, aligned_count, missing_count


def apply_nozzle_grid_spacing(
    records: list[dict] | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    records = records or []
    spacing_rows, column_count, row_count = _grid_spacing_rows(records, columns, rows, column_spacing, row_spacing)
    nozzle_count = len(_ordered_nozzle_numbers(records))
    capacity = column_count * row_count
    status = f"Applied {column_count} x {row_count} grid spacing to {max(nozzle_count - 1, 0)} nozzle pair(s)."
    if nozzle_count > capacity:
        status += f" {nozzle_count} nozzles exceed {capacity} grid slots, so spacing continues row by row."
    return (
        gr.update(value=True),
        gr.update(
            headers=ADVANCED_NOZZLE_SPACING_HEADERS,
            value=spacing_rows,
            row_count=(len(spacing_rows), "fixed"),
            column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
            label="Advanced Nozzle Spacing",
        ),
        status,
    )


def auto_align_split_parts(
    records: list[dict] | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
) -> tuple:
    records = records or []
    grid_shape = _split_grid_shape(records)
    if grid_shape is not None:
        columns, rows = grid_shape
    (
        spacing_rows,
        column_count,
        row_count,
        aligned_count,
        missing_count,
    ) = _auto_align_grid_spacing_rows(records, columns, rows, column_spacing, row_spacing)

    if aligned_count <= 0:
        if missing_count > 0:
            status = (
                "Auto align needs the split pieces' generated G-code: press "
                "Generate G-Code first (files generated before this feature "
                "lack the PathOrigin header), then align again."
            )
        else:
            status = "No split-sibling nozzle connections found. Auto align was not applied."
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            status,
        )

    status = (
        f"Auto aligned {aligned_count} split nozzle connection(s) from the generated "
        f"G-code in a {column_count} x {row_count} grid. The exact per-connection "
        "gaps are in the Advanced Grid Spacing table."
    )
    if missing_count:
        status += (
            f"  \n{missing_count} connection(s) skipped: regenerate G-code for those "
            "shapes to add the PathOrigin header, then align again."
        )
    return (
        gr.update(value=column_count),
        gr.update(value=row_count),
        gr.update(),
        gr.update(),
        gr.update(value=True),
        gr.update(
            headers=ADVANCED_NOZZLE_SPACING_HEADERS,
            value=spacing_rows,
            row_count=(len(spacing_rows), "fixed"),
            column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
        ),
        status,
    )


def update_nozzle_grid_preset(
    preset: str | None,
    records: list[dict] | None,
    columns: Any,
    rows: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nozzle_count = max(1, len(_ordered_nozzle_numbers(records or [])))
    if preset == "One row":
        return gr.update(value=nozzle_count), gr.update(value=1)
    if preset == "One column":
        return gr.update(value=1), gr.update(value=nozzle_count)
    if preset and " x " in preset:
        left, right = preset.split(" x ", 1)
        return gr.update(value=max(1, _coerce_int(left, 1))), gr.update(value=max(1, _coerce_int(right, 1)))
    return gr.update(value=max(1, _coerce_int(columns, 1))), gr.update(value=max(1, _coerce_int(rows, 1)))


def _dropdown_update(records: list[dict], selected: str | None = None) -> dict[str, Any]:
    choices = [_shape_choice(record) for record in records]
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _gcode_dropdown_update(records: list[dict], selected: str | None = None, include_upload: bool = False) -> dict[str, Any]:
    choices = [_shape_choice(record) for record in records if record.get("gcode_path")]
    if include_upload:
        # The visualization source radio: the parallel view leads and is the
        # default; single shapes and the upload option follow.
        choices = [GCODE_SOURCE_PARALLEL, *choices, GCODE_SOURCE_UPLOAD]
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _gcode_zip_update(records: list[dict] | None) -> dict[str, Any]:
    """Bundle every generated G-code file into one ZIP for the Download All
    button. Rebuilt wherever the individual download list is refreshed so the
    two can never disagree; the button hides when there is nothing to bundle."""
    paths = [
        record.get("gcode_path")
        for record in (records or [])
        if record.get("gcode_path") and Path(record["gcode_path"]).exists()
    ]
    if not paths:
        return gr.update(visible=False)
    zip_path = Path(tempfile.mkdtemp(prefix="gcode_zip_")) / "all_shapes_gcode.zip"
    used_names: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            arcname = Path(path).name
            stem, suffix = Path(path).stem, Path(path).suffix
            counter = 2
            while arcname in used_names:
                arcname = f"{stem}_{counter}{suffix}"
                counter += 1
            used_names.add(arcname)
            bundle.write(path, arcname=arcname)
    return gr.update(value=str(zip_path), visible=True)


def _merge_file_paths(*file_groups: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in file_groups:
        for path in _uploaded_file_paths(group):
            key = str(Path(path))
            if key in seen:
                continue
            seen.add(key)
            merged.append(path)
    return merged


def _append_file_paths(*file_groups: Any) -> list[str]:
    paths: list[str] = []
    for group in file_groups:
        paths.extend(_uploaded_file_paths(group))
    return paths


def sync_uploaded_shapes(
    files: Any,
    records: list[dict] | None,
    settings_table: Any | None = None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    next_records = _records_from_files(files, records)
    settings = _shape_settings_rows(next_records)
    return (
        next_records,
        settings,
        _dropdown_update(next_records),
        _gcode_dropdown_update(next_records),
        _gcode_dropdown_update(next_records, include_upload=True),
        [record.get("gcode_path") for record in next_records if record.get("gcode_path")],
        _gcode_zip_update(next_records),
    )


def load_sample_shapes(
    files: Any,
    records: list[dict] | None,
    settings_table: Any | None = None,
    sample_set: str | None = None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    set_name = str(sample_set or "")
    filenames = SAMPLE_STL_SETS.get(set_name, SAMPLE_STL_SETS[DEFAULT_SAMPLE_STL_SET])
    paths = [str(SAMPLE_STL_DIR / filename) for filename in filenames if (SAMPLE_STL_DIR / filename).exists()]
    merged_paths = _append_file_paths(files, paths)
    outputs = sync_uploaded_shapes(merged_paths, records, None)
    nozzle_map = SAMPLE_SET_NOZZLES.get(set_name, {})
    if nozzle_map:
        # Group the set's multi-material parts onto their shared nozzles.
        next_records = outputs[0]
        for record in next_records:
            filename = Path(str(record.get("stl_path") or "")).name
            if filename in nozzle_map:
                record["nozzle"] = nozzle_map[filename]
        outputs = (next_records, _shape_settings_rows(next_records), *outputs[2:])
    return (gr.update(value=merged_paths), *outputs)


def _shape_delete_outputs(
    records: list[dict],
    last_delete_at: float | None,
    upload_update: Any | None = None,
) -> tuple:
    return (
        upload_update if upload_update is not None else gr.update(),
        records,
        _shape_settings_rows(records),
        _dropdown_update(records),
        _gcode_dropdown_update(records),
        _gcode_dropdown_update(records, include_upload=True),
        [record.get("gcode_path") for record in records if record.get("gcode_path")],
        _gcode_zip_update(records),
        float(last_delete_at or 0.0),
    )


def _shape_select_noop() -> tuple:
    """Skip every output: a cell click that is not a Delete click must not
    touch anything. This handler fires on EVERY cell selection (queued, so it
    lands late); echoing records/rows here raced the dimension normalizer's
    write-back — the stale echo clobbered the recomputed proportions on the
    first Keep Proportions edit and re-rendered the table mid-typing."""
    return tuple(gr.skip() for _ in range(9))


def delete_shape_from_settings(
    records: list[dict] | None,
    settings_table: Any | None,
    last_delete_at: float | None,
    evt: gr.SelectData,
) -> tuple:
    now = time.monotonic()
    rows = _normalise_rows(settings_table)
    selected = getattr(evt, "index", None)
    if not isinstance(selected, (list, tuple)) or len(selected) < 2:
        return _shape_select_noop()

    try:
        row_index, column_index = int(selected[0]), int(selected[1])
    except (TypeError, ValueError):
        return _shape_select_noop()
    delete_column_index = len(SHAPE_SETTINGS_HEADERS) - 1
    if column_index != delete_column_index or row_index < 0 or row_index >= len(rows):
        return _shape_select_noop()
    if last_delete_at and now - float(last_delete_at) < DELETE_SHAPE_COOLDOWN_SECONDS:
        return _shape_select_noop()

    current_records = _apply_shape_settings(records or [], settings_table)
    try:
        delete_idx = int(float(rows[row_index][0]))
    except (IndexError, TypeError, ValueError):
        delete_idx = row_index + 1

    next_records = _reindex_shape_records([
        record for record in current_records if int(record.get("idx", 0)) != delete_idx
    ])
    upload_paths = [record.get("stl_path") for record in next_records if record.get("stl_path")]
    return _shape_delete_outputs(
        next_records,
        now,
        gr.update(value=upload_paths),
    )


def reset_shape_dimensions(records: list[dict] | None, settings_table: Any | None = None) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    reset_records: list[dict] = []
    for record in records:
        copy = dict(record)
        original_x = copy.get("original_x")
        original_y = copy.get("original_y")
        original_z = copy.get("original_z")
        if original_x is None or original_y is None or original_z is None:
            original_x, original_y, original_z = _default_target_extents_for_stl(str(copy.get("stl_path", "")))
        original_x = round(float(original_x), 1)
        original_y = round(float(original_y), 1)
        original_z = round(float(original_z), 1)
        copy["original_x"] = original_x
        copy["original_y"] = original_y
        copy["original_z"] = original_z
        copy["target_x"] = original_x
        copy["target_y"] = original_y
        copy["target_z"] = original_z
        copy["last_scaled_axis"] = "target_x"
        reset_records.append(copy)
    return reset_records, _shape_settings_rows(reset_records)


BULK_BOOL_COLUMNS = {
    "Contour Tracing": "contour_tracing",
    "Lead In": "lead_in",
}


def apply_bulk_bool_selection(
    records: list[dict] | None,
    settings_table: Any,
    payload: str | None,
) -> tuple[list[dict], list[list[Any]]]:
    """Set a checkbox column for ALL shapes ("columnIndex|0/1" from the sink).

    Backs the header select-all checkboxes: Gradio's own implementation is
    unreliable (see the head-script hijack), so the whole column is set here
    and the table re-rendered canonically.
    """
    records = _apply_shape_settings(records or [], settings_table)
    parts = str(payload or "").split("|")
    if len(parts) >= 2:
        try:
            column_index = int(parts[0])
            value = bool(int(parts[1]))
        except (TypeError, ValueError):
            column_index = -1
            value = False
        if 0 <= column_index < len(SHAPE_SETTINGS_HEADERS):
            key = BULK_BOOL_COLUMNS.get(SHAPE_SETTINGS_HEADERS[column_index])
            if key:
                records = [dict(record, **{key: value}) for record in records]
    return records, _shape_settings_rows(records)


def _bool_cells_need_rewrite(settings_table: Any) -> bool:
    """True when the checkbox columns carry non-boolean cell values.

    Gradio's header "select all" writes the STRING "true"/"false" into the
    column's cells (and leaves some rendered as plain text or stale
    checkboxes in neighbouring columns) instead of booleans. Detecting that
    lets the normalizer answer with a canonical re-render, which restores
    real checkboxes and stomps any visual strays within one round-trip.
    """
    rows = _normalise_rows(settings_table)
    contour_pos = SHAPE_SETTINGS_HEADERS.index("Contour Tracing")
    bool_positions = (contour_pos, contour_pos + 1)
    for row in rows:
        if len(row) < len(SHAPE_SETTINGS_HEADERS):
            continue
        for pos in bool_positions:
            if not isinstance(row[pos], bool):
                return True
    return False


def _last_edited_nozzles(records: list[dict] | None, settings_table: Any) -> set[int]:
    """Record idx whose Nozzle cell differs from the record — i.e. shapes the
    user just moved onto a (possibly new) nozzle via the table."""
    rows = _normalise_rows(settings_table)
    previous_by_idx: dict[int, dict] = {}
    for record in records or []:
        try:
            previous_by_idx[int(record.get("idx", 0))] = record
        except (TypeError, ValueError):
            continue

    nozzle_pos = SHAPE_SETTINGS_HEADERS.index("Nozzle")
    changed: set[int] = set()
    for row in rows:
        try:
            idx = int(float(row[0]))
        except (IndexError, TypeError, ValueError):
            continue
        previous = previous_by_idx.get(idx)
        if not previous or len(row) <= nozzle_pos:
            continue
        old_nozzle = _record_nozzle_number(previous, idx)
        new_nozzle = _coerce_int(row[nozzle_pos], old_nozzle)
        if new_nozzle > 0 and new_nozzle != old_nozzle:
            changed.add(idx)
    return changed


def _last_edited_pressures(records: list[dict] | None, settings_table: Any) -> dict[int, float]:
    """Record idx -> new pressure for rows whose Pressure cell differs from
    the record — i.e. pressures the user just edited in the table."""
    rows = _normalise_rows(settings_table)
    previous_by_idx: dict[int, dict] = {}
    for record in records or []:
        try:
            previous_by_idx[int(record.get("idx", 0))] = record
        except (TypeError, ValueError):
            continue

    pressure_pos = SHAPE_SETTINGS_HEADERS.index("Pressure (psi)")
    edited: dict[int, float] = {}
    for row in rows:
        try:
            idx = int(float(row[0]))
        except (IndexError, TypeError, ValueError):
            continue
        previous = previous_by_idx.get(idx)
        if not previous or len(row) <= pressure_pos:
            continue
        old_pressure = _coerce_float(previous.get("pressure"), 25.0)
        new_pressure = _coerce_float(row[pressure_pos], old_pressure)
        if not math.isclose(new_pressure, old_pressure, rel_tol=0.0, abs_tol=1e-9):
            edited[idx] = new_pressure
    return edited


def _last_edited_ports(records: list[dict] | None, settings_table: Any) -> set[int]:
    """Record idx whose Port cell differs from the record — i.e. shapes the
    user just moved onto a (possibly new) port via the table."""
    rows = _normalise_rows(settings_table)
    previous_by_idx: dict[int, dict] = {}
    for record in records or []:
        try:
            previous_by_idx[int(record.get("idx", 0))] = record
        except (TypeError, ValueError):
            continue

    port_pos = SHAPE_SETTINGS_HEADERS.index("Port")
    changed: set[int] = set()
    for row in rows:
        try:
            idx = int(float(row[0]))
        except (IndexError, TypeError, ValueError):
            continue
        previous = previous_by_idx.get(idx)
        if not previous or len(row) <= port_pos:
            continue
        old_port = _coerce_int(previous.get("port"), 1)
        new_port = _coerce_int(row[port_pos], old_port)
        if new_port != old_port:
            changed.add(idx)
    return changed


def _sync_port_pressures(
    records: list[dict],
    edited_pressures: dict[int, float],
    joined_idx: set[int] | None = None,
) -> list[dict]:
    """Sync pressures across shapes sharing a serial Port.

    Pressure is a PORT property — one regulator per serial port — so every
    shape on a port must carry the same pressure. The source value must be
    unambiguous: the pressure the user just edited (all edits agreeing), or
    — when a shape just JOINED the group by a port edit — the incumbent
    members' shared pressure, which the newcomer adopts. Groups already in
    sync, or with no clear source (e.g. stale .change echoes that flag
    conflicting members at once), are left untouched, so the event storm
    converges instead of ping-ponging.
    """
    joined_idx = joined_idx or set()
    by_port: dict[int, list[dict]] = {}
    for record in records:
        by_port.setdefault(_coerce_int(record.get("port"), 1), []).append(record)

    for members in by_port.values():
        if len(members) < 2:
            continue
        pressures = [_coerce_float(member.get("pressure"), 25.0) for member in members]
        if all(math.isclose(p, pressures[0], rel_tol=0.0, abs_tol=1e-9) for p in pressures[1:]):
            continue  # port group already in sync

        edited_values = {
            round(edited_pressures[int(member.get("idx", 0))], 9)
            for member in members
            if int(member.get("idx", 0)) in edited_pressures
        }
        incumbents = [
            member for member in members if int(member.get("idx", 0)) not in joined_idx
        ]
        newcomers = [
            member for member in members if int(member.get("idx", 0)) in joined_idx
        ]
        if len(edited_values) == 1:
            value = next(iter(edited_values))
        elif not edited_values and newcomers and incumbents:
            incumbent_pressures = [
                _coerce_float(member.get("pressure"), 25.0) for member in incumbents
            ]
            if not all(
                math.isclose(p, incumbent_pressures[0], rel_tol=0.0, abs_tol=1e-9)
                for p in incumbent_pressures[1:]
            ):
                continue
            # A port edit pulled newcomers into the group: they adopt the
            # incumbents' shared pressure.
            value = incumbent_pressures[0]
        else:
            continue  # ambiguous (stale echo): do not guess

        for member in members:
            member["pressure"] = value
    return records


def _record_scale_factors(record: dict) -> tuple[float, float, float] | None:
    """Per-axis target/original factors, or None when they can't be computed."""
    try:
        originals = [float(record.get(f"original_{axis}")) for axis in ("x", "y", "z")]
        targets = [float(record.get(f"target_{axis}")) for axis in ("x", "y", "z")]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) or v <= 0 for v in originals + targets):
        return None
    return tuple(target / original for target, original in zip(targets, originals))


def _propagate_group_scale_factors(
    records: list[dict],
    edited_axes: dict[int, str],
    recomputed_idx: set[int],
    joined_idx: set[int] | None = None,
) -> list[dict]:
    """Sync each multi-material group's scale factors from the edited member.

    Shapes sharing a nozzle are one assembly, so a dimension edit on one
    part scales EVERY part by the same per-axis factor (target/original) —
    absolute values would distort assemblies whose parts differ in size.
    The source member must be unambiguous: the one whose row was actually
    recomputed this round (Keep Proportions), the single member the user
    edited, or — when a shape just JOINED the group by a nozzle edit — the
    incumbent members' shared factors, which the newcomer adopts. Groups
    already in sync, or with no clear source (e.g. stale .change echoes that
    flag several members at once), are left untouched, so the event storm
    converges instead of ping-ponging.
    """
    joined_idx = joined_idx or set()

    def _triples_close(a: tuple, b: tuple) -> bool:
        return all(
            math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9) for fa, fb in zip(a, b)
        )

    for members in _multi_material_groups(records).values():
        factors = {id(member): _record_scale_factors(member) for member in members}
        if any(value is None for value in factors.values()):
            continue
        triples = list(factors.values())
        if all(_triples_close(triples[0], triple) for triple in triples[1:]):
            continue  # group already in sync

        recomputed_members = [
            member for member in members if int(member.get("idx", 0)) in recomputed_idx
        ]
        edited_members = [
            member for member in members if int(member.get("idx", 0)) in edited_axes
        ]
        newcomers = [
            member for member in members if int(member.get("idx", 0)) in joined_idx
        ]
        incumbents = [
            member for member in members if int(member.get("idx", 0)) not in joined_idx
        ]
        if len(recomputed_members) == 1:
            source = recomputed_members[0]
        elif len(edited_members) == 1:
            source = edited_members[0]
        elif newcomers and incumbents and all(
            _triples_close(factors[id(incumbents[0])], factors[id(member)])
            for member in incumbents[1:]
        ):
            # Nozzle edit pulled newcomers into the group: they adopt the
            # incumbents' shared scale factors.
            source = incumbents[0]
        else:
            continue  # ambiguous (stale echo): do not guess

        source_factors = factors[id(source)]
        for member in members:
            if member is source:
                continue
            for axis, factor in zip(("x", "y", "z"), source_factors):
                member[f"target_{axis}"] = round(
                    float(member[f"original_{axis}"]) * factor, 1
                )
            member["last_scaled_axis"] = source.get(
                "last_scaled_axis", member.get("last_scaled_axis")
            )
    return records


def normalize_shape_dimensions_for_mode(
    records: list[dict] | None,
    settings_table: Any | None,
    scale_mode: str | None,
) -> tuple:
    """Apply table edits to the records; in Keep Proportions, rescale each
    shape's other dimensions from the edited axis. In BOTH modes, a
    dimension edit on a multi-material group member (shapes sharing a
    nozzle) propagates its scale factors to the whole group, so assemblies
    stay proportional as one unit — and a pressure edit propagates to every
    shape sharing the same serial Port (one pressure regulator per port).

    Wired to the table's .change event, WHICH ALSO FIRES FOR OUR OWN
    WRITE-BACK (Gradio's Dataframe does not emit .input for cell edits at
    all). The cascade is broken server-side: the table output is skipped
    whenever normalization did not change any dimension — so a user edit
    costs exactly two rounds (recompute + converged no-op), and programmatic
    table updates cost one no-op round instead of looping.
    """
    edited_axes = _last_edited_target_axes(records, settings_table)
    joined_idx = _last_edited_nozzles(records, settings_table)
    edited_pressures = _last_edited_pressures(records, settings_table)
    port_joined_idx = _last_edited_ports(records, settings_table)
    records = _apply_shape_settings(records or [], settings_table)
    if _normalize_scale_mode(scale_mode) != SCALE_MODE_UNIFORM_FACTOR:
        normalized = [dict(record) for record in records]
        for record in normalized:
            idx = int(record.get("idx", 0))
            if idx in edited_axes:
                record["last_scaled_axis"] = edited_axes[idx]
            if record.get("stl_path"):
                _round_targets_to_tenths(record)
        normalized = _propagate_group_scale_factors(normalized, edited_axes, set(), joined_idx)
        normalized = _sync_port_pressures(normalized, edited_pressures, port_joined_idx)
        changed = any(
            not math.isclose(
                float(before.get(key) or 0.0),
                float(after.get(key) or 0.0),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for before, after in zip(records, normalized)
            for key in (*TARGET_DIMENSION_KEYS, "pressure")
        )
        if not changed and not _bool_cells_need_rewrite(settings_table):
            return normalized, gr.skip()
        return normalized, _shape_settings_rows(normalized)

    recomputed_idx: set[int] = set()
    normalized: list[dict] = []
    for record in records:
        copy = dict(record)
        if not copy.get("stl_path"):
            # Split pieces: their targets are CELL sizes while original_* is
            # inherited from the parent shape, so the ratio logic would
            # misread them as mid-edit and "restore" the parent dimensions.
            # Piece dimensions are informational — never rescale them.
            normalized.append(copy)
            continue
        _round_targets_to_tenths(copy)
        originals = np.asarray([
            copy.get("original_x"),
            copy.get("original_y"),
            copy.get("original_z"),
        ], dtype=float)
        targets = np.asarray([
            copy.get("target_x"),
            copy.get("target_y"),
            copy.get("target_z"),
        ], dtype=float)
        if (
            originals.shape != (3,)
            or targets.shape != (3,)
            or not np.all(np.isfinite(originals))
            or not np.all(np.isfinite(targets))
            or np.any(originals <= 0)
            or np.any(targets <= 0)
        ):
            normalized.append(copy)
            continue
        idx = int(copy.get("idx", 0))

        # Anchor on the row's own evidence: in a proportional row all three
        # target/original ratios agree; a user edit makes exactly one ratio
        # the odd one out. This is ORDER-INDEPENDENT — the .change event
        # storm delivers stale/echoed tables whose rows are self-consistent,
        # and those must recompute to themselves (skip) instead of being
        # diffed against fresher records, mis-anchored, and reverted.
        ratios = targets / originals

        def _close(a: float, b: float) -> bool:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)

        if _close(ratios[0], ratios[1]) and _close(ratios[1], ratios[2]):
            # Already proportional (a pristine row or an echo of our own
            # write-back): nothing to recompute.
            if idx in edited_axes:
                copy["last_scaled_axis"] = edited_axes[idx]
            normalized.append(copy)
            continue
        # others_agree[i] means the OTHER two ratios agree -> axis i was edited.
        others_agree = [
            _close(ratios[1], ratios[2]),
            _close(ratios[0], ratios[2]),
            _close(ratios[0], ratios[1]),
        ]
        if sum(others_agree) == 1:
            anchor_index = others_agree.index(True)
        else:
            # All three differ (e.g. custom dims from Independent mode):
            # fall back to the detected edit, then the last anchor.
            anchor_key = edited_axes.get(idx) or copy.get("last_scaled_axis") or "target_x"
            try:
                anchor_index = TARGET_DIMENSION_KEYS.index(str(anchor_key))
            except ValueError:
                anchor_index = 0
        scale = float(targets[anchor_index] / originals[anchor_index])
        copy["last_scaled_axis"] = TARGET_DIMENSION_KEYS[anchor_index]
        scaled = originals * scale
        copy["target_x"] = round(float(scaled[0]), 1)
        copy["target_y"] = round(float(scaled[1]), 1)
        copy["target_z"] = round(float(scaled[2]), 1)
        recomputed_idx.add(idx)
        normalized.append(copy)

    normalized = _propagate_group_scale_factors(
        normalized, edited_axes, recomputed_idx, joined_idx
    )
    normalized = _sync_port_pressures(normalized, edited_pressures, port_joined_idx)

    # Idempotence guard (breaks the .change write-back cascade): only write
    # the table when a dimension or pressure actually changed — or when the
    # checkbox columns need a canonical re-render after a header "select all".
    changed = any(
        not math.isclose(
            float(before.get(key) or 0.0),
            float(after.get(key) or 0.0),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for before, after in zip(records, normalized)
        for key in (*TARGET_DIMENSION_KEYS, "pressure")
    )
    if not changed and not _bool_cells_need_rewrite(settings_table):
        return normalized, gr.skip()
    return normalized, _shape_settings_rows(normalized)


def show_selected_model(
    records: list[dict] | None,
    selected: str | None,
    settings_table: Any,
    scale_mode: str | None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    pos = _selected_record_index(records, selected)
    if pos < 0:
        return _viewer_update(None), "No model loaded."
    record = records[pos]
    return load_single_model(
        record.get("stl_path"),
        False,  # full opacity (the 75%-opacity option was removed)
        True,
        scale_mode,
        record.get("target_x"),
        record.get("target_y"),
        record.get("target_z"),
    )


def _polygon_patch(polygon, **kwargs):
    """A filled matplotlib patch for a shapely Polygon, holes included."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in [polygon.exterior, *polygon.interiors]:
        coords = list(ring.coords)
        if len(coords) < 3:
            continue
        verts.extend(coords)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(coords) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(verts, codes), **kwargs)


def _layer_preview_message(message: str):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5.4, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=11, color="#666666")
    return fig


def update_layer_preview(
    records: list[dict] | None,
    selected: str | None,
    settings_table: Any,
    layer_value: float,
) -> tuple:
    """Slider + figure for the sliced-layer preview.

    Draws the selected shape's layer polygons in its print color. Shapes on
    the same nozzle are parts of one assembly, so their polygons at the same
    Z are drawn too (dimmer) — this is where multi-material slicing can be
    checked before generating any G-code.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    records = _apply_shape_settings(list(records or []), settings_table)
    pos = _selected_record_index(records, selected)
    if pos < 0:
        return gr.update(maximum=1, value=1), _layer_preview_message("Load a shape to preview its sliced layers.")
    record = records[pos]
    stack = record.get("layer_stack")
    if stack is None or not getattr(stack, "layers", None):
        return (
            gr.update(maximum=1, value=1),
            _layer_preview_message("Generate G-Code (or split) to slice this shape and see its layer outlines."),
        )

    layer_count = len(stack.layers)
    layer_index = max(1, min(layer_count, _coerce_int(layer_value, 1)))
    z_value = stack.z_values[layer_index - 1]
    half_layer = float(stack.layer_height or 0.8) / 2.0

    nozzle = _record_nozzle_number(record, int(record.get("idx", pos + 1) or (pos + 1)))
    drawn: list[tuple[dict, Any]] = []  # (record, MultiPolygon at ~z)
    for other_pos, other in enumerate(records):
        other_stack = other.get("layer_stack")
        if other_stack is None or not getattr(other_stack, "layers", None) or other_pos == pos:
            continue
        if _record_nozzle_number(other, int(other.get("idx", other_pos + 1) or (other_pos + 1))) != nozzle:
            continue
        z_values = other_stack.z_values
        nearest = min(range(len(z_values)), key=lambda k: abs(z_values[k] - z_value))
        if abs(z_values[nearest] - z_value) <= half_layer:
            drawn.append((other, other_stack.layers[nearest]))
    drawn.append((record, stack.layers[layer_index - 1]))  # selected on top

    fig = Figure(figsize=(5.4, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    legend_handles = []
    has_material = False
    for member, layer in drawn:
        is_selected = member is record
        color = str(member.get("color") or _default_color(int(member.get("idx", 0) or 0)))
        for polygon in getattr(layer, "geoms", []):
            if polygon.is_empty:
                continue
            has_material = True
            ax.add_patch(
                _polygon_patch(
                    polygon,
                    facecolor=color,
                    alpha=0.85 if is_selected else 0.45,
                    edgecolor="#333333",
                    linewidth=0.7,
                )
            )
        if len(drawn) > 1:
            legend_handles.append(
                Patch(facecolor=color, alpha=0.85 if is_selected else 0.45, label=str(member.get("name") or f"Shape {member.get('idx')}"))
            )

    # Fixed frame across the whole stack (assembly bounds) so the view
    # doesn't jump between layers.
    xs: list[float] = []
    ys: list[float] = []
    for member, _layer in drawn:
        bounds = member["layer_stack"].bounds
        xs.extend((bounds[0][0], bounds[1][0]))
        ys.extend((bounds[0][1], bounds[1][1]))
    margin = max(1.0, 0.05 * max(max(xs) - min(xs), max(ys) - min(ys)))
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    name = str(record.get("name") or f"Shape {record.get('idx')}")
    title = f"{name} — layer {layer_index}/{layer_count} (z = {z_value:.2f} mm)"
    if not has_material:
        title += "  [empty layer]"
    ax.set_title(title, fontsize=10)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    return gr.update(maximum=layer_count, value=layer_index), fig


def _slice_params_snapshot(
    record: dict,
    layer_height: float,
    scale_mode: str | None,
    slice_plan: tuple[list[float], tuple[float, float, float]] | None = None,
) -> dict:
    z_levels = slice_plan[0] if slice_plan else None
    anchor = slice_plan[1] if slice_plan else None
    return {
        "layer_height": float(layer_height),
        "scale_mode": _normalize_scale_mode(scale_mode),
        "target_x": record.get("target_x"),
        "target_y": record.get("target_y"),
        "target_z": record.get("target_z"),
        # Multi-material groups: the shared Z grid + scale anchor
        # fingerprint. Adding/removing an assembly part changes them, which
        # correctly marks every part's slices stale.
        "z_grid": (round(z_levels[0], 6), len(z_levels)) if z_levels else None,
        "scale_anchor": tuple(round(v, 6) for v in anchor) if anchor else None,
    }


def _multi_material_groups(records: list[dict]) -> dict[int, list[dict]]:
    """Multi-material groups: uploaded shapes that share a nozzle number.

    Shapes on the same nozzle print from the same physical position, so
    their STLs are parts of one assembly (one STL per material). They must
    slice on one shared Z grid and keep their modelled positions relative to
    each other. Shapes alone on their nozzle stay ordinary. Split pieces
    (no stl_path) are excluded — they carry their own frame alignment.
    """
    by_nozzle: dict[int, list[dict]] = {}
    for record in records:
        if not record.get("stl_path"):
            continue
        nozzle = _record_nozzle_number(record, int(record.get("idx", 1) or 1))
        by_nozzle.setdefault(nozzle, []).append(record)
    return {nozzle: members for nozzle, members in by_nozzle.items() if len(members) > 1}


def _stamp_multi_material_frames(records: list[dict], fil_width: float = 0.8) -> None:
    """Stamp each sliced stack's multi-material group frame and seam-free
    contour paths (or clear them).

    Group members get the group's combined XY bbox as `align_frame`, so the
    reference alignment moves them as one rigid unit, and `contour_paths`
    that exclude material-to-material interfaces — the assembled parts form
    ONE shape, so only its true outer surface is contoured (boundary within
    half a bead of a sibling material counts as an interface, covering both
    exact contact and fit-tolerance gaps). Idempotent — safe to call before
    every reference build so nozzle renumbering in the table takes effect
    without re-slicing.
    """
    grouped_ids: set[int] = set()
    for members in _multi_material_groups(records).values():
        stacks = [
            member.get("layer_stack")
            for member in members
            if member.get("layer_stack") is not None
        ]
        if len(stacks) < 2:
            continue
        frame = (
            min(stack.bounds[0][0] for stack in stacks),
            min(stack.bounds[0][1] for stack in stacks),
            max(stack.bounds[1][0] for stack in stacks),
            max(stack.bounds[1][1] for stack in stacks),
        )
        for stack in stacks:
            stack.align_frame = frame
            stack.contour_paths = group_contour_paths(
                stack,
                [other for other in stacks if other is not stack],
                tolerance=float(fil_width or 0.8) / 2.0,
            )
            grouped_ids.add(id(stack))
    for record in records:
        stack = record.get("layer_stack")
        if stack is not None and id(stack) not in grouped_ids and record.get("stl_path"):
            stack.align_frame = None
            stack.contour_paths = None


def _multi_material_slice_plan(
    records: list[dict],
    layer_height: float,
    scale_mode: str | None,
) -> tuple[list[float], tuple[float, float, float]] | None:
    """(shared Z grid, shared scale anchor) for one multi-material group.

    Group members must slice on the SAME planes so a part that starts
    higher gets empty lower layers instead of having its first material
    layer treated as layer 0 — and any target-dimension scaling must happen
    about ONE shared point (the group's combined un-scaled corner), or
    same-factor scaling would still shift the parts relative to each other.
    """
    loaded: list[tuple[Any, tuple[float, float, float]]] = []
    corner = [math.inf, math.inf, math.inf]
    for record in records:
        stl_path = record.get("stl_path")
        if not stl_path:
            continue
        try:
            mesh = load_mesh(stl_path)
            scale_factors = _resolve_mesh_scale_factors(
                mesh,
                True,
                scale_mode,
                record.get("target_x"),
                record.get("target_y"),
                record.get("target_z"),
            )
        except Exception:
            continue
        loaded.append((mesh, scale_factors))
        for axis in range(3):
            corner[axis] = min(corner[axis], float(mesh.bounds[0][axis]))
    if not loaded or not all(math.isfinite(value) for value in corner):
        return None

    anchor = (corner[0], corner[1], corner[2])
    z_lo = math.inf
    z_hi = -math.inf
    for mesh, scale_factors in loaded:
        try:
            scaled = scale_mesh(mesh, scale_factors, anchor=anchor)
        except Exception:
            continue
        z_lo = min(z_lo, float(scaled.bounds[0][2]))
        z_hi = max(z_hi, float(scaled.bounds[1][2]))
    if not math.isfinite(z_lo) or not math.isfinite(z_hi):
        return None
    return calculate_z_levels(z_lo, z_hi, float(layer_height)), anchor


def _slice_record(
    record: dict,
    layer_height: float,
    scale_mode: str | None,
    progress_callback=None,
    slice_plan: tuple[list[float], tuple[float, float, float]] | None = None,
) -> LayerStack:
    stl_path = record["stl_path"]
    mesh = load_mesh(stl_path)
    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        True,
        scale_mode,
        record.get("target_x"),
        record.get("target_y"),
        record.get("target_z"),
    )
    stack = slice_stl_to_layers(
        stl_path,
        layer_height=float(layer_height),
        progress_callback=progress_callback,
        scale_factors=scale_factors,
        name=str(record.get("name") or Path(stl_path).stem),
        z_levels=slice_plan[0] if slice_plan else None,
        scale_anchor=slice_plan[1] if slice_plan else None,
    )
    record["layer_stack"] = stack
    record["slice_params"] = _slice_params_snapshot(record, layer_height, scale_mode, slice_plan)
    return stack


def _group_z_levels_by_record(
    records: list[dict],
    layer_height: float,
    scale_mode: str | None,
    messages: list[str] | None = None,
) -> dict[int, tuple[list[float], tuple[float, float, float]]]:
    """Per multi-material group member: (shared Z grid, shared scale anchor),
    keyed by record id."""
    plan_by_record: dict[int, tuple[list[float], tuple[float, float, float]]] = {}
    for nozzle, members in sorted(_multi_material_groups(records).items()):
        plan = _multi_material_slice_plan(members, layer_height, scale_mode)
        if plan is None:
            continue
        z_levels = plan[0]
        for member in members:
            plan_by_record[id(member)] = plan
        if messages is not None:
            names = ", ".join(str(m.get("name") or f"Shape {m['idx']}") for m in members)
            note = (
                f"Multi-material group (nozzle {nozzle}): {names} — sliced on one "
                f"shared Z grid ({len(z_levels)} layers), positions locked together."
            )
            messages.append(note)
    return plan_by_record


def generate_dynamic_layer_stacks(
    records: list[dict] | None,
    settings_table: Any,
    layer_height: float,
    scale_mode: str | None,
    fil_width: float = 0.8,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    if not records:
        return records, "Upload at least one STL first.", None
    total = len(records)
    messages: list[str] = []
    plan_by_record = _group_z_levels_by_record(records, layer_height, scale_mode, messages)
    for pos, record in enumerate(records):
        stl_path = record.get("stl_path")
        if not stl_path:
            if record.get("layer_stack") is not None:
                messages.append(f"Shape {record['idx']}: kept existing split-piece slices.")
            else:
                messages.append(f"Shape {record['idx']}: skipped (no STL file).")
            continue

        def report_progress(cur: int, tot: int, offset: int = pos) -> None:
            progress((offset + cur / tot) / total, desc=f"Slicing shape {offset + 1} of {total}...")

        try:
            stack = _slice_record(
                record, layer_height, scale_mode, report_progress, plan_by_record.get(id(record))
            )
            (x_min, y_min, _z_min), (x_max, y_max, _z_max) = stack.bounds
            messages.append(
                f"Shape {record['idx']}: sliced {len(stack.layers)} layers "
                f"(footprint {x_max - x_min:.2f} x {y_max - y_min:.2f} mm)."
            )
        except Exception as exc:
            messages.append(f"Shape {record['idx']}: failed ({exc}).")
    ref_layers = generate_dynamic_reference_stack(records, fil_width)
    if ref_layers is not None:
        messages.append("Reference layers: updated automatically.")
    else:
        messages.append("Reference layers: skipped (no sliced shapes available).")
    return records, "\n".join(messages), ref_layers


def generate_dynamic_reference_stack(
    records: list[dict] | None,
    fil_width: float = 0.8,
) -> LayerStack | None:
    # Snapping the alignment to the fil grid keeps split pieces' scan-grid
    # phase intact under shared reference motion (exact one-fil seam pitch).
    # Shapes sharing a nozzle are multi-material assembly parts: their group
    # frame makes them align as one rigid unit at their modeled positions.
    records = records or []
    _stamp_multi_material_frames(records, fil_width)
    return build_reference_stack(
        [record.get("layer_stack") for record in records],
        grid=float(fil_width) if fil_width else None,
    )


SPLIT_STATUS_DEFAULT = (
    "Pick a shape to split for multi-nozzle printing - shapes are sliced "
    "automatically when you split. Shapes that share a nozzle number are one "
    "multi-material assembly: selecting any of them splits the **whole "
    "group** together as one shape."
)


def describe_split_source(records: list[dict] | None, selected: str | None) -> str:
    """Split-source note: warn up front when the selection splits a whole group."""
    records = records or []
    pos = _selected_record_index(records, selected)
    if pos < 0 or pos >= len(records):
        return SPLIT_STATUS_DEFAULT
    source = records[pos]
    if not source.get("stl_path"):
        return SPLIT_STATUS_DEFAULT
    nozzle = _record_nozzle_number(source, int(source.get("idx", pos + 1) or (pos + 1)))
    members = _multi_material_groups(records).get(nozzle, [])
    if len(members) <= 1:
        return SPLIT_STATUS_DEFAULT
    names = ", ".join(
        "**{}**".format(member.get("name") or f"Shape {member.get('idx')}")
        for member in members
    )
    return (
        f"⚠️ This shape is part of the multi-material group on nozzle {nozzle} "
        f"({names}). Splitting it splits the **whole group as one shape**: every "
        "material is clipped by the same cells, each cell's pieces share a nozzle, "
        "and every piece gets its own valve."
    )


def _split_group_records(
    records: list[dict],
    group_members: list[dict],
    group_nozzle: int,
    split_column_count: int,
    split_row_count: int,
    overlapping_layers: bool,
    starting_nozzle: Any,
    starting_valve: Any,
    fil_width: float,
    selected: str | None,
    _outputs,
) -> tuple:
    """Split a whole multi-material group as one shape.

    Every material is clipped by the same cell grid over the group's
    combined bounds and one shared scan frame, so cell-mates assemble
    exactly. Pieces are emitted cell-major: each cell's pieces share a
    nozzle (making the cell a multi-material group again, with the same
    alignment and seam-free-contour behavior), and every piece gets its own
    valve. Cells where a material has no geometry are skipped.
    """
    unsliced = [
        member
        for member in group_members
        if member.get("layer_stack") is None or not member["layer_stack"].layers
    ]
    if unsliced:
        return _outputs(
            records,
            selected,
            f"Split failed: shape(s) on nozzle {group_nozzle} could not be sliced - "
            "check their STLs.",
        )
    stacks = [member["layer_stack"] for member in group_members]
    layer_counts = {len(stack.layers) for stack in stacks}
    z_starts = {round(stack.z_values[0], 6) for stack in stacks if stack.z_values}
    if len(layer_counts) != 1 or len(z_starts) > 1:
        return _outputs(
            records,
            selected,
            f"Split failed: the shapes on nozzle {group_nozzle} were sliced on different "
            "Z grids - edit a dimension (or nozzle) so the group re-slices together.",
        )

    # Stamp group frames + seam-free contours so pieces inherit contour
    # linework that already excludes material-to-material interfaces.
    _stamp_multi_material_frames(records, float(fil_width))
    frame = stacks[0].align_frame

    try:
        pieces_by_member = [
            (
                member,
                split_layer_stack_grid(
                    member["layer_stack"],
                    columns=split_column_count,
                    rows=split_row_count,
                    overlapping_layers=bool(overlapping_layers),
                    overlap=float(fil_width) if overlapping_layers else 0.0,
                    grid=float(fil_width),
                    frame=frame,
                ),
            )
            for member in group_members
        ]
    except Exception as exc:
        return _outputs(records, selected, f"Split failed: {exc}")

    first_member = group_members[0]
    first_nozzle = max(1, _coerce_int(starting_nozzle, 1))
    first_valve = max(
        1, _coerce_int(starting_valve, _coerce_int(first_member.get("valve", 4), 4))
    )
    split_group_id = f"split-{int(time.time() * 1_000_000)}-{first_member.get('idx', 1)}"
    cell_count = split_column_count * split_row_count
    split_records: list[dict] = []
    valve_cursor = first_valve
    for cell in range(cell_count):
        row_index = cell // split_column_count + 1
        col_index = cell % split_column_count + 1
        for member, pieces in pieces_by_member:
            piece = pieces[cell]
            if all(layer.is_empty for layer in piece.layers):
                continue  # this material has nothing in this cell
            (piece_x_min, piece_y_min, _z_min), (piece_x_max, piece_y_max, _z_max) = piece.bounds
            member_name = str(member.get("name") or f"Shape {member.get('idx')}")
            piece_record = dict(member)
            piece_record.update({
                "name": f"{member_name} - R{row_index}C{col_index}",
                "stl_path": None,
                "target_x": round((piece_x_max - piece_x_min) or member.get("target_x", DEFAULT_TARGET_EXTENTS[0]), 1),
                "target_y": round((piece_y_max - piece_y_min) or member.get("target_y", DEFAULT_TARGET_EXTENTS[1]), 1),
                "nozzle": first_nozzle + cell,
                "valve": valve_cursor,
                "split_group_id": split_group_id,
                "split_index": cell,
                "split_row": row_index,
                "split_col": col_index,
                "split_rows": split_row_count,
                "split_columns": split_column_count,
                "layer_stack": piece,
                "slice_params": member.get("slice_params"),
                "gcode_path": None,
            })
            valve_cursor += 1
            split_records.append(piece_record)

    member_ids = {id(member) for member in group_members}
    first_pos = min(
        index for index, record in enumerate(records) if id(record) in member_ids
    )
    kept = [record for record in records if id(record) not in member_ids]
    insert_at = sum(1 for record in records[:first_pos] if id(record) not in member_ids)
    next_records = _reindex_shape_records(
        [*kept[:insert_at], *split_records, *kept[insert_at:]]
    )
    split_selected = (
        _shape_choice(next_records[insert_at]) if insert_at < len(next_records) else None
    )
    status = (
        f"Split the multi-material group on nozzle {group_nozzle} "
        f"({len(group_members)} materials) as one shape into {cell_count} cells "
        f"({split_column_count} columns x {split_row_count} rows) - "
        f"{len(split_records)} piece(s); material-empty cells skipped.  \n"
        f"Each cell's pieces share a nozzle (nozzles {first_nozzle}-"
        f"{first_nozzle + cell_count - 1}); valves {first_valve}-{valve_cursor - 1}."
    )
    if overlapping_layers:
        status += (
            "  \nOverlapping Layers is enabled: split boundaries alternate by one "
            "filament width per layer so neighbouring pieces interlock."
        )
    return _outputs(next_records, split_selected, status)


def split_selected_shape_for_grid(
    records: list[dict] | None,
    selected: str | None,
    settings_table: Any | None,
    columns: float,
    rows: float,
    overlapping_layers: bool,
    starting_nozzle: float,
    starting_valve: float,
    fil_width: float,
    layer_height: float = 0.8,
    scale_mode: str | None = None,
    undo_stack: list[list[dict]] | None = None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    # One-button flow: shapes are sliced (or re-sliced when their settings
    # changed) straight from the table here, so splitting never needs a
    # separate slicing step.
    slice_messages: list[str] = []
    if records:
        _ensure_records_sliced(records, float(layer_height or 0.8), scale_mode, slice_messages)
    # Undo stack: each split PUSHES its pre-split records (sliced), and Undo
    # Split pops them one at a time (capped to the last 10 splits). Pushed
    # only when a split actually happened — every failure path returns the
    # ORIGINAL `records` list object, detected by identity.
    next_undo_stack = [*(undo_stack or []), [dict(record) for record in records]][-10:]

    def _outputs(next_records: list[dict], selected_value: str | None, status: str) -> tuple:
        if slice_messages:
            status = "  \n".join([*slice_messages, status])
        return (
            next_records,
            _shape_settings_rows(next_records),
            _dropdown_update(next_records, selected_value),
            [record.get("gcode_path") for record in next_records if record.get("gcode_path")],
            _gcode_dropdown_update(next_records),
            _gcode_dropdown_update(next_records, include_upload=True),
            _dropdown_update(next_records, selected_value),
            status,
            _gcode_zip_update(next_records),
            next_undo_stack if next_records is not records else gr.skip(),
        )

    if not records:
        return _outputs(records, None, "Add a shape before splitting it.")

    pos = _selected_record_index(records, selected)
    if pos < 0:
        pos = 0
    source = records[pos]
    stack = source.get("layer_stack")
    if stack is None or not getattr(stack, "layers", None):
        return _outputs(
            records,
            selected,
            f"Split failed: Shape {source.get('idx', pos + 1)} could not be sliced - check its STL.",
        )

    split_column_count = max(1, _coerce_int(columns, 2))
    split_row_count = max(1, _coerce_int(rows, 1))

    # A shape sharing its nozzle with others is one material of a
    # multi-material assembly: the whole group splits together as ONE shape,
    # every material clipped by the same cell grid over the group's combined
    # bounds. Cell-mates share a nozzle (so each cell is itself a
    # multi-material group); every piece keeps its own valve.
    group_nozzle = _record_nozzle_number(source, int(source.get("idx", pos + 1) or (pos + 1)))
    group_members = (
        _multi_material_groups(records).get(group_nozzle, [])
        if source.get("stl_path")
        else []
    )
    if len(group_members) > 1:
        return _split_group_records(
            records,
            group_members,
            group_nozzle,
            split_column_count,
            split_row_count,
            overlapping_layers,
            starting_nozzle,
            starting_valve,
            fil_width,
            selected,
            _outputs,
        )

    try:
        pieces = split_layer_stack_grid(
            stack,
            columns=split_column_count,
            rows=split_row_count,
            overlapping_layers=bool(overlapping_layers),
            overlap=float(fil_width) if overlapping_layers else 0.0,
            # Whole-fil cells (last piece absorbs the remainder): keeps the
            # required nozzle spacing uniform under shared reference motion.
            grid=float(fil_width),
        )
    except Exception as exc:
        return _outputs(records, selected, f"Split failed: {exc}")

    base_name = str(source.get("name") or f"Shape {source.get('idx', pos + 1)}")
    first_nozzle = max(1, _coerce_int(starting_nozzle, 1))
    first_valve = max(1, _coerce_int(starting_valve, _coerce_int(source.get("valve", 4), 4)))
    split_group_id = f"split-{int(time.time() * 1_000_000)}-{source.get('idx', pos + 1)}"
    split_records: list[dict] = []
    for index, piece in enumerate(pieces):
        row_index = index // split_column_count + 1
        col_index = index % split_column_count + 1
        (piece_x_min, piece_y_min, _z_min), (piece_x_max, piece_y_max, _z_max) = piece.bounds
        piece_record = dict(source)
        piece_record.update({
            "name": f"{base_name} - R{row_index}C{col_index}",
            "stl_path": None,
            "target_x": (piece_x_max - piece_x_min) or source.get("target_x", DEFAULT_TARGET_EXTENTS[0]),
            "target_y": (piece_y_max - piece_y_min) or source.get("target_y", DEFAULT_TARGET_EXTENTS[1]),
            "nozzle": first_nozzle + index,
            "valve": first_valve + index,
            "split_group_id": split_group_id,
            "split_index": index,
            "split_row": row_index,
            "split_col": col_index,
            "split_rows": split_row_count,
            "split_columns": split_column_count,
            "layer_stack": piece,
            "slice_params": source.get("slice_params"),
            "gcode_path": None,
        })
        split_records.append(piece_record)
    next_records = _reindex_shape_records([*records[:pos], *split_records, *records[pos + 1:]])
    split_selected = _shape_choice(next_records[pos]) if pos < len(next_records) else None
    status = (
        f"Split Shape {source.get('idx', pos + 1)} into {len(pieces)} print-ready piece stacks "
        f"({split_column_count} columns x {split_row_count} rows).  \n"
        f"Nozzles {first_nozzle}-{first_nozzle + len(pieces) - 1}; valves {first_valve}-{first_valve + len(pieces) - 1}."
    )
    if overlapping_layers:
        status += (
            "  \nOverlapping Layers is enabled: split boundaries alternate by one "
            "filament width per layer so neighbouring pieces interlock."
        )
    return _outputs(next_records, split_selected, status)


def _contour_tracing_sources(records: list[dict]) -> list[ContourSource]:
    sources: list[ContourSource] = []
    for record in records:
        if not record.get("contour_tracing"):
            continue
        stack = record.get("layer_stack")
        if stack is None or not getattr(stack, "layers", None):
            continue
        sources.append(
            ContourSource(
                owner_idx=int(record.get("idx", len(sources) + 1)),
                stack=stack,
            )
        )
    return sources


def _ensure_records_sliced(
    records: list[dict],
    layer_height: float,
    scale_mode: str | None,
    messages: list[str],
) -> bool:
    """Re-slice records whose layers are missing or stale for the current settings."""
    plan_by_record = _group_z_levels_by_record(records, layer_height, scale_mode)
    resliced = False
    for record in records:
        stl_path = record.get("stl_path")
        if not stl_path:
            continue  # Split pieces carry their clipped layers; nothing to re-slice.
        slice_plan = plan_by_record.get(id(record))
        current = _slice_params_snapshot(record, layer_height, scale_mode, slice_plan)
        if record.get("layer_stack") is not None and record.get("slice_params") == current:
            continue
        try:
            stack = _slice_record(record, layer_height, scale_mode, None, slice_plan)
            messages.append(
                f"Shape {record['idx']}: sliced automatically ({len(stack.layers)} layers)."
            )
        except Exception as exc:
            record["layer_stack"] = None
            messages.append(f"Shape {record['idx']}: slicing failed ({exc}).")
        resliced = True
    return resliced


def _lead_in_assembly_extension(records: list[dict], direction: str | None) -> float:
    """Extra lead-in clearance so split pieces purge clear of the assembly.

    Under shared reference motion every head executes the same lead-in
    offset, so head k's purge patch lands one cell over from head k-1's -
    right on a sibling's print area - unless the clearance exceeds the
    assembly's remaining extent along the purge axis. The split cells are
    equal sized, so that extent is (count - 1) * cell for the deepest piece.
    Returned as one batch-wide value (the max over all split pieces) so
    every shape's lead-in stays identical and the shared motion stays in
    sync.
    """
    axis = 0 if (direction or LEAD_IN_DIRECTION_LEFT) in ("Left", "Right") else 1
    extension = 0.0
    for record in records:
        if not record.get("split_group_id"):
            continue
        stack = record.get("layer_stack")
        if stack is None:
            continue
        (min_x, min_y, _z_min), (max_x, max_y, _z_max) = stack.bounds
        cell = (max_x - min_x) if axis == 0 else (max_y - min_y)
        count = _coerce_int(
            record.get("split_columns" if axis == 0 else "split_rows"), 1
        )
        extension = max(extension, cell * max(0, count - 1))
    return extension


def refresh_parallel_on_tab_select(mode: str | None, *args: Any) -> tuple:
    """Re-send the parallel LINE plot when the Visualization tab is opened.

    The generate chain renders the parallel view into the hidden tab, where
    Gradio's Plot component doesn't mount the figure — re-issuing it on tab
    select makes it appear. Tube renders are skipped: the user made those
    while ON the tab (already mounted), and they are expensive to redo.
    """
    if mode != "line":
        return gr.skip(), gr.skip()
    return render_dynamic_parallel(*args, tube=False)


def undo_last_split(undo_stack: list[list[dict]] | None) -> tuple:
    """Restore the shapes as they were before the most recent split.

    The undo slot is a STACK of pre-split snapshots (one per split, newest
    last), so repeated splits can be unwound one at a time."""
    stack = list(undo_stack or [])
    if not stack:
        return (
            *(gr.skip() for _ in range(7)),
            "Nothing to undo: no split has been made yet.",
            gr.skip(),
            gr.skip(),
        )
    snapshot = stack.pop()
    records = [dict(record) for record in snapshot]
    status = "Undid the last split: restored the shapes as they were before it."
    if stack:
        plural = "s" if len(stack) != 1 else ""
        status += f" ({len(stack)} earlier split{plural} can still be undone.)"
    return (
        records,
        _shape_settings_rows(records),
        _dropdown_update(records),
        [record.get("gcode_path") for record in records if record.get("gcode_path")],
        _gcode_dropdown_update(records),
        _gcode_dropdown_update(records, include_upload=True),
        _dropdown_update(records),
        status,
        _gcode_zip_update(records),
        stack or None,
    )


def _gcode_settings_snapshot(
    record: dict,
    raster_pattern: str | None,
    pressure_ramp_enabled: bool,
    lead_in_length: float,
    lead_in_clearance: float,
    lead_in_lines: float,
    lead_in_direction: str | None,
    layer_height: float,
    fil_width: float,
    scale_mode: str | None,
    sweep_buffer: float = 0.8,
    lead_in_orientation: str | None = None,
) -> dict:
    """Fingerprint of every setting that shapes this record's G-code.

    Stamped on the record at generation time; the staleness banner compares
    it against the CURRENT table/options so outdated files are never
    downloaded or printed unnoticed. Color is excluded (it never reaches
    the G-code)."""
    return {
        "targets": tuple(
            round(_coerce_float(record.get(key), 0.0), 6) for key in TARGET_DIMENSION_KEYS
        ),
        "pressure": round(_coerce_float(record.get("pressure"), 25.0), 6),
        "valve": _coerce_int(record.get("valve"), 4),
        "port": _coerce_int(record.get("port"), 1),
        "nozzle": _record_nozzle_number(record, int(record.get("idx", 1) or 1)),
        "infill": round(_coerce_float(record.get("infill"), 100.0), 6),
        "contour_tracing": bool(record.get("contour_tracing")),
        "lead_in": bool(record.get("lead_in")),
        "raster_pattern": str(raster_pattern or ""),
        "pressure_ramp": bool(pressure_ramp_enabled),
        "lead_in_params": (
            round(_coerce_float(lead_in_length, 5.0), 6),
            round(_coerce_float(lead_in_clearance, 5.0), 6),
            _coerce_int(lead_in_lines, 3),
            str(lead_in_direction or ""),
            str(lead_in_orientation or LEAD_IN_LINE_AUTO),
        ),
        "layer_height": round(_coerce_float(layer_height, 0.8), 6),
        "fil_width": round(_coerce_float(fil_width, 0.8), 6),
        "scale_mode": _normalize_scale_mode(scale_mode),
        "sweep_buffer": round(_coerce_float(sweep_buffer, 0.8), 6),
    }


GCODE_STALE_MESSAGE = (
    "&#9888;&#65039; **Settings changed since this G-code was generated** — "
    "press Generate G-Code again before downloading, visualizing, or printing."
)


def check_gcode_staleness(
    records: list[dict] | None,
    settings_table: Any,
    raster_pattern: str | None,
    pressure_ramp_enabled: bool,
    lead_in_length: float,
    lead_in_clearance: float,
    lead_in_lines: float,
    lead_in_direction: str | None,
    layer_height: float,
    fil_width: float,
    scale_mode: str | None,
    sweep_buffer: float = 0.8,
    lead_in_orientation: str | None = None,
) -> str:
    """Warning banner text when generated G-code no longer matches the settings."""
    records = _apply_shape_settings(records or [], settings_table)
    if not any(record.get("gcode_path") for record in records):
        return ""  # nothing generated yet: nothing can be stale
    snapshot_args = (
        raster_pattern,
        pressure_ramp_enabled,
        lead_in_length,
        lead_in_clearance,
        lead_in_lines,
        lead_in_direction,
        layer_height,
        fil_width,
        scale_mode,
        sweep_buffer,
        lead_in_orientation,
    )
    for record in records:
        if not record.get("gcode_path"):
            # A shape added after the last generation has no file at all.
            return GCODE_STALE_MESSAGE
        if record.get("gcode_snapshot") != _gcode_settings_snapshot(record, *snapshot_args):
            return GCODE_STALE_MESSAGE
    return ""


_SHAPE_EXPORT_FIELDS = (
    "name",
    "target_x",
    "target_y",
    "target_z",
    "pressure",
    "valve",
    "nozzle",
    "port",
    "color",
    "infill",
    "contour_tracing",
    "lead_in",
)


def export_project_settings(
    records: list[dict] | None,
    settings_table: Any,
    layer_height: float,
    fil_width: float,
    scale_mode: str | None,
    raster_pattern: str | None,
    pressure_ramp_enabled: bool,
    sweep_buffer: float,
    lead_in_length: float,
    lead_in_clearance: float,
    lead_in_lines: float,
    lead_in_direction: str | None,
    lead_in_orientation: str | None,
    nozzle_speed: Any,
) -> tuple[str | None, str]:
    """Write the session's settings to a small JSON file, keyed by STL name.

    Sessions are otherwise lost on a page refresh or Space restart: the
    export carries every per-shape table setting plus the generation
    options; re-upload the same STLs and import to restore them. Split
    pieces are derived geometry and cannot round-trip.
    """
    records = _apply_shape_settings(records or [], settings_table)
    shapes = []
    for record in records:
        if not record.get("stl_path"):
            continue
        entry = {"file": Path(str(record["stl_path"])).name}
        for field in _SHAPE_EXPORT_FIELDS:
            entry[field] = record.get(field)
        shapes.append(entry)
    if not shapes:
        return None, "Nothing to export: load at least one STL first."
    payload = {
        "app": "ParallelPrint",
        "version": 1,
        "shapes": shapes,
        "generation": {
            "layer_height": _coerce_float(layer_height, 0.8),
            "fil_width": _coerce_float(fil_width, 0.8),
            "scale_mode": _normalize_scale_mode(scale_mode),
            "raster_pattern": str(raster_pattern or ""),
            "pressure_ramp_enabled": bool(pressure_ramp_enabled),
            "sweep_buffer": _coerce_float(sweep_buffer, 0.8),
            "lead_in_length": _coerce_float(lead_in_length, 5.0),
            "lead_in_clearance": _coerce_float(lead_in_clearance, 5.0),
            "lead_in_lines": _coerce_int(lead_in_lines, 3),
            "lead_in_direction": str(lead_in_direction or LEAD_IN_DIRECTION_LEFT),
            "lead_in_orientation": str(lead_in_orientation or LEAD_IN_LINE_AUTO),
            "nozzle_speed": _coerce_float(nozzle_speed, 10.0),
        },
    }
    settings_path = Path(tempfile.mkdtemp(prefix="pp_settings_")) / "parallelprint_settings.json"
    settings_path.write_text(json.dumps(payload, indent=2))
    status = f"Exported settings for {len(shapes)} shape(s) and the generation options."
    if any(not record.get("stl_path") for record in records):
        status += " Split pieces are not exported — re-split after importing."
    return str(settings_path), status


def import_project_settings(
    settings_upload: Any,
    records: list[dict] | None,
    settings_table: Any,
) -> tuple:
    """Apply an exported settings file to the loaded shapes and options.

    Shape settings match by STL filename (duplicates apply in order); files
    in the export that are not currently loaded are reported so they can be
    uploaded and re-imported. Generation options apply regardless.
    """

    def _skip_options() -> tuple:
        return tuple(gr.skip() for _ in range(12))

    paths = _uploaded_file_paths(settings_upload)
    if not paths:
        return gr.skip(), gr.skip(), "", *_skip_options()
    try:
        payload = json.loads(Path(paths[0]).read_text())
        if not isinstance(payload, dict) or "shapes" not in payload:
            raise ValueError("not a ParallelPrint settings file")
    except (OSError, ValueError) as exc:
        return gr.skip(), gr.skip(), f"Import failed: {exc}", *_skip_options()

    records = _apply_shape_settings(records or [], settings_table)
    queues: dict[str, list[dict]] = {}
    for entry in payload.get("shapes", []):
        if isinstance(entry, dict) and entry.get("file"):
            queues.setdefault(str(entry["file"]), []).append(entry)

    updated: list[dict] = []
    applied = 0
    for record in records:
        copy = dict(record)
        filename = Path(str(copy.get("stl_path") or "")).name
        queue = queues.get(filename)
        if queue:
            entry = queue.pop(0)
            for field in _SHAPE_EXPORT_FIELDS:
                if field in entry and entry[field] is not None:
                    copy[field] = entry[field]
            _round_targets_to_tenths(copy)
            applied += 1
        updated.append(copy)
    missing = sorted({name for name, queue in queues.items() if queue})

    generation = payload.get("generation", {}) if isinstance(payload.get("generation"), dict) else {}

    def option(key: str):
        return gr.update(value=generation[key]) if key in generation else gr.skip()

    messages = [f"Imported settings for {applied} shape(s)."]
    if generation:
        messages.append("Generation options restored.")
    if missing:
        messages.append(
            "Not currently loaded (upload them, then import again): " + ", ".join(f"`{name}`" for name in missing) + "."
        )
    return (
        updated,
        _shape_settings_rows(updated),
        "  \n".join(messages),
        option("raster_pattern"),
        option("pressure_ramp_enabled"),
        option("sweep_buffer"),
        option("lead_in_length"),
        option("lead_in_clearance"),
        option("lead_in_lines"),
        option("lead_in_direction"),
        option("lead_in_orientation"),
        option("layer_height"),
        option("fil_width"),
        option("scale_mode"),
        option("nozzle_speed"),
    )


def assign_unique_valves(
    records: list[dict] | None, settings_table: Any
) -> tuple[list[dict], list[list[Any]]]:
    """Give every shape its own valve: first come keeps its number, later
    duplicates (and unset valves) move to the smallest unused number."""
    records = _apply_shape_settings(records or [], settings_table)
    updated = [dict(record) for record in records]
    claimed: set[int] = set()
    reassign: list[dict] = []
    for record in updated:
        valve = _coerce_int(record.get("valve"), 0)
        if valve > 0 and valve not in claimed:
            claimed.add(valve)
        else:
            reassign.append(record)
    for record in reassign:
        valve = _next_unused_valve(claimed)
        record["valve"] = valve
        claimed.add(valve)
    return updated, _shape_settings_rows(updated)


def generate_dynamic_gcode(
    records: list[dict] | None,
    settings_table: Any,
    raster_pattern: str | None,
    pressure_ramp_enabled: bool,
    lead_in_length: float,
    lead_in_clearance: float,
    lead_in_lines: float,
    lead_in_direction: str | None,
    layer_height: float,
    fil_width: float,
    scale_mode: str | None,
    nozzle_speed: Any = None,
    sweep_buffer: float = 0.8,
    lead_in_orientation: str | None = None,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    messages: list[str] = []
    progress(0.02, desc="Slicing shapes…")
    _ensure_records_sliced(records, layer_height, scale_mode, messages)
    # Shared reference motion is always on: every head traces the combined
    # outline and dispenses only its own geometry. Rebuilt with the CURRENT
    # fil width so the alignment snap grid matches this generation.
    progress(0.12, desc="Building the shared reference outline…")
    ref_layers = generate_dynamic_reference_stack(records, fil_width)
    contour_sources = _contour_tracing_sources(records)
    if contour_sources:
        enabled = ", ".join(f"Shape {source.owner_idx}" for source in contour_sources)
        messages.append(f"Contour tracing enabled for {enabled}.")
    # Circle Spiral under shared motion: every shape's own wall radius joins
    # the one shared ring set (same list for every shape, so motion stays in
    # sync). Split pieces carry a scan frame and are excluded.
    wall_sources = [
        record["layer_stack"]
        for record in records
        if record.get("layer_stack") is not None
        and getattr(record["layer_stack"], "scan_frame", None) is None
    ]
    # Infill motion optimization: raster lines/rings that NO head dispenses
    # on are dropped from the shared motion (e.g. every shape at 50% halves
    # the path). Same list for every shape, so the shared motion stays in
    # sync across heads.
    motion_infill_fractions = [
        _coerce_float(record.get("infill", 100.0), 100.0) / 100.0
        for record in records
        if record.get("layer_stack") is not None
        and getattr(record["layer_stack"], "layers", None)
    ]
    # Lead-in is driven entirely by the per-shape "Lead In" column: the
    # purge motion exists whenever any shape dispenses it (all heads must
    # share the motion), and each shape's own flag gates its valve.
    lead_in_enabled = any(record.get("lead_in") for record in records)
    effective_lead_in_clearance = float(lead_in_clearance or 0.0)
    if lead_in_enabled:
        lead_in_extension = _lead_in_assembly_extension(records, lead_in_direction)
        if lead_in_extension > 0.0:
            effective_lead_in_clearance += lead_in_extension
            messages.append(
                f"Lead-in clearance extended by {lead_in_extension:.1f} mm so every "
                "nozzle's purge patch lands clear of the split assembly."
            )
    # The pressure regulator is a PORT device: when shapes share a serial
    # port, exactly ONE of their files owns the pressure commands (preset,
    # toggle, per-layer ramp) — the print host compiles every file onto one
    # timeline and duplicated toggles would flip the regulator on/off/on.
    ports_owned: set[int] = set()
    generatable = sum(
        1
        for record in records
        if record.get("layer_stack") is not None and getattr(record["layer_stack"], "layers", None)
    )
    generated_count = 0
    for record in records:
        stack = record.get("layer_stack")
        if stack is None or not getattr(stack, "layers", None):
            messages.append(f"Shape {record['idx']}: skipped (no sliced layers).")
            continue
        if ref_layers is None:
            messages.append(f"Shape {record['idx']}: skipped (no combined reference outline; slice a shape first).")
            continue
        shape_name = str(record.get("name") or stack.name or f"shape_{record['idx']}").replace(" ", "_")
        generated_count += 1
        progress(
            0.15 + 0.8 * (generated_count - 1) / max(1, generatable),
            desc=f"Generating {shape_name} ({generated_count}/{generatable})…",
        )
        port_number = _coerce_int(record.get("port"), 1)
        owns_port_pressure = port_number not in ports_owned
        try:
            gcode_path = generate_vector_gcode(
                stack,
                shape_name=shape_name,
                pressure=float(record.get("pressure", 25.0)),
                valve=int(record.get("valve", 4)),
                port=int(record.get("port", 1)),
                layer_height=float(layer_height),
                fil_width=float(fil_width),
                pressure_ramp_enabled=bool(pressure_ramp_enabled),
                # Always one constant speed: every move is emitted as G1 (no
                # G0 rapid travel); the valve commands still mark print vs
                # travel.
                all_g1=True,
                motion=ref_layers,
                raster_pattern=raster_pattern,
                contour_sources=contour_sources,
                active_contour_owner=int(record.get("idx", 0)),
                infill=_coerce_float(record.get("infill", 100.0), 100.0) / 100.0,
                motion_infill_fractions=motion_infill_fractions,
                emit_pressure_commands=owns_port_pressure,
                # Same buffer for every shape: the shared motion must match.
                sweep_buffer=max(0.0, _coerce_float(sweep_buffer, 0.8)),
                lead_in_enabled=bool(lead_in_enabled),
                lead_in_length=float(lead_in_length),
                lead_in_clearance=effective_lead_in_clearance,
                lead_in_lines=max(1, _coerce_int(lead_in_lines, 3)),
                lead_in_direction=lead_in_direction or LEAD_IN_DIRECTION_LEFT,
                lead_in_orientation=lead_in_orientation,
                lead_in_dispense=bool(record.get("lead_in", True)),
                wall_sources=wall_sources,
                origin_sink=(origin_sink := {}),
            )
            ports_owned.add(port_number)
            record["gcode_path"] = str(gcode_path)
            # World anchor of the toolpath (kept OUT of the G-code file):
            # Auto Align and the visualizations read it from the record.
            record["path_origin"] = origin_sink.get("path_origin")
            # Fingerprint the settings this file was generated with, so the
            # staleness banner can flag later edits.
            record["gcode_snapshot"] = _gcode_settings_snapshot(
                record,
                raster_pattern,
                pressure_ramp_enabled,
                lead_in_length,
                lead_in_clearance,
                lead_in_lines,
                lead_in_direction,
                layer_height,
                fil_width,
                scale_mode,
                sweep_buffer,
                lead_in_orientation,
            )
            messages.append(f"Shape {record['idx']}: wrote `{gcode_path.name}`.")
        except Exception as exc:
            messages.append(f"Shape {record['idx']}: failed ({exc}).")

    generated_paths = [record.get("gcode_path") for record in records if record.get("gcode_path")]
    if generated_paths:
        # All heads share one motion path, so one file's length is the job's.
        try:
            shared_length = _parsed_path_length(parse_gcode_path(Path(generated_paths[0]).read_text()))
        except OSError:
            shared_length = 0.0
        speed = _coerce_float(nozzle_speed, 0.0) or 10.0
        estimate = _print_time_estimate(shared_length, speed)
        if estimate:
            messages.insert(
                0,
                f"**Print path {shared_length:,.0f} mm — about {estimate} at {speed:g} mm/s** "
                "(constant speed; the Visualization tab's Nozzle Speed sets the rate).",
            )
    progress(1.0, desc="Done")
    return (
        records,
        ref_layers,
        generated_paths,
        "\n".join(messages),
        _gcode_dropdown_update(records),
        _gcode_dropdown_update(records, include_upload=True),
        _gcode_zip_update(records),
    )


def load_selected_gcode_text(records: list[dict] | None, selected: str | None) -> str:
    pos = _selected_record_index(records or [], selected)
    if pos < 0:
        return "# No G-code generated yet."
    path = (records or [])[pos].get("gcode_path")
    if not path:
        return f"# No G-code generated for Shape {(records or [])[pos].get('idx', 1)} yet."
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"# Failed to read G-code file: {exc}"


def _parts_from_records(records: list[dict] | None) -> tuple[list[dict], list[str]]:
    parts: list[dict] = []
    messages: list[str] = []
    for record in records or []:
        path = record.get("gcode_path")
        idx = int(record.get("idx", len(parts) + 1))
        if not path:
            messages.append(f"Shape {idx}: no G-code (press Generate G-Code on the Shapes & G-Code tab).")
            continue
        try:
            parsed = parse_gcode_path(Path(path).read_text())
        except OSError as exc:
            messages.append(f"Shape {idx}: failed to read ({exc}).")
            continue
        if not parsed.get("point_count"):
            messages.append(f"Shape {idx}: no G0/G1 moves found.")
            continue
        # The toolpath's world anchor lives on the RECORD (not in the G-code
        # file); legacy files that still carry a "; PathOrigin" comment keep
        # working through the parser.
        if parsed.get("path_origin") is None and record.get("path_origin"):
            parsed["path_origin"] = tuple(record["path_origin"])
        nozzle = _record_nozzle_number(record, idx)
        parts.append({"idx": idx, "nozzle": nozzle, "color": record.get("color", _default_color(idx)), "parsed": parsed})
        messages.append(f"Shape {idx} (Nozzle {nozzle}): {parsed['point_count']} moves, {parsed.get('layer_count', 0)} layer(s).")
    return parts, messages


def _parsed_path_length(parsed: dict) -> float:
    """Total tool-path length (print + travel) of a parsed G-code file, mm."""
    total = 0.0
    for key in ("print_segments", "travel_segments"):
        for segment in parsed.get(key) or []:
            for start, end in zip(segment, segment[1:]):
                total += math.dist(start, end)
    return total


def _format_duration(total_seconds: float) -> str:
    hours, remainder = divmod(int(round(total_seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds:02d} s"
    return f"{seconds} s"


def _print_time_estimate(length_mm: float, nozzle_speed: Any) -> str | None:
    """Human-readable print duration at a constant nozzle speed, or None.

    Every move is emitted as G1 at one constant speed, so time is simply
    path length / speed (valve switching is assumed instantaneous)."""
    speed = _coerce_float(nozzle_speed, 0.0)
    if speed <= 0.0 or length_mm <= 0.0:
        return None
    return _format_duration(length_mm / speed)


def render_dynamic_nozzle_spacing(
    records: list[dict] | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
) -> tuple[Any, str]:
    parts, _messages = _parts_from_records(records)
    if not parts:
        return None, "No shape G-code available. Generate G-code first."
    offsets, spacings = _resolve_nozzle_grid_layout(
        parts,
        columns,
        rows,
        column_spacing,
        row_spacing,
        use_grid_individual_spacing,
        grid_spacing_table,
    )
    return build_nozzle_spacing_figure(parts, offsets, spacings), _format_nozzle_spacing_status(parts, offsets, spacings)


def render_dynamic_toolpath(
    source: str | None,
    uploaded_path: str | None,
    records: list[dict] | None,
    travel_opacity: float,
    print_opacity: float,
    print_width: float,
    travel_width: float,
    nozzle_speed: Any = None,
    tube: bool = True,
) -> tuple[Any, str, dict]:
    # Print color comes from the shape's table Color (uploads default to
    # orange); travel is always grey. No per-view color options.
    travel_color = TOOLPATH_TRAVEL_COLOR
    print_color = TOOLPATH_UPLOAD_PRINT_COLOR
    if source == GCODE_SOURCE_UPLOAD:
        path = uploaded_path
        label = "uploaded file"
    else:
        pos = _selected_record_index(records or [], source)
        record = (records or [])[pos] if pos >= 0 else {}
        path = record.get("gcode_path")
        label = source or "selected shape"
        print_color = str(record.get("color") or print_color)
    if not path:
        return None, f"No G-code available for {label}.", {}
    try:
        parsed = parse_gcode_path(Path(path).read_text())
    except OSError as exc:
        return None, f"Failed to read G-code file: {exc}", {}
    if parsed["point_count"] == 0:
        return None, "No G0/G1 movement lines found in the file.", {}
    figure = build_toolpath_figure(
        parsed,
        travel_opacity=travel_opacity,
        print_opacity=print_opacity,
        travel_color=travel_color,
        print_color=print_color,
        print_width=print_width,
        travel_width=travel_width,
        tube=tube,
    )
    (x_min, y_min, z_min), (x_max, y_max, z_max) = parsed["bounds"]
    status = (
        f"**{parsed['point_count']} moves parsed** - {len(parsed['print_segments'])} print segment(s), "
        f"{len(parsed['travel_segments'])} travel segment(s).  \n"
        f"Bounds: X [{x_min:.2f}, {x_max:.2f}], Y [{y_min:.2f}, {y_max:.2f}], Z [{z_min:.2f}, {z_max:.2f}] mm."
    )
    path_length = _parsed_path_length(parsed)
    estimate = _print_time_estimate(path_length, nozzle_speed)
    if estimate:
        status += (
            f"  \nPath length {path_length:,.0f} mm - estimated print time at "
            f"{_coerce_float(nozzle_speed, 0.0):g} mm/s: **{estimate}**."
        )
    return figure, status, parsed


def render_dynamic_toolpath_lines(*args: Any) -> tuple[Any, str, dict, str, dict[str, Any], dict[str, Any]]:
    figure, status, parsed = render_dynamic_toolpath(*args, tube=False)
    return figure, status, parsed, "line", gr.update(visible=False), gr.update(visible=False)


def render_dynamic_toolpath_tubes(*args: Any) -> tuple[Any, str, dict, str, dict[str, Any], dict[str, Any]]:
    figure, status, parsed = render_dynamic_toolpath(*args, tube=True)
    has_animation = bool(parsed.get("point_count"))
    return figure, status, parsed, "tube", gr.update(visible=has_animation), gr.update(visible=has_animation)


def rerender_dynamic_toolpath_current_mode(mode: str, *args: Any) -> tuple[Any, str, dict]:
    return render_dynamic_toolpath(*args, tube=(mode != "line"))


def render_dynamic_parallel(
    records: list[dict] | None,
    settings_table: Any,
    travel_opacity: float,
    filament_width: float,
    travel_width: float,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
    nozzle_speed: Any = None,
    tube: bool = True,
) -> tuple[Any, str]:
    records = _apply_shape_settings(records or [], settings_table)
    parts, messages = _parts_from_records(records)
    if not parts:
        return None, "No shape G-code available. Press Generate G-Code on the Shapes & G-Code tab first."
    offsets, spacings = _resolve_nozzle_grid_layout(
        parts,
        columns,
        rows,
        column_spacing,
        row_spacing,
        use_grid_individual_spacing,
        grid_spacing_table,
    )
    figure = build_parallel_figure(
        parts,
        part_offsets=offsets,
        filament_width=float(filament_width),
        travel_width=float(travel_width),
        travel_opacity=float(travel_opacity),
        tube=tube,
    )
    # All heads share one motion path, so the print takes as long as the
    # longest part's tool path at the constant nozzle speed.
    longest = max(_parsed_path_length(part["parsed"]) for part in parts)
    estimate = _print_time_estimate(longest, nozzle_speed)
    if estimate:
        messages.append(
            f"Estimated print time at {_coerce_float(nozzle_speed, 0.0):g} mm/s: **{estimate}** "
            f"({longest:,.0f} mm per nozzle; all heads move together)."
        )
    messages.append(_format_nozzle_spacing_status(parts, offsets, spacings))
    return figure, "  \n".join(messages)


def render_dynamic_parallel_lines(*args: Any) -> tuple[Any, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    figure, status = render_dynamic_parallel(*args, tube=False)
    has_data = figure is not None
    return figure, status, "line", gr.update(visible=False), gr.update(visible=False), gr.update(visible=has_data)


def render_dynamic_parallel_tubes(*args: Any) -> tuple[Any, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    figure, status = render_dynamic_parallel(*args, tube=True)
    has_anim = figure is not None
    return figure, status, "tube", gr.update(visible=has_anim), gr.update(visible=has_anim), gr.update(visible=has_anim)


def rerender_dynamic_parallel_current_mode(mode: str, *args: Any) -> tuple[Any, str]:
    return render_dynamic_parallel(*args, tube=(mode != "line"))


def export_dynamic_parallel_gif(
    records: list[dict] | None,
    settings_table: Any,
    travel_opacity: float,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
    duration: float,
    fps: float,
    elev: float,
    azim: float,
    progress: gr.Progress = gr.Progress(),
) -> str | None:
    records = _apply_shape_settings(records or [], settings_table)
    parts, _messages = _parts_from_records(records)
    if not parts:
        return None
    offsets, _spacings = _resolve_nozzle_grid_layout(
        parts,
        columns,
        rows,
        column_spacing,
        row_spacing,
        use_grid_individual_spacing,
        grid_spacing_table,
    )

    def report(frame: int, total: int) -> None:
        progress(frame / max(total, 1), desc=f"Rendering GIF frame {frame}/{total}")

    out_path = Path(tempfile.mkdtemp(prefix="parallel_gif_")) / "parallel_print.gif"
    result = build_parallel_gif(
        parts,
        out_path=out_path,
        part_offsets=offsets,
        travel_opacity=float(travel_opacity),
        duration=float(duration),
        fps=int(fps),
        elev=float(elev),
        azim=float(azim),
        progress_cb=report,
    )
    return str(result) if result else None

def build_dynamic_demo() -> gr.Blocks:
    with gr.Blocks(title="ParallelPrint: STL to G-Code", css=APP_CSS, head=APP_HEAD + TOOLPATH_ANIM_HEAD + PARALLEL_ANIM_HEAD) as demo:
        shape_records = gr.State([])
        last_shape_delete_at = gr.State(0.0)
        ref_layers = gr.State(None)

        with gr.Tab("Shapes & G-Code"):
            gr.Markdown(
                """
                # Shapes & G-Code
                Upload any number of STL files, edit per-shape dimensions and print settings in the table, then press **Generate G-Code** below - every shape is sliced automatically as part of generation (and before a split).
                Pick each shape's plot color straight from the dropdown in the **Color** column.
                """
            )
            with gr.Row():
                stl_upload = gr.File(
                    label="STL Files",
                    file_types=[".stl"],
                    type="filepath",
                    file_count="multiple",
                    allow_reordering=True,
                )
                with gr.Column(scale=0, min_width=200):
                    load_samples_button = gr.Button("Load Sample STLs", variant="secondary", size="sm", elem_id="load-sample-stls-button")
                    sample_set_selector = gr.Dropdown(
                        choices=list(SAMPLE_STL_SETS),
                        value=DEFAULT_SAMPLE_STL_SET,
                        label="Sample Set",
                        container=False,
                    )
                    sync_uploads_button = gr.Button("Sync Uploaded STLs", variant="secondary", size="sm")
                    reset_dimensions_button = gr.Button("Reset Dimensions", variant="secondary", size="sm")
                    assign_valves_button = gr.Button("Assign Unique Valves", variant="secondary", size="sm")
                    scale_mode = gr.Radio(
                        choices=[SCALE_MODE_TARGET_DIMENSIONS, SCALE_MODE_UNIFORM_FACTOR],
                        value=SCALE_MODE_TARGET_DIMENSIONS,
                        label="Scaling Mode",
                    )

            with gr.Accordion("Save / Load Settings", open=False, elem_classes=["settings-accordion"]):
                gr.Markdown(
                    "Sessions are lost on a page refresh, so settings travel as a small JSON file "
                    "keyed by STL filename: **Export** downloads every table setting plus the "
                    "generation options; re-upload the same STLs later and **Import** restores them."
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=220):
                        export_settings_button = gr.Button("Export Settings", variant="secondary")
                        settings_export_file = gr.File(label="Settings File", interactive=False, height=110)
                    with gr.Column(scale=1, min_width=220):
                        settings_import_upload = gr.File(
                            label="Import Settings (.json)",
                            file_types=[".json"],
                            interactive=True,
                            height=110,
                        )
                settings_status = gr.Markdown("")

            # Visually hidden (not visible=False: Gradio would omit them
            # from the DOM entirely and the color-select relay needs them).
            color_sink = gr.Textbox(
                label="color sink",
                container=False,
                elem_id="pp-color-sink",
                elem_classes=["pp-visually-hidden"],
            )
            color_apply = gr.Button(
                "apply color",
                elem_id="pp-color-apply",
                elem_classes=["pp-visually-hidden"],
            )
            bulk_sink = gr.Textbox(
                label="bulk sink",
                container=False,
                elem_id="pp-bulk-sink",
                elem_classes=["pp-visually-hidden"],
            )
            bulk_apply = gr.Button(
                "apply bulk",
                elem_id="pp-bulk-apply",
                elem_classes=["pp-visually-hidden"],
            )
            shape_settings = gr.Dataframe(
                headers=SHAPE_SETTINGS_HEADERS,
                value=[],
                row_count=(0, "dynamic"),
                column_count=(len(SHAPE_SETTINGS_HEADERS), "fixed"),
                datatype=SHAPE_SETTINGS_DATATYPES,
                interactive=True,
                static_columns=[SHAPE_SETTINGS_HEADERS.index("Color")],
                label="Shape Settings",
                elem_id="shape-settings-table",
                # Narrow fixed widths; two-word headers ("Pressure (psi)",
                # "Contour Tracing", ...) wrap to two lines via CSS. Each
                # column must fit its longest WORD in the widest host font
                # (the Space renders a wide monospace) plus the header
                # checkbox — too narrow and Gradio ellipsizes ("Con...").
                column_widths=[
                    "64px",   # Shape
                    "154px",  # STL
                    "62px",   # X (mm)
                    "62px",   # Y (mm)
                    "62px",   # Z (mm)
                    "92px",   # Pressure (psi)
                    "62px",   # Valve
                    "68px",   # Nozzle
                    "58px",   # Port
                    "104px",  # Color
                    "72px",   # Infill %
                    "112px",  # Contour Tracing
                    "80px",   # Lead In
                    "68px",   # Delete
                ],
            )
            with gr.Row():
                layer_height = gr.Number(label="Layer Height (mm)", value=0.8, minimum=0.0001, step=0.01)
                fil_width = gr.Number(label="Filament/Line Width (mm)", value=0.8, minimum=0.0001, step=0.01)

            with gr.Accordion("Multi-Nozzle Split", open=False, elem_classes=["settings-accordion"]):
                with gr.Row():
                    split_source = gr.Dropdown(label="Source Shape", choices=[], value=None, allow_custom_value=False)
                    split_refresh_sources = gr.Button("Refresh Source Shapes", variant="secondary", size="sm")
                with gr.Row():
                    split_columns = gr.Number(label="Columns (X)", value=2, minimum=1, step=1)
                    split_rows = gr.Number(label="Rows (Y)", value=1, minimum=1, step=1)
                    split_start_nozzle = gr.Number(label="Starting Nozzle", value=1, minimum=1, step=1)
                    split_start_valve = gr.Number(label="Starting Valve", value=4, minimum=1, step=1)
                split_overlapping_layers = gr.Checkbox(label="Overlapping Layers", value=False)
                with gr.Row():
                    split_button = gr.Button("Split Selected Shape into Grid Pieces", variant="primary", scale=3)
                    split_undo_button = gr.Button("Undo Split", variant="secondary", size="sm", scale=1, min_width=110)
                split_status = gr.Markdown(SPLIT_STATUS_DEFAULT)
                split_undo = gr.State(None)

            with gr.Accordion("Selected Shape Preview", open=False, elem_classes=["settings-accordion"]):
                with gr.Row():
                    selected_shape = gr.Dropdown(label="Preview Shape", choices=[], value=None, allow_custom_value=False)
                    refresh_preview_button = gr.Button("Regenerate Preview", variant="secondary", size="sm")

                with gr.Row():
                    with gr.Column(scale=2, min_width=420):
                        model_viewer = gr.Model3D(
                            label="Selected 3D Viewer",
                            display_mode="solid",
                            clear_color=(0.94, 0.95, 0.97, 1.0),
                            camera_position=FRONT_CAMERA,
                            height=360,
                        )
                    with gr.Column(scale=2, min_width=380):
                        layer_preview_plot = gr.Plot(label="Sliced Layer Preview")
                        layer_preview_slider = gr.Slider(
                            label="Layer",
                            minimum=1,
                            maximum=1,
                            step=1,
                            value=1,
                        )
                    with gr.Column(scale=1, min_width=300):
                        model_details = gr.Markdown("No model loaded.")

            gr.Markdown(
                """
                ---
                ### Generate G-Code
                One button slices every shape from the current table settings and writes its G-code. Pressure, valve, nozzle, port, and color come from the Shape Settings table.
                All shapes share one combined nozzle path (each dispenses only its own geometry), so parallel heads always stay in sync.
                """
            )
            gcode_pressure_ramp_enabled = gr.Checkbox(label="Increase Pressure Each Layer", value=True)
            with gr.Row(elem_id="gcode-raster-row"):
                gcode_raster_pattern = gr.Dropdown(
                    label="Raster Pattern",
                    choices=list(RASTER_PATTERN_CHOICES),
                    value=RASTER_PATTERN_SAME_DIRECTION,
                    allow_custom_value=False,
                    scale=3,
                    info="Tool path style used for every layer.",
                )
                gcode_sweep_buffer = gr.Number(
                    label="Sweep Buffer (mm)",
                    value=0.8,
                    minimum=0.0,
                    step=0.1,
                    scale=1,
                    min_width=170,
                    info="Valve-settle travel before/after each raster line (0 = none).",
                )
            with gr.Accordion("Lead In Options", open=False, elem_classes=["settings-accordion"]):
                gr.Markdown("Applies to shapes with **Lead In** checked in the Shape Settings table.")
                with gr.Row():
                    gcode_lead_in_length = gr.Number(label="Lead In Length (mm)", value=5.0, minimum=0.1, step=0.1, info="Length of each purge stroke.")
                    gcode_lead_in_clearance = gr.Number(
                        label="Lead In Clearance (mm)",
                        value=5.0,
                        minimum=0.0,
                        step=0.1,
                        info="Distance between the shape and the purge patch.",
                    )
                    gcode_lead_in_lines = gr.Number(label="Lead In Raster Lines", value=3, minimum=1, step=1, info="Number of purge strokes in the patch.")
                with gr.Row():
                    gcode_lead_in_direction = gr.Dropdown(
                        label="Lead In Position",
                        choices=list(LEAD_IN_DIRECTION_CHOICES),
                        value=LEAD_IN_DIRECTION_LEFT,
                        allow_custom_value=False,
                        info="Which side of the shape the purge patch sits on.",
                    )
                    gcode_lead_in_orientation = gr.Dropdown(
                        label="Lead In Line Direction",
                        choices=list(LEAD_IN_LINE_CHOICES),
                        value=LEAD_IN_LINE_AUTO,
                        allow_custom_value=False,
                        info="Which way the purge strokes run within the patch.",
                    )
            gcode_stale_banner = gr.Markdown("", elem_id="gcode-stale-banner")
            gcode_button = gr.Button("Generate G-Code", variant="primary")
            with gr.Row():
                with gr.Column(scale=4):
                    gcode_downloads = gr.File(label="Download G-Code Files", file_count="multiple", interactive=False, elem_classes=["gcode-download"])
                with gr.Column(scale=1, min_width=200):
                    gcode_download_all = gr.DownloadButton(
                        "Download All (.zip)",
                        variant="secondary",
                        visible=False,
                    )
            gcode_status = gr.Markdown("")
            with gr.Row():
                gcode_text_source = gr.Dropdown(label="Preview G-Code", choices=[], value=None, allow_custom_value=False)
                refresh_gcode_text_button = gr.Button("Refresh G-Code Preview", variant="secondary", size="sm")
            gcode_text = gr.Code(label="Selected G-Code", language=None, lines=18, max_lines=18, interactive=False, elem_classes=["gcode-view"])

        with gr.Tab("Visualization") as viz_tab:
            gr.Markdown(
                "### Print Visualization\n"
                "Defaults to the parallel print of every generated shape, laid out with the "
                "spacing from the Nozzle Spacing accordion below. Pick a single shape (or "
                "upload a G-code file) to inspect one tool path — colors come from the Shape "
                "Settings table."
            )
            with gr.Row():
                gcode_source = gr.Radio(
                    choices=[GCODE_SOURCE_PARALLEL, GCODE_SOURCE_UPLOAD],
                    value=GCODE_SOURCE_PARALLEL,
                    label="What to visualize",
                )
                with gr.Column(scale=0, min_width=180):
                    viz_nozzle_speed = gr.Number(
                        label="Nozzle Speed (mm/s)",
                        value=10.0,
                        minimum=0.01,
                        step=0.5,
                        info="Used for the print time estimate.",
                    )
                with gr.Column(elem_id="gcode-upload-col"):
                    gcode_upload = gr.File(label="Upload G-Code", file_types=[".txt", ".gcode", ".nc"], interactive=True, height=110)

            with gr.Accordion("Nozzle Spacing", open=False, elem_classes=["settings-accordion"]):
                with gr.Group():
                    with gr.Row():
                        nozzle_grid_preset = gr.Dropdown(
                            label="Common Layout",
                            choices=NOZZLE_LAYOUT_PRESETS,
                            value="Custom",
                            allow_custom_value=False,
                        )
                        nozzle_grid_columns = gr.Number(label="Grid Columns", value=2, minimum=1, step=1)
                        nozzle_grid_rows = gr.Number(label="Grid Rows", value=2, minimum=1, step=1)
                        nozzle_grid_column_spacing = gr.Number(label="Column Gap (X, mm)", value=5.0, step=0.1)
                        nozzle_grid_row_spacing = gr.Number(label="Row Gap (Y, mm)", value=5.0, step=0.1)
                    with gr.Row():
                        auto_align_split_parts_button = gr.Button("Auto Align Split Parts", variant="secondary", size="sm")
                        nozzle_grid_use_individual_spacing = gr.Checkbox(label="Advanced Grid Spacing", value=False)
                    # Always visible: the "Advanced Grid Spacing" checkbox
                    # controls whether these per-connection gaps are USED,
                    # not whether the table shows. Toggling visibility from
                    # events proved racy (the table wouldn't appear on the
                    # first Auto Align click).
                    nozzle_grid_spacing_table = gr.Dataframe(
                        headers=ADVANCED_NOZZLE_SPACING_HEADERS,
                        value=[],
                        row_count=(0, "fixed"),
                        column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
                        interactive=True,
                        label="Advanced Grid Spacing (used when the checkbox is on)",
                        elem_id="nozzle-grid-spacing-table",
                    )
                nozzle_preview_button = gr.Button("Visualize Nozzle Spacing", variant="secondary", elem_id="visualize-nozzle-spacing-button")
                with gr.Row():
                    with gr.Column(scale=3, min_width=420):
                        nozzle_spacing_plot = gr.Plot(label="Nozzle Spacing")
                    with gr.Column(scale=1, min_width=260):
                        nozzle_spacing_status = gr.Markdown("Generate G-code, then visualize nozzle spacing.")

            # Parallel view (the default).
            with gr.Column(visible=True) as parallel_section:
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        pp_travel_opacity = gr.Slider(label="Travel opacity (0 = hidden)", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
                        parallel_line_button = gr.Button("Render Parallel Print - Line Plot", variant="primary")
                        parallel_render_button = gr.Button("Render Parallel Print - Tube Plot with Animation", variant="primary")
                        gr.Markdown("&#9888;&#65039; Building multiple tube plots can take a while for high-resolution models.", elem_id="parallel-render-warning")
                        parallel_anim_controls = gr.HTML(PARALLEL_CONTROLS_HTML, visible=False)
                        with gr.Row(visible=False) as pp_width_row:
                            pp_filament_width = gr.Slider(label="Filament width (mm)", minimum=0.1, maximum=3.0, value=0.8, step=0.05, min_width=150)
                            pp_travel_width = gr.Slider(label="Travel width (mm)", minimum=0.05, maximum=3.0, value=0.2, step=0.05, min_width=150)
                        parallel_status = gr.Markdown("")
                        with gr.Group(visible=False) as pp_export_group:
                            gr.Markdown("**Export animation (GIF)** - a server-side line animation of the parallel print.")
                            with gr.Row():
                                pp_gif_duration = gr.Slider(label="Duration (s)", minimum=2.0, maximum=20.0, value=6.0, step=1.0, min_width=150)
                                pp_gif_fps = gr.Slider(label="Frames per second", minimum=5, maximum=30, value=10, step=1, min_width=150)
                            with gr.Row():
                                pp_elev = gr.Slider(label="Elevation angle", minimum=0, maximum=90, value=22, step=1, min_width=150)
                                pp_azim = gr.Slider(label="Azimuth angle", minimum=-180, maximum=180, value=-60, step=5, min_width=150)
                            pp_gif_travel_opacity = gr.Slider(label="Travel opacity in GIF (0 = hidden)", minimum=0.0, maximum=1.0, value=0.15, step=0.05)
                            pp_export_button = gr.Button("Export Animation as GIF", variant="primary")
                            pp_gif_file = gr.File(label="Download GIF", interactive=False)
                    with gr.Column(scale=3, min_width=500):
                        parallel_plot = gr.Plot(label="Parallel Tool Paths", elem_id="parallel_plot")

            # Single tool path view (a generated shape or an uploaded file).
            with gr.Column(visible=False) as single_section:
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        with gr.Row():
                            travel_opacity_slider = gr.Slider(label="Travel (G0) opacity", minimum=0.0, maximum=1.0, value=0.2, step=0.05, min_width=150)
                            print_opacity_slider = gr.Slider(label="Print (G1) opacity", minimum=0.0, maximum=1.0, value=1.0, step=0.05, min_width=150)
                        render_line_button = gr.Button("Render Tool Path - Line Plot", variant="primary")
                        render_tube_button = gr.Button("Render Tool Path - Tube Plot with Animation", variant="primary")
                        gr.Markdown(
                            "&#9888;&#65039; For high-resolution models (small layer heights), the tube plot can take a while to build and render.",
                            elem_id="tube-render-warning",
                        )
                        anim_controls = gr.HTML(TOOLPATH_CONTROLS_HTML, visible=False)
                        with gr.Row(visible=False) as width_row:
                            travel_width_slider = gr.Slider(label="Travel width (mm)", minimum=0.05, maximum=1.2, value=0.2, step=0.05, min_width=150)
                            print_width_slider = gr.Slider(label="Filament width (mm)", minimum=0.1, maximum=1.2, value=0.8, step=0.05, min_width=150)
                        toolpath_status = gr.Markdown("")
                    with gr.Column(scale=3, min_width=500):
                        toolpath_plot = gr.Plot(label="Tool Path", elem_id="toolpath_plot")

            parsed_state = gr.State({})
            render_mode = gr.State("tube")
            parallel_mode = gr.State("tube")

        grid_spacing_refresh_inputs = [
            shape_records,
            nozzle_grid_columns,
            nozzle_grid_rows,
            nozzle_grid_column_spacing,
            nozzle_grid_row_spacing,
            nozzle_grid_spacing_table,
            nozzle_grid_use_individual_spacing,
        ]

        shape_sync_outputs = [shape_records, shape_settings, selected_shape, gcode_text_source, gcode_source, gcode_downloads, gcode_download_all]
        stl_upload.change(fn=sync_uploaded_shapes, inputs=[stl_upload, shape_records, shape_settings], outputs=shape_sync_outputs).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        sync_uploads_button.click(fn=sync_uploaded_shapes, inputs=[stl_upload, shape_records, shape_settings], outputs=shape_sync_outputs).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        load_samples_button.click(fn=load_sample_shapes, inputs=[stl_upload, shape_records, shape_settings, sample_set_selector], outputs=[stl_upload, *shape_sync_outputs]).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        color_apply.click(
            fn=apply_color_selection,
            inputs=[shape_records, shape_settings, color_sink],
            outputs=[shape_records, shape_settings],
            queue=False,
        )
        bulk_apply.click(
            fn=apply_bulk_bool_selection,
            inputs=[shape_records, shape_settings, bulk_sink],
            outputs=[shape_records, shape_settings],
            queue=False,
        )
        shape_settings.select(
            fn=delete_shape_from_settings,
            inputs=[shape_records, shape_settings, last_shape_delete_at],
            outputs=[stl_upload, *shape_sync_outputs, last_shape_delete_at],
            queue=False,
        ).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )

        preview_inputs = [shape_records, selected_shape, shape_settings, scale_mode]
        # .change is the ONLY event the Dataframe emits for cell edits (it has
        # no working .input) — and it also fires for the normalizer's own
        # write-back. The infinite/minute-long cascade is prevented inside
        # normalize_shape_dimensions_for_mode: it skips the table output
        # whenever no dimension actually changed.
        shape_settings.change(
            fn=normalize_shape_dimensions_for_mode,
            inputs=[shape_records, shape_settings, scale_mode],
            outputs=[shape_records, shape_settings],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        selected_shape.change(fn=show_selected_model, inputs=preview_inputs, outputs=[model_viewer, model_details])
        refresh_preview_button.click(fn=show_selected_model, inputs=preview_inputs, outputs=[model_viewer, model_details])
        # Sliced-layer preview. The slider uses .release only: the handler
        # writes the slider back (clamped max/value), and a .change wiring
        # would re-fire on that programmatic write.
        layer_preview_inputs = [shape_records, selected_shape, shape_settings, layer_preview_slider]
        layer_preview_outputs = [layer_preview_slider, layer_preview_plot]
        selected_shape.change(fn=update_layer_preview, inputs=layer_preview_inputs, outputs=layer_preview_outputs)
        refresh_preview_button.click(fn=update_layer_preview, inputs=layer_preview_inputs, outputs=layer_preview_outputs)
        layer_preview_slider.release(fn=update_layer_preview, inputs=layer_preview_inputs, outputs=layer_preview_outputs)
        scale_mode.change(
            fn=normalize_shape_dimensions_for_mode,
            inputs=[shape_records, shape_settings, scale_mode],
            outputs=[shape_records, shape_settings],
            queue=False,
        ).then(
            fn=show_selected_model,
            inputs=preview_inputs,
            outputs=[model_viewer, model_details],
        )
        reset_dimensions_button.click(
            fn=reset_shape_dimensions,
            inputs=[shape_records, shape_settings],
            outputs=[shape_records, shape_settings],
        ).then(
            fn=show_selected_model,
            inputs=preview_inputs,
            outputs=[model_viewer, model_details],
        )

        split_refresh_sources.click(fn=lambda records: _dropdown_update(records), inputs=[shape_records], outputs=[split_source], queue=False)
        # .input (user selections only): programmatic dropdown refills after a
        # split must not overwrite the split result message.
        split_source.input(
            fn=describe_split_source,
            inputs=[shape_records, split_source],
            outputs=[split_status],
            queue=False,
        )
        split_button.click(
            fn=split_selected_shape_for_grid,
            inputs=[
                shape_records,
                split_source,
                shape_settings,
                split_columns,
                split_rows,
                split_overlapping_layers,
                split_start_nozzle,
                split_start_valve,
                fil_width,
                layer_height,
                scale_mode,
                split_undo,
            ],
            outputs=[
                shape_records,
                shape_settings,
                selected_shape,
                gcode_downloads,
                gcode_text_source,
                gcode_source,
                split_source,
                split_status,
                gcode_download_all,
                split_undo,
            ],
        ).then(
            fn=generate_dynamic_reference_stack,
            inputs=[shape_records, fil_width],
            outputs=[ref_layers],
        ).then(
            fn=update_layer_preview,
            inputs=layer_preview_inputs,
            outputs=layer_preview_outputs,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        split_undo_button.click(
            fn=undo_last_split,
            inputs=[split_undo],
            outputs=[
                shape_records,
                shape_settings,
                selected_shape,
                gcode_downloads,
                gcode_text_source,
                gcode_source,
                split_source,
                split_status,
                gcode_download_all,
                split_undo,
            ],
        ).then(
            fn=generate_dynamic_reference_stack,
            inputs=[shape_records, fil_width],
            outputs=[ref_layers],
        ).then(
            fn=update_layer_preview,
            inputs=layer_preview_inputs,
            outputs=layer_preview_outputs,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        assign_valves_button.click(
            fn=assign_unique_valves,
            inputs=[shape_records, shape_settings],
            outputs=[shape_records, shape_settings],
            queue=False,
        )
        export_settings_button.click(
            fn=export_project_settings,
            inputs=[
                shape_records,
                shape_settings,
                layer_height,
                fil_width,
                scale_mode,
                gcode_raster_pattern,
                gcode_pressure_ramp_enabled,
                gcode_sweep_buffer,
                gcode_lead_in_length,
                gcode_lead_in_clearance,
                gcode_lead_in_lines,
                gcode_lead_in_direction,
                gcode_lead_in_orientation,
                viz_nozzle_speed,
            ],
            outputs=[settings_export_file, settings_status],
            queue=False,
        )
        settings_import_upload.change(
            fn=import_project_settings,
            inputs=[settings_import_upload, shape_records, shape_settings],
            outputs=[
                shape_records,
                shape_settings,
                settings_status,
                gcode_raster_pattern,
                gcode_pressure_ramp_enabled,
                gcode_sweep_buffer,
                gcode_lead_in_length,
                gcode_lead_in_clearance,
                gcode_lead_in_lines,
                gcode_lead_in_direction,
                gcode_lead_in_orientation,
                layer_height,
                fil_width,
                scale_mode,
                viz_nozzle_speed,
            ],
        )

        # Stale-G-code banner: re-checked on every table change and every
        # generation option change; generation itself re-stamps the
        # snapshots, so its chain clears the banner.
        stale_inputs = [
            shape_records,
            shape_settings,
            gcode_raster_pattern,
            gcode_pressure_ramp_enabled,
            gcode_lead_in_length,
            gcode_lead_in_clearance,
            gcode_lead_in_lines,
            gcode_lead_in_direction,
            layer_height,
            fil_width,
            scale_mode,
            gcode_sweep_buffer,
            gcode_lead_in_orientation,
        ]
        shape_settings.change(
            fn=check_gcode_staleness,
            inputs=stale_inputs,
            outputs=[gcode_stale_banner],
            queue=False,
        )
        for stale_control in (
            gcode_raster_pattern,
            gcode_pressure_ramp_enabled,
            gcode_lead_in_length,
            gcode_lead_in_clearance,
            gcode_lead_in_lines,
            gcode_lead_in_direction,
            layer_height,
            fil_width,
            scale_mode,
            gcode_sweep_buffer,
            gcode_lead_in_orientation,
        ):
            stale_control.change(
                fn=check_gcode_staleness,
                inputs=stale_inputs,
                outputs=[gcode_stale_banner],
                queue=False,
            )

        # Defined before the generate chain so it can auto-render the
        # parallel view with fresh files (the same lists drive the
        # Visualization tab wiring further down).
        parallel_render_inputs = [
            shape_records,
            shape_settings,
            pp_travel_opacity,
            pp_filament_width,
            pp_travel_width,
            nozzle_grid_columns,
            nozzle_grid_rows,
            nozzle_grid_column_spacing,
            nozzle_grid_row_spacing,
            nozzle_grid_use_individual_spacing,
            nozzle_grid_spacing_table,
            viz_nozzle_speed,
        ]
        parallel_outputs = [parallel_plot, parallel_status, parallel_mode, parallel_anim_controls, pp_width_row, pp_export_group]

        gcode_button.click(
            fn=generate_dynamic_gcode,
            inputs=[
                shape_records,
                shape_settings,
                gcode_raster_pattern,
                gcode_pressure_ramp_enabled,
                gcode_lead_in_length,
                gcode_lead_in_clearance,
                gcode_lead_in_lines,
                gcode_lead_in_direction,
                layer_height,
                fil_width,
                scale_mode,
                viz_nozzle_speed,
                gcode_sweep_buffer,
                gcode_lead_in_orientation,
            ],
            outputs=[shape_records, ref_layers, gcode_downloads, gcode_status, gcode_text_source, gcode_source, gcode_download_all],
        ).then(
            fn=load_selected_gcode_text,
            inputs=[shape_records, gcode_text_source],
            outputs=[gcode_text],
        ).then(
            # Generation slices the shapes, so refresh everything that
            # depends on the sliced stacks (the Slice Shapes button is gone).
            fn=update_layer_preview,
            inputs=layer_preview_inputs,
            outputs=layer_preview_outputs,
        ).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        ).then(
            fn=check_gcode_staleness,
            inputs=stale_inputs,
            outputs=[gcode_stale_banner],
            queue=False,
        ).then(
            # Fresh files: refresh the parallel view so the Visualization
            # tab always shows the current print.
            fn=render_dynamic_parallel_lines,
            inputs=parallel_render_inputs,
            outputs=parallel_outputs,
        )
        gcode_text_source.change(fn=load_selected_gcode_text, inputs=[shape_records, gcode_text_source], outputs=[gcode_text])
        refresh_gcode_text_button.click(fn=load_selected_gcode_text, inputs=[shape_records, gcode_text_source], outputs=[gcode_text])
        auto_align_split_parts_button.click(
            fn=auto_align_split_parts,
            inputs=[
                shape_records,
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
            ],
            outputs=[
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
                nozzle_grid_use_individual_spacing,
                nozzle_grid_spacing_table,
                nozzle_spacing_status,
            ],
            queue=False,
        )
        nozzle_grid_preset.change(
            fn=update_nozzle_grid_preset,
            inputs=[nozzle_grid_preset, shape_records, nozzle_grid_columns, nozzle_grid_rows],
            outputs=[nozzle_grid_columns, nozzle_grid_rows],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        for grid_spacing_control in (
            nozzle_grid_columns,
            nozzle_grid_rows,
            nozzle_grid_column_spacing,
            nozzle_grid_row_spacing,
            nozzle_grid_use_individual_spacing,
        ):
            # .input (user edits only), NOT .change: Auto Align sets these
            # controls programmatically and .change listeners would fire and
            # rebuild the spacing table, clobbering the aligned gaps it just
            # wrote. Flows that set them programmatically (preset dropdown,
            # Auto Align) update the table explicitly themselves.
            grid_spacing_control.input(
                fn=_grid_spacing_table_update,
                inputs=grid_spacing_refresh_inputs,
                outputs=[nozzle_grid_spacing_table],
                queue=False,
            )
        nozzle_preview_button.click(
            fn=render_dynamic_nozzle_spacing,
            inputs=[
                shape_records,
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
                nozzle_grid_use_individual_spacing,
                nozzle_grid_spacing_table,
            ],
            outputs=[nozzle_spacing_plot, nozzle_spacing_status],
        )

        gcode_source.change(
            fn=None,
            inputs=[gcode_source],
            outputs=[],
            js="""(src) => {
                const col = document.getElementById('gcode-upload-col');
                if (col) col.style.display = (src === '""" + GCODE_SOURCE_UPLOAD + """') ? 'flex' : 'none';
                return [];
            }""",
        )
        # The source radio also decides which view is shown: the parallel
        # print (default) or the single-toolpath view for one shape / upload.
        gcode_source.change(
            fn=lambda source: (
                gr.update(visible=source == GCODE_SOURCE_PARALLEL),
                gr.update(visible=source != GCODE_SOURCE_PARALLEL),
            ),
            inputs=[gcode_source],
            outputs=[parallel_section, single_section],
            queue=False,
        )
        render_inputs = [
            gcode_source,
            gcode_upload,
            shape_records,
            travel_opacity_slider,
            print_opacity_slider,
            print_width_slider,
            travel_width_slider,
            viz_nozzle_speed,
        ]
        render_line_button.click(fn=render_dynamic_toolpath_lines, inputs=render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state, render_mode, anim_controls, width_row])
        render_tube_button.click(fn=render_dynamic_toolpath_tubes, inputs=render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state, render_mode, anim_controls, width_row])
        travel_width_slider.release(fn=rerender_dynamic_toolpath_current_mode, inputs=[render_mode] + render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state])
        print_width_slider.release(fn=rerender_dynamic_toolpath_current_mode, inputs=[render_mode] + render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state])

        def sync_width_sliders(value: float):
            # Visualization filament widths track the slicer's Filament/Line
            # Width (what the G-code is actually generated with); travel
            # lines render at a quarter of it.
            width = float(value or 0.8)
            travel = width / 4
            return (
                gr.update(value=width, minimum=min(0.1, width), maximum=width * 1.5),
                gr.update(value=travel, minimum=min(0.05, travel), maximum=width * 1.5),
                gr.update(value=width, minimum=min(0.1, width), maximum=max(3.0, width * 1.5)),
                gr.update(value=travel, minimum=min(0.05, travel), maximum=max(3.0, width * 1.5)),
            )

        fil_width.change(
            fn=sync_width_sliders,
            inputs=[fil_width],
            outputs=[print_width_slider, travel_width_slider, pp_filament_width, pp_travel_width],
            queue=False,
        )

        parallel_line_button.click(fn=render_dynamic_parallel_lines, inputs=parallel_render_inputs, outputs=parallel_outputs)
        parallel_render_button.click(fn=render_dynamic_parallel_tubes, inputs=parallel_render_inputs, outputs=parallel_outputs)
        viz_tab.select(
            fn=refresh_parallel_on_tab_select,
            inputs=[parallel_mode] + parallel_render_inputs,
            outputs=[parallel_plot, parallel_status],
        )
        pp_filament_width.release(fn=rerender_dynamic_parallel_current_mode, inputs=[parallel_mode] + parallel_render_inputs, outputs=[parallel_plot, parallel_status])
        pp_travel_width.release(fn=rerender_dynamic_parallel_current_mode, inputs=[parallel_mode] + parallel_render_inputs, outputs=[parallel_plot, parallel_status])
        pp_export_button.click(
            fn=export_dynamic_parallel_gif,
            inputs=[
                shape_records,
                shape_settings,
                pp_gif_travel_opacity,
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
                nozzle_grid_use_individual_spacing,
                nozzle_grid_spacing_table,
                pp_gif_duration,
                pp_gif_fps,
                pp_elev,
                pp_azim,
            ],
            outputs=[pp_gif_file],
        )

    return demo


demo = build_dynamic_demo()


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
