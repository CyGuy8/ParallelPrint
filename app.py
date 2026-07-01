from __future__ import annotations

import tempfile
import math
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
from PIL import Image, ImageDraw, ImageFont
import trimesh

from gcode_viewer import (
    build_nozzle_spacing_figure,
    build_parallel_figure,
    build_parallel_gif,
    build_toolpath_figure,
    parse_gcode_path,
)
from stl_slicer import (
    SliceStack,
    load_mesh,
    scale_factors_for_target_extents,
    scale_mesh,
    slice_stl_to_tiffs,
)
from tiff_to_gcode import (
    CONTOUR_MODE_ROW_ENVELOPE,
    RASTER_PATTERN_CHOICES,
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
    generate_snake_path_gcode,
)


ViewerState = dict[str, Any]
SAMPLE_STL_FILENAMES = ("Hollow_Pyramid.stl", "Rounded_Cube_Through_Holes.stl", "halfsphere.stl")
SAMPLE_STL_DIR = Path(__file__).resolve().parent / "sample_stls"
DEFAULT_TARGET_EXTENTS = (20.0, 20.0, 20.0)
DELETE_SHAPE_COOLDOWN_SECONDS = 1.0
UNIFORM_TARGET_AXES = ("X", "Y", "Z")
SCALE_MODE_TARGET_DIMENSIONS = "Independent X/Y/Z"
SCALE_MODE_UNIFORM_FACTOR = "Keep Proportions"
TARGET_DIMENSION_KEYS = ("target_x", "target_y", "target_z")
FRONT_CAMERA = (90, 80, None)
NOZZLE_LAYOUT_GRID = "Grid Layout"
NOZZLE_LAYOUT_PAIR_TABLE = "Custom Spacing"
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
AUTO_ALIGN_X_RASTER_OFFSETS = (-3.2, -0.8)
AUTO_ALIGN_Y_RASTER_OFFSETS = (-0.8, -3.2)
APP_CSS = """
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
    function start() {
        enableUndoButtons();
        document.addEventListener('focusin', suppressDeleteCellEditor);
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


def _read_slice_preview(path: str) -> Image.Image:
    with Image.open(path) as image:
        preview = image.copy()

    # Upscale low-resolution TIFF previews so they fill the viewer area better.
    min_display_side = 480
    width, height = preview.size
    max_dim = max(width, height)
    if max_dim > 0 and max_dim < min_display_side:
        scale = min_display_side / max_dim
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        preview = preview.resize(new_size, resample=Image.Resampling.NEAREST)

    return preview


def _empty_state() -> ViewerState:
    return {
        "tiff_paths": [],
        "z_values": [],
        "pixel_size": 0.0,
        "x_min": 0.0,
        "y_min": 0.0,
        "image_width": 0,
        "image_height": 0,
    }


def _reset_slider() -> dict[str, Any]:
    return gr.update(minimum=0, maximum=0, value=0, step=1, interactive=False)


def _stack_to_state(stack: SliceStack) -> ViewerState:
    (x_min, y_min, _z_min), (_x_max, _y_max, _z_max) = stack.bounds
    return {
        "tiff_paths": [str(path) for path in stack.tiff_paths],
        "z_values": stack.z_values,
        "pixel_size": stack.pixel_size,
        "x_min": x_min,
        "y_min": y_min,
        "image_width": stack.image_size[0],
        "image_height": stack.image_size[1],
    }


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


def _slice_label(state: ViewerState, index: int) -> str:
    path = Path(state["tiff_paths"][index]).name
    z_value = state["z_values"][index]
    total = len(state["tiff_paths"])
    return f"Slice {index + 1} / {total} | z = {z_value:.4f} | {path}"


def _annotate_preview(
    image: Image.Image,
    pixel_size: float,
    x_min: float,
    y_min: float,
    orig_width: int,
    orig_height: int,
) -> Image.Image:
    """Draw a blue origin crosshair with axis labels and a scale bar."""
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)

    preview_w, preview_h = rgb.size
    scale_x = preview_w / orig_width if orig_width else 1.0
    scale_y = preview_h / orig_height if orig_height else 1.0

    BLUE = (50, 120, 255)

    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    try:
        small_font = ImageFont.load_default(size=12)
    except TypeError:
        small_font = font

    # --- Origin crosshair & axis indicators ---
    origin_px = (0.0 - x_min) / pixel_size
    origin_py_from_bottom = (0.0 - y_min) / pixel_size
    origin_img_y = orig_height - 1 - origin_py_from_bottom

    ox = int(round(origin_px * scale_x))
    oy = int(round(origin_img_y * scale_y))

    arm = 20
    margin_edge = 8  # inset from image border for off-screen indicators
    on_screen = 0 <= ox < preview_w and 0 <= oy < preview_h

    if on_screen:
        # +X axis (rightward)
        x_start = max(0, ox)
        x_end = min(preview_w - 1, ox + arm)
        if x_end > x_start:
            draw.line([(x_start, oy), (x_end, oy)], fill=BLUE, width=2)
            draw.polygon(
                [(x_end, oy), (x_end - 5, oy - 4), (x_end - 5, oy + 4)],
                fill=BLUE,
            )
            if x_end + 4 < preview_w:
                draw.text((x_end + 4, oy - 7), "X", fill=BLUE, font=small_font)

        # +Y axis (upward in world = upward in image)
        y_end = max(0, oy - arm)
        y_start = min(preview_h - 1, oy)
        if y_start > y_end:
            draw.line([(ox, y_start), (ox, y_end)], fill=BLUE, width=2)
            draw.polygon(
                [(ox, y_end), (ox - 4, y_end + 5), (ox + 4, y_end + 5)],
                fill=BLUE,
            )
            if y_end - 16 >= 0:
                draw.text((ox + 5, y_end - 16), "Y", fill=BLUE, font=small_font)

        # -X stub (leftward from origin)
        stub = min(8, max(0, ox))
        if stub > 0:
            draw.line([(ox - stub, oy), (ox, oy)], fill=BLUE, width=2)

        # -Y stub (downward from origin in image)
        stub_y = min(8, max(0, preview_h - 1 - oy))
        if stub_y > 0:
            draw.line([(ox, oy), (ox, oy + stub_y)], fill=BLUE, width=2)

        # Origin label
        lx = ox + arm + 4 if ox + arm + 40 < preview_w else ox - 45
        ly = oy + 6
        if 0 <= ly < preview_h:
            draw.text((max(0, lx), ly), "(0, 0)", fill=BLUE, font=small_font)

    else:
        # Origin is off-screen — draw edge indicator(s) pointing toward it.
        arrow_len = 14
        arrow_half = 5

        # Compute direction label text showing approximate origin coordinates
        origin_x_mm = x_min
        origin_y_mm = y_min
        coord_text = f"Origin ({-origin_x_mm:+.1f}, {-origin_y_mm:+.1f})"

        if ox < 0:
            # Origin is to the LEFT — draw left-pointing arrow on left edge
            ay = max(margin_edge + arrow_half, min(preview_h - margin_edge - arrow_half, oy))
            draw.polygon(
                [(margin_edge, ay), (margin_edge + arrow_len, ay - arrow_half), (margin_edge + arrow_len, ay + arrow_half)],
                fill=BLUE,
            )
            draw.text((margin_edge + arrow_len + 4, ay - 7), coord_text, fill=BLUE, font=small_font)
        elif ox >= preview_w:
            # Origin is to the RIGHT
            ay = max(margin_edge + arrow_half, min(preview_h - margin_edge - arrow_half, oy))
            rx = preview_w - margin_edge
            draw.polygon(
                [(rx, ay), (rx - arrow_len, ay - arrow_half), (rx - arrow_len, ay + arrow_half)],
                fill=BLUE,
            )
            tw = len(coord_text) * 7
            draw.text((max(0, rx - arrow_len - tw - 4), ay - 7), coord_text, fill=BLUE, font=small_font)

        if oy < 0:
            # Origin is ABOVE — draw upward-pointing arrow on top edge
            ax = max(margin_edge + arrow_half, min(preview_w - margin_edge - arrow_half, ox))
            draw.polygon(
                [(ax, margin_edge), (ax - arrow_half, margin_edge + arrow_len), (ax + arrow_half, margin_edge + arrow_len)],
                fill=BLUE,
            )
        elif oy >= preview_h:
            # Origin is BELOW — draw downward-pointing arrow on bottom edge
            ax = max(margin_edge + arrow_half, min(preview_w - margin_edge - arrow_half, ox))
            by = preview_h - margin_edge
            draw.polygon(
                [(ax, by), (ax - arrow_half, by - arrow_len), (ax + arrow_half, by - arrow_len)],
                fill=BLUE,
            )
            # If we didn't already draw a left/right label, label here
            if 0 <= ox < preview_w:
                draw.text((ax + arrow_half + 4, by - arrow_len - 2), coord_text, fill=BLUE, font=small_font)

    # --- Scale bar (bottom-left) ---
    image_width_mm = orig_width * pixel_size
    target_bar_mm = image_width_mm * 0.2
    nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    bar_mm = min(nice, key=lambda v: abs(v - target_bar_mm))

    bar_px = (bar_mm / pixel_size) * scale_x
    margin = 12
    bar_y = preview_h - margin
    bar_x0 = margin
    bar_x1 = bar_x0 + bar_px
    cap = 5

    draw.line([(int(bar_x0), int(bar_y)), (int(bar_x1), int(bar_y))], fill=BLUE, width=3)
    draw.line([(int(bar_x0), int(bar_y - cap)), (int(bar_x0), int(bar_y + cap))], fill=BLUE, width=2)
    draw.line([(int(bar_x1), int(bar_y - cap)), (int(bar_x1), int(bar_y + cap))], fill=BLUE, width=2)

    bar_label = f"{bar_mm:g} mm"
    draw.text((int(bar_x0), int(bar_y - 20)), bar_label, fill=BLUE, font=font)

    return rgb


def _render_selected_slice(state: ViewerState, index: int) -> tuple[str, Image.Image | None]:
    tiff_paths = state.get("tiff_paths", [])
    if not tiff_paths:
        return "No slice stack loaded yet.", None

    bounded_index = max(0, min(int(index), len(tiff_paths) - 1))
    selected_path = tiff_paths[bounded_index]
    preview = _read_slice_preview(selected_path)

    pixel_size = state.get("pixel_size", 0.0)
    if pixel_size and pixel_size > 0:
        preview = _annotate_preview(
            preview,
            pixel_size=pixel_size,
            x_min=state.get("x_min", 0.0),
            y_min=state.get("y_min", 0.0),
            orig_width=state.get("image_width", 0) or preview.size[0],
            orig_height=state.get("image_height", 0) or preview.size[1],
        )

    return (
        _slice_label(state, bounded_index),
        preview,
    )


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


def jump_to_slice(state: ViewerState, index: float) -> tuple[str, Image.Image | None]:
    return _render_selected_slice(state, int(index))


GCODE_SOURCE_UPLOAD = "Upload G-Code file"


PARALLEL_COLOR_CHOICES = [
    ("Orange", "#ff7f0e"), ("Blue", "#1f77b4"), ("Green", "#2ca02c"),
    ("Red", "#d62728"), ("Purple", "#9467bd"), ("Pink", "#e377c2"),
    ("Teal", "#17becf"), ("Black", "#000000"),
]
DEFAULT_PARALLEL_COLORS = ("#ff7f0e", "#1f77b4", "#2ca02c")


def _resolve_nozzle_layout(
    parts: list[dict],
    same_spacing: bool | None,
    part_gap_12_x: float | None,
    part_gap_12_y: float | None,
    part_gap_23_x: float | None,
    part_gap_23_y: float | None,
    *extra_pair_gaps: float | None,
) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    offsets: dict[int, tuple[float, float]] = {}
    spacings: list[dict] = []
    if not parts:
        return offsets, spacings

    grouped: dict[int, list[dict]] = {}
    for part in parts:
        grouped.setdefault(_record_nozzle_number(part, int(part.get("idx", 1) or 1)), []).append(part)
    ordered_nozzles = sorted(grouped)
    offsets[ordered_nozzles[0]] = (0.0, 0.0)

    def nozzle_bounds(nozzle: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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

    first_spacing = (float(part_gap_12_x or 0.0), float(part_gap_12_y or 0.0))
    raw_pairs: list[tuple[float, float]] = [
        first_spacing,
        (float(part_gap_23_x or 0.0), float(part_gap_23_y or 0.0)),
    ]
    for index in range(0, len(extra_pair_gaps), 2):
        raw_pairs.append((
            float(extra_pair_gaps[index] or 0.0),
            float(extra_pair_gaps[index + 1] or 0.0) if index + 1 < len(extra_pair_gaps) else 0.0,
        ))
    max_pair_count = max(0, len(ordered_nozzles) - 1)
    while len(raw_pairs) < max_pair_count:
        raw_pairs.append(first_spacing)
    if same_spacing:
        raw_pairs = [first_spacing for _ in raw_pairs]

    pair_spacing = {
        (ordered_nozzles[index], ordered_nozzles[index + 1]): raw_pairs[index]
        for index in range(min(len(raw_pairs), max_pair_count))
    }

    def spacing_between(prev_idx: int, cur_idx: int) -> tuple[float, float]:
        if (prev_idx, cur_idx) in pair_spacing:
            return pair_spacing[(prev_idx, cur_idx)]
        try:
            start_pos = ordered_nozzles.index(prev_idx)
            end_pos = ordered_nozzles.index(cur_idx)
        except ValueError:
            return 0.0, 0.0
        if end_pos <= start_pos:
            return 0.0, 0.0
        total_x = 0.0
        total_y = 0.0
        for pair_pos in range(start_pos, end_pos):
            pair = (ordered_nozzles[pair_pos], ordered_nozzles[pair_pos + 1])
            step_x, step_y = pair_spacing.get(pair, first_spacing)
            total_x += step_x
            total_y += step_y
        return total_x, total_y

    for prev_idx, cur_idx in zip(ordered_nozzles, ordered_nozzles[1:]):
        gap_x, y_step = spacing_between(prev_idx, cur_idx)
        prev_offset_x, prev_offset_y = offsets[prev_idx]
        (_, _, _), (prev_xmax, _prev_ymax, _) = nozzle_bounds(prev_idx)
        (cur_xmin, _cur_ymin, _), (_, _, _) = nozzle_bounds(cur_idx)
        dx = (prev_xmax + prev_offset_x + gap_x) - cur_xmin
        dy = prev_offset_y + y_step
        offsets[cur_idx] = (dx, dy)
        spacings.append({
            "from": prev_idx,
            "to": cur_idx,
            "dx": dx - prev_offset_x,
            "dy": y_step,
        })

    return offsets, spacings


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


def _resolve_layout_from_spacing_controls(
    parts: list[dict],
    layout_mode: str | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool | None,
    grid_spacing_table: Any,
    use_individual_spacing: bool,
    spacing_table: Any,
) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    if layout_mode != NOZZLE_LAYOUT_PAIR_TABLE:
        return _resolve_nozzle_grid_layout(
            parts,
            columns,
            rows,
            column_spacing,
            row_spacing,
            use_grid_individual_spacing,
            grid_spacing_table,
        )
    gap12x, gap12y, gap23x, gap23y, extra = _spacing_args_from_table(spacing_table, use_individual_spacing)
    return _resolve_nozzle_layout(
        parts,
        _same_spacing_from_individual(use_individual_spacing),
        gap12x,
        gap12y,
        gap23x,
        gap23y,
        *extra,
    )


def _same_spacing_from_individual(use_individual_spacing: bool | None) -> bool:
    return not bool(use_individual_spacing)


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


def shift_slice(state: ViewerState, index: float, delta: int) -> tuple[int, str, Image.Image | None]:
    tiff_paths = state.get("tiff_paths", [])
    if not tiff_paths:
        return 0, "No slice stack loaded yet.", None

    new_index = max(0, min(int(index) + delta, len(tiff_paths) - 1))
    label, preview = _render_selected_slice(state, new_index)
    return new_index, label, preview


def generate_reference_stack(
    *states: ViewerState,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Combine all available TIFF stacks into a single reference stack.

    For each pixel in each layer the result is black (0) when *any* source
    stack has a black pixel at that position, and white (255) only when *all*
    sources are white.  Images of different sizes are centred on a canvas
    sized to the largest dimensions.
    """
    active_states = [s for s in states if s.get("tiff_paths")]

    if not active_states:
        return (
            _empty_state(),
            _reset_slider(),
            "No TIFF stacks available. Generate TIFF stacks first.",
            None,
        )

    max_layers = max(len(s["tiff_paths"]) for s in active_states)

    # Determine the largest image dimensions across all stacks.
    max_width = 0
    max_height = 0
    source_sizes: list[tuple[int, int]] = []
    for state in active_states:
        w = state.get("image_width", 0)
        h = state.get("image_height", 0)
        if not w or not h:
            with Image.open(state["tiff_paths"][0]) as img:
                w, h = img.size
        source_sizes.append((w, h))
        max_width = max(max_width, w)
        max_height = max(max_height, h)

    # Compute annotation metadata from the first active state, accounting for
    # the centering offset applied to its image on the larger canvas.
    first = active_states[0]
    first_w, first_h = source_sizes[0]
    ref_pixel_size = first.get("pixel_size", 0.0)
    x_off_first = (max_width - first_w) // 2
    y_off_first = (max_height - first_h) // 2
    ref_x_min = first.get("x_min", 0.0) - x_off_first * ref_pixel_size
    ref_y_min = first.get("y_min", 0.0) - y_off_first * ref_pixel_size

    output_dir = Path(tempfile.mkdtemp(prefix="reference_stack_"))
    slices_dir = output_dir / "tiff_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    tiff_paths: list[Path] = []
    z_values: list[float] = []

    for layer_idx in range(max_layers):
        progress(
            layer_idx / max_layers,
            desc=f"Compositing reference layer {layer_idx + 1}/{max_layers}",
        )

        # Start with an all-white canvas.
        ref_array = np.full((max_height, max_width), 255, dtype=np.uint8)

        for state in active_states:
            paths = state["tiff_paths"]
            if layer_idx >= len(paths):
                continue  # Stack exhausted – contributes white.

            with Image.open(paths[layer_idx]) as img:
                arr = np.asarray(img)

            h, w = arr.shape[:2]
            y_off = (max_height - h) // 2
            x_off = (max_width - w) // 2

            # Black (0) wins: pixel-wise minimum keeps any black pixel.
            region = ref_array[y_off : y_off + h, x_off : x_off + w]
            ref_array[y_off : y_off + h, x_off : x_off + w] = np.minimum(region, arr)

        ref_image = Image.fromarray(ref_array, mode="L")
        tiff_path = slices_dir / f"ref_slice_{layer_idx:04d}.tif"
        ref_image.save(tiff_path, compression="tiff_deflate")
        tiff_paths.append(tiff_path)

        # Use z-value from the first active state that covers this layer.
        z_val = 0.0
        for state in active_states:
            if layer_idx < len(state["z_values"]):
                z_val = state["z_values"][layer_idx]
                break
        z_values.append(z_val)

    ref_state: ViewerState = {
        "tiff_paths": [str(p) for p in tiff_paths],
        "z_values": z_values,
        "pixel_size": ref_pixel_size,
        "x_min": ref_x_min,
        "y_min": ref_y_min,
        "image_width": max_width,
        "image_height": max_height,
    }

    label, preview = _render_selected_slice(ref_state, 0)
    slider = gr.update(
        minimum=0,
        maximum=max(0, len(tiff_paths) - 1),
        value=0,
        step=1,
        interactive=len(tiff_paths) > 1,
    )

    return ref_state, slider, label, preview


def _zip_tiff_paths(tiff_paths: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for tiff_path in tiff_paths:
            archive.write(tiff_path, arcname=tiff_path.name)


def _partition_length(length: int, count: int) -> list[tuple[int, int]]:
    base = length // count
    remainder = length % count
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        end = start + size
        spans.append((start, end))
        start = end
    return spans


def _padded_grid_axis(length: int, count: int) -> dict[str, Any]:
    cell_size = max(1, math.ceil(length / count))
    padded_length = cell_size * count
    total_pad = padded_length - length
    leading_pad = total_pad // 2
    trailing_pad = total_pad - leading_pad
    spans = [
        (index * cell_size, (index + 1) * cell_size)
        for index in range(count)
    ]
    return {
        "cell_size": cell_size,
        "padded_length": padded_length,
        "leading_pad": leading_pad,
        "trailing_pad": trailing_pad,
        "spans": spans,
    }


def _layer_split_spans(length: int, count: int, layer_index: int, overlap_pixels: int) -> list[tuple[int, int]]:
    base_spans = _partition_length(length, count)
    if overlap_pixels <= 0 or count <= 1:
        return base_spans

    boundaries = [base_spans[0][0], *[end for _start, end in base_spans]]
    adjusted = list(boundaries)
    for boundary_index in range(1, len(boundaries) - 1):
        direction = 1 if (layer_index + boundary_index) % 2 == 1 else -1
        lower = adjusted[boundary_index - 1] + 1
        upper = boundaries[boundary_index + 1] - 1
        adjusted[boundary_index] = max(lower, min(upper, boundaries[boundary_index] + direction * overlap_pixels))
    return [(adjusted[index], adjusted[index + 1]) for index in range(count)]


def split_tiff_stack_grid(
    state: ViewerState,
    base_name: str = "split_shape",
    columns: float = 2,
    rows: float = 1,
    overlapping_layers: bool | None = False,
    progress: gr.Progress = gr.Progress(),
) -> list[dict[str, Any]]:
    tiff_paths = [Path(path) for path in state.get("tiff_paths", [])]
    if not tiff_paths:
        raise ValueError("Generate a TIFF stack for the selected shape before splitting it.")

    with Image.open(tiff_paths[0]) as first_image:
        width, height = first_image.size
    column_count = max(1, _coerce_int(columns, 2))
    row_count = max(1, _coerce_int(rows, 1))
    if column_count > width:
        raise ValueError(f"Cannot split {width}-pixel-wide TIFF slices into {column_count} columns.")
    if row_count > height:
        raise ValueError(f"Cannot split {height}-pixel-tall TIFF slices into {row_count} rows.")

    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base_name).strip("_") or "split_shape"
    output_dir = Path(tempfile.mkdtemp(prefix=f"{safe_name}_split_"))
    x_axis = _padded_grid_axis(width, column_count)
    y_axis = _padded_grid_axis(height, row_count)
    x_spans = x_axis["spans"]
    y_spans = y_axis["spans"]
    overlap_pixels = 1 if overlapping_layers else 0
    overlap_x = overlap_pixels if column_count > 1 else 0
    overlap_y = overlap_pixels if row_count > 1 else 0
    padded_width = int(x_axis["padded_length"])
    padded_height = int(y_axis["padded_length"])
    working_width = padded_width + (2 * overlap_x)
    working_height = padded_height + (2 * overlap_y)
    source_x = int(x_axis["leading_pad"]) + overlap_x
    source_y = int(y_axis["leading_pad"]) + overlap_y
    working_x_min = base_x_min = float(state.get("x_min", 0.0) or 0.0)
    working_y_min = base_y_min = float(state.get("y_min", 0.0) or 0.0)
    pixel_size = float(state.get("pixel_size", 0.0) or 0.0)
    working_x_min = base_x_min - (source_x * pixel_size)
    working_y_min = base_y_min - ((int(y_axis["trailing_pad"]) + overlap_y) * pixel_size)
    pieces: list[dict[str, Any]] = []
    for row_index, (y_start, y_end) in enumerate(y_spans, start=1):
        for col_index, (x_start, x_end) in enumerate(x_spans, start=1):
            piece_dir = output_dir / f"r{row_index:02d}_c{col_index:02d}_tiff_slices"
            piece_dir.mkdir(parents=True, exist_ok=True)
            pieces.append({
                "row": row_index,
                "col": col_index,
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end,
                "tiff_dir": piece_dir,
                "tiff_paths": [],
            })

    for index, source_path in enumerate(tiff_paths):
        progress(index / max(len(tiff_paths), 1), desc=f"Splitting layer {index + 1}/{len(tiff_paths)}")
        with Image.open(source_path) as image:
            layer = image.convert("L")
            if layer.size != (width, height):
                raise ValueError("All TIFF slices must have the same dimensions to split the stack.")

            padded_layer = Image.new("L", (working_width, working_height), 255)
            padded_layer.paste(layer, (source_x, source_y))
            layer_x_spans = [
                (x_start + overlap_x, x_end + overlap_x)
                for x_start, x_end in _layer_split_spans(
                    padded_width,
                    column_count,
                    index,
                    overlap_x,
                )
            ]
            layer_y_spans = [
                (y_start + overlap_y, y_end + overlap_y)
                for y_start, y_end in _layer_split_spans(
                    padded_height,
                    row_count,
                    index,
                    overlap_y,
                )
            ]
            for piece in pieces:
                canvas_x_start = piece["x_start"]
                canvas_x_end = piece["x_end"]
                canvas_y_start = piece["y_start"]
                canvas_y_end = piece["y_end"]
                if overlapping_layers:
                    canvas_x_start -= overlap_x
                    canvas_x_end += overlap_x
                    canvas_y_start -= overlap_y
                    canvas_y_end += overlap_y
                canvas_x_start += overlap_x
                canvas_x_end += overlap_x
                canvas_y_start += overlap_y
                canvas_y_end += overlap_y
                x_start, x_end = layer_x_spans[piece["col"] - 1]
                y_start, y_end = layer_y_spans[piece["row"] - 1]
                if overlapping_layers:
                    piece_image = Image.new("L", (canvas_x_end - canvas_x_start, canvas_y_end - canvas_y_start), 255)
                    piece_image.paste(
                        padded_layer.crop((x_start, y_start, x_end, y_end)),
                        (x_start - canvas_x_start, y_start - canvas_y_start),
                    )
                else:
                    piece_image = padded_layer.crop((x_start, y_start, x_end, y_end))
                piece_path = piece["tiff_dir"] / f"slice_{index:04d}.tif"
                piece_image.save(piece_path, compression="tiff_deflate")
                piece["tiff_paths"].append(piece_path)

    z_values = list(state.get("z_values", []))
    if len(z_values) < len(tiff_paths):
        z_values.extend([0.0] * (len(tiff_paths) - len(z_values)))

    for piece in pieces:
        canvas_x_start = piece["x_start"]
        canvas_x_end = piece["x_end"]
        canvas_y_start = piece["y_start"]
        canvas_y_end = piece["y_end"]
        if overlapping_layers:
            canvas_x_start -= overlap_x
            canvas_x_end += overlap_x
            canvas_y_start -= overlap_y
            canvas_y_end += overlap_y
        canvas_x_start += overlap_x
        canvas_x_end += overlap_x
        canvas_y_start += overlap_y
        canvas_y_end += overlap_y
        piece_width = canvas_x_end - canvas_x_start
        piece_height = canvas_y_end - canvas_y_start
        zip_path = output_dir / f"{safe_name}_r{piece['row']:02d}_c{piece['col']:02d}_tiff_slices.zip"
        _zip_tiff_paths(piece["tiff_paths"], zip_path)
        piece_state: ViewerState = {
            "tiff_paths": [str(path) for path in piece["tiff_paths"]],
            "z_values": z_values[: len(piece["tiff_paths"])],
            "pixel_size": pixel_size,
            "x_min": working_x_min + (canvas_x_start * pixel_size),
            "y_min": working_y_min + ((working_height - canvas_y_end) * pixel_size),
            "image_width": piece_width,
            "image_height": piece_height,
            "zip_path": str(zip_path),
            "overlapping_layers": bool(overlapping_layers),
        }
        piece["state"] = piece_state
        piece["zip_path"] = zip_path
    return pieces


def split_tiff_stack_left_right(
    state: ViewerState,
    base_name: str = "split_shape",
    progress: gr.Progress = gr.Progress(),
) -> tuple[ViewerState, ViewerState, Path, Path]:
    pieces = split_tiff_stack_grid(state, base_name=base_name, columns=2, rows=1, progress=progress)
    return pieces[0]["state"], pieces[1]["state"], pieces[0]["zip_path"], pieces[1]["zip_path"]


SHAPE_SETTINGS_HEADERS = [
    "Shape",
    "STL",
    "Target X (mm)",
    "Target Y (mm)",
    "Target Z (mm)",
    "Pressure (psi)",
    "Valve",
    "Nozzle",
    "Port",
    "Color",
    "Contour Tracing",
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
    "str",
    "bool",
    "str",
]
SIMPLE_NOZZLE_SPACING_HEADERS = [
    "Spacing Mode",
    "Applies To",
    "X edge spacing (mm)",
    "Y nozzle spacing (mm)",
]
ADVANCED_NOZZLE_SPACING_HEADERS = [
    "From Nozzle",
    "To Nozzle",
    "X edge spacing (mm)",
    "Y nozzle spacing (mm)",
]
NOZZLE_SPACING_HEADERS = SIMPLE_NOZZLE_SPACING_HEADERS


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


def _records_from_files(files: Any, previous_records: list[dict] | None = None) -> list[dict]:
    previous_by_path: dict[str | None, list[dict]] = {}
    for record in previous_records or []:
        previous_by_path.setdefault(record.get("stl_path"), []).append(record)
    used_nozzles: set[int] = set()
    records: list[dict] = []
    for index, path in enumerate(_uploaded_file_paths(files), start=1):
        previous_queue = previous_by_path.get(path) or []
        previous = previous_queue.pop(0) if previous_queue else {}
        name = previous.get("name") or Path(path).stem or f"Shape {index}"
        default_x, default_y, default_z = _default_target_extents_for_stl(path)
        nozzle = _record_nozzle_number(previous, index) if previous else _next_unused_nozzle(used_nozzles)
        used_nozzles.add(nozzle)
        records.append({
            "idx": index,
            "name": name,
            "stl_path": path,
            "original_x": previous.get("original_x", default_x),
            "original_y": previous.get("original_y", default_y),
            "original_z": previous.get("original_z", default_z),
            "target_x": previous.get("target_x", default_x),
            "target_y": previous.get("target_y", default_y),
            "target_z": previous.get("target_z", default_z),
            "last_scaled_axis": previous.get("last_scaled_axis", "target_x"),
            "pressure": previous.get("pressure", 25.0),
            "valve": previous.get("valve", 4),
            "nozzle": nozzle,
            "port": previous.get("port", 1),
            "color": previous.get("color", _default_color(index)),
            "contour_tracing": previous.get("contour_tracing", False),
            "tiff_state": previous.get("tiff_state", _empty_state()),
            "zip_path": previous.get("zip_path"),
            "gcode_path": previous.get("gcode_path"),
        })
    return records


def _reindex_shape_records(records: list[dict]) -> list[dict]:
    reindexed: list[dict] = []
    for index, record in enumerate(records, start=1):
        copy = dict(record)
        copy["idx"] = index
        reindexed.append(copy)
    return reindexed


def _shape_settings_rows(records: list[dict]) -> list[list[Any]]:
    return [
        [
            record["idx"],
            record["name"],
            record.get("target_x", DEFAULT_TARGET_EXTENTS[0]),
            record.get("target_y", DEFAULT_TARGET_EXTENTS[1]),
            record.get("target_z", DEFAULT_TARGET_EXTENTS[2]),
            record.get("pressure", 25.0),
            record.get("valve", 4),
            _record_nozzle_number(record, int(record["idx"])),
            record.get("port", 1),
            record.get("color", _default_color(record["idx"])),
            bool(record.get("contour_tracing", False)),
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
            contour_pos = 10 if has_nozzle_column else 9
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
            if len(row) > color_pos and row[color_pos]:
                copy["color"] = str(row[color_pos])
            try:
                copy["contour_tracing"] = _coerce_bool(row[contour_pos], bool(copy.get("contour_tracing", False)))
            except IndexError:
                copy["contour_tracing"] = bool(copy.get("contour_tracing", False))
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
                old_value = float(previous.get(key))
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


def _spacing_table_update(records: list[dict], existing_table: Any | None = None, use_individual_spacing: bool | None = False) -> dict[str, Any]:
    pairs = _spacing_pairs_from_table(existing_table)
    first_pair = pairs[0] if pairs else (5.0, 0.0)

    if not use_individual_spacing:
        return gr.update(
            headers=SIMPLE_NOZZLE_SPACING_HEADERS,
            value=[["Same spacing", "All neighboring nozzles", first_pair[0], first_pair[1]]],
            row_count=(1, "fixed"),
            column_count=(len(SIMPLE_NOZZLE_SPACING_HEADERS), "fixed"),
            label="Nozzle Spacing",
        )

    rows: list[list[Any]] = []
    ordered_nozzles = _ordered_nozzle_numbers(records)
    for index, (first, second) in enumerate(zip(ordered_nozzles, ordered_nozzles[1:])):
        gap_x, gap_y = pairs[index] if index < len(pairs) else first_pair
        rows.append([
            _nozzle_spacing_label(first, records),
            _nozzle_spacing_label(second, records),
            gap_x,
            gap_y,
        ])
    return gr.update(
        headers=ADVANCED_NOZZLE_SPACING_HEADERS,
        value=rows,
        row_count=(len(rows), "fixed"),
        column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
        label="Advanced Nozzle Spacing",
    )


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
        label="Advanced Grid Spacing",
        visible=bool(use_individual_grid_spacing),
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


def _auto_align_grid_spacing_rows(
    records: list[dict],
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    raster_pattern: str | None,
) -> tuple[list[list[Any]], int, int, int]:
    spacing_rows, column_count, row_count = _grid_spacing_rows(
        records,
        columns,
        rows,
        column_spacing,
        row_spacing,
    )
    auto_x_gap, auto_y_gap = _auto_align_split_offsets(raster_pattern)
    records_by_nozzle = _records_by_nozzle(records)
    ordered_nozzles = _ordered_nozzle_numbers(records)
    aligned_count = 0
    for index, (first, second) in enumerate(zip(ordered_nozzles, ordered_nozzles[1:])):
        if not _split_pair_was_created_together(records_by_nozzle, first, second):
            continue
        gap_x, gap_y = _grid_default_gap_for_pair(index, column_count, auto_x_gap, auto_y_gap)
        spacing_rows[index][2] = gap_x
        spacing_rows[index][3] = gap_y
        aligned_count += 1
    return spacing_rows, column_count, row_count, aligned_count


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


def _auto_align_split_offsets(raster_pattern: str | None) -> tuple[float, float]:
    if raster_pattern == RASTER_PATTERN_Y_DIRECTION:
        return AUTO_ALIGN_Y_RASTER_OFFSETS
    return AUTO_ALIGN_X_RASTER_OFFSETS


def auto_align_split_parts_for_raster(
    records: list[dict] | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    raster_pattern: str | None,
) -> tuple:
    x_gap, y_gap = _auto_align_split_offsets(raster_pattern)
    raster_label = raster_pattern or RASTER_PATTERN_SAME_DIRECTION
    spacing_rows, column_count, row_count, aligned_count = _auto_align_grid_spacing_rows(
        records or [],
        columns,
        rows,
        column_spacing,
        row_spacing,
        raster_pattern,
    )
    if aligned_count <= 0:
        return (
            gr.update(value=NOZZLE_LAYOUT_GRID),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "No split-sibling nozzle connections found. Auto align was not applied.",
        )

    return (
        gr.update(value=NOZZLE_LAYOUT_GRID),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        gr.update(value=True),
        gr.update(
            headers=ADVANCED_NOZZLE_SPACING_HEADERS,
            value=spacing_rows,
            row_count=(len(spacing_rows), "fixed"),
            column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
            label="Advanced Grid Spacing",
            visible=True,
        ),
        (
            f"Auto aligned {aligned_count} split nozzle connection(s) for {raster_label} "
            f"in a {column_count} x {row_count} grid: Column Gap X {x_gap:.2f} mm, Row Gap Y {y_gap:.2f} mm."
        ),
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


def update_nozzle_spacing_mode(layout_mode: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    custom_selected = layout_mode == NOZZLE_LAYOUT_PAIR_TABLE
    return gr.update(visible=not custom_selected), gr.update(visible=custom_selected)


def update_lead_in_options_visibility(enabled: bool | None) -> dict[str, Any]:
    return gr.update(visible=bool(enabled))


def _dropdown_update(records: list[dict], selected: str | None = None) -> dict[str, Any]:
    choices = [_shape_choice(record) for record in records]
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _gcode_dropdown_update(records: list[dict], selected: str | None = None, include_upload: bool = False) -> dict[str, Any]:
    choices = [_shape_choice(record) for record in records if record.get("gcode_path")]
    if include_upload:
        choices.append(GCODE_SOURCE_UPLOAD)
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


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
    existing_spacing: Any | None = None,
    use_individual_spacing: bool | None = False,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    next_records = _records_from_files(files, records)
    settings = _shape_settings_rows(next_records)
    spacing = _spacing_table_update(next_records, existing_spacing, use_individual_spacing)
    return (
        next_records,
        settings,
        spacing,
        _dropdown_update(next_records),
        _gcode_dropdown_update(next_records),
        _gcode_dropdown_update(next_records, include_upload=True),
        [record.get("zip_path") for record in next_records if record.get("zip_path")],
        [record.get("gcode_path") for record in next_records if record.get("gcode_path")],
    )


def load_sample_shapes(
    files: Any,
    records: list[dict] | None,
    settings_table: Any | None = None,
    existing_spacing: Any | None = None,
    use_individual_spacing: bool | None = False,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    paths = [str(SAMPLE_STL_DIR / filename) for filename in SAMPLE_STL_FILENAMES if (SAMPLE_STL_DIR / filename).exists()]
    merged_paths = _append_file_paths(files, paths)
    return (
        gr.update(value=merged_paths),
        *sync_uploaded_shapes(merged_paths, records, None, existing_spacing, use_individual_spacing),
    )


def update_nozzle_spacing_table_mode(
    records: list[dict] | None,
    existing_spacing: Any | None,
    use_individual_spacing: bool | None,
) -> dict[str, Any]:
    return _spacing_table_update(records or [], existing_spacing, use_individual_spacing)


def _shape_delete_outputs(
    records: list[dict],
    existing_spacing: Any | None,
    use_individual_spacing: bool | None,
    last_delete_at: float | None,
    upload_update: Any | None = None,
) -> tuple:
    return (
        upload_update if upload_update is not None else gr.update(),
        records,
        _shape_settings_rows(records),
        _spacing_table_update(records, existing_spacing, use_individual_spacing),
        _dropdown_update(records),
        _gcode_dropdown_update(records),
        _gcode_dropdown_update(records, include_upload=True),
        [record.get("zip_path") for record in records if record.get("zip_path")],
        [record.get("gcode_path") for record in records if record.get("gcode_path")],
        float(last_delete_at or 0.0),
    )


def delete_shape_from_settings(
    records: list[dict] | None,
    settings_table: Any | None,
    existing_spacing: Any | None,
    use_individual_spacing: bool | None,
    last_delete_at: float | None,
    evt: gr.SelectData,
) -> tuple:
    now = time.monotonic()
    rows = _normalise_rows(settings_table)
    selected = getattr(evt, "index", None)
    current_records = _apply_shape_settings(records or [], settings_table)
    if not isinstance(selected, (list, tuple)) or len(selected) < 2:
        return _shape_delete_outputs(current_records, existing_spacing, use_individual_spacing, last_delete_at)

    try:
        row_index, column_index = int(selected[0]), int(selected[1])
    except (TypeError, ValueError):
        return _shape_delete_outputs(current_records, existing_spacing, use_individual_spacing, last_delete_at)
    delete_column_index = len(SHAPE_SETTINGS_HEADERS) - 1
    if column_index != delete_column_index or row_index < 0 or row_index >= len(rows):
        return _shape_delete_outputs(current_records, existing_spacing, use_individual_spacing, last_delete_at)
    if last_delete_at and now - float(last_delete_at) < DELETE_SHAPE_COOLDOWN_SECONDS:
        return _shape_delete_outputs(current_records, existing_spacing, use_individual_spacing, last_delete_at)

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
        existing_spacing,
        use_individual_spacing,
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
            copy["original_x"] = original_x
            copy["original_y"] = original_y
            copy["original_z"] = original_z
        copy["target_x"] = original_x
        copy["target_y"] = original_y
        copy["target_z"] = original_z
        copy["last_scaled_axis"] = "target_x"
        reset_records.append(copy)
    return reset_records, _shape_settings_rows(reset_records)


def normalize_shape_dimensions_for_mode(
    records: list[dict] | None,
    settings_table: Any | None,
    scale_mode: str | None,
) -> tuple:
    edited_axes = _last_edited_target_axes(records, settings_table)
    records = _apply_shape_settings(records or [], settings_table)
    if _normalize_scale_mode(scale_mode) != SCALE_MODE_UNIFORM_FACTOR:
        for record in records:
            idx = int(record.get("idx", 0))
            if idx in edited_axes:
                record["last_scaled_axis"] = edited_axes[idx]
        return records, _shape_settings_rows(records)

    normalized: list[dict] = []
    for record in records:
        copy = dict(record)
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
        anchor_key = edited_axes.get(idx) or copy.get("last_scaled_axis") or "target_x"
        try:
            anchor_index = TARGET_DIMENSION_KEYS.index(str(anchor_key))
        except ValueError:
            anchor_index = 0
        scale = float(targets[anchor_index] / originals[anchor_index])
        copy["last_scaled_axis"] = TARGET_DIMENSION_KEYS[anchor_index]
        scaled = originals * scale
        copy["target_x"] = round(float(scaled[0]), 6)
        copy["target_y"] = round(float(scaled[1]), 6)
        copy["target_z"] = round(float(scaled[2]), 6)
        normalized.append(copy)
    return normalized, _shape_settings_rows(normalized)


def normalize_shape_settings_and_spacing(
    records: list[dict] | None,
    settings_table: Any | None,
    scale_mode: str | None,
    existing_spacing: Any | None,
    use_individual_spacing: bool | None,
) -> tuple:
    updated_records, updated_settings = normalize_shape_dimensions_for_mode(records, settings_table, scale_mode)
    return (
        updated_records,
        updated_settings,
        _spacing_table_update(updated_records, existing_spacing, use_individual_spacing),
    )


def show_selected_model(
    records: list[dict] | None,
    selected: str | None,
    settings_table: Any,
    opacity: float,
    scale_mode: str | None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    pos = _selected_record_index(records, selected)
    if pos < 0:
        return _viewer_update(None), "No model loaded.", _reset_slider(), "No slice stack loaded yet.", None
    record = records[pos]
    viewer, details = load_single_model(
        record.get("stl_path"),
        opacity,
        True,
        scale_mode,
        record.get("target_x"),
        record.get("target_y"),
        record.get("target_z"),
    )
    state = record.get("tiff_state") or _empty_state()
    label, preview = _render_selected_slice(state, 0)
    slider = gr.update(
        minimum=0,
        maximum=max(0, len(state.get("tiff_paths", [])) - 1),
        value=0,
        step=1,
        interactive=len(state.get("tiff_paths", [])) > 1,
    )
    return viewer, details, slider, label, preview


def jump_to_selected_slice(records: list[dict] | None, selected: str | None, index: float) -> tuple[str, Image.Image | None]:
    pos = _selected_record_index(records or [], selected)
    if pos < 0:
        return "No slice stack loaded yet.", None
    return _render_selected_slice((records or [])[pos].get("tiff_state") or _empty_state(), int(index))


def shift_selected_slice(records: list[dict] | None, selected: str | None, index: float, delta: int) -> tuple:
    pos = _selected_record_index(records or [], selected)
    if pos < 0:
        return gr.update(value=0), "No slice stack loaded yet.", None
    state = (records or [])[pos].get("tiff_state") or _empty_state()
    paths = state.get("tiff_paths", [])
    if not paths:
        return gr.update(value=0), "No slice stack loaded yet.", None
    next_index = max(0, min(int(index) + delta, len(paths) - 1))
    label, preview = _render_selected_slice(state, next_index)
    return gr.update(value=next_index), label, preview


def _tiff_preview_update(records: list[dict], selected: str | None = None) -> tuple[dict[str, Any], dict[str, Any], str, Image.Image | None]:
    dropdown = _dropdown_update(records, selected)
    selected_value = dropdown.get("value") if isinstance(dropdown, dict) else selected
    pos = _selected_record_index(records, selected_value)
    state = (records[pos].get("tiff_state") if pos >= 0 else None) or _empty_state()
    label, preview = _render_selected_slice(state, 0)
    slider = gr.update(
        minimum=0,
        maximum=max(0, len(state.get("tiff_paths", [])) - 1),
        value=0,
        step=1,
        interactive=len(state.get("tiff_paths", [])) > 1,
    )
    return dropdown, slider, label, preview


def generate_dynamic_stacks(
    records: list[dict] | None,
    settings_table: Any,
    layer_height: float,
    pixel_size: float,
    scale_mode: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    if not records:
        return (
            records,
            [],
            "Upload at least one STL first.",
            _dropdown_update(records),
            _reset_slider(),
            "No slice stack loaded yet.",
            None,
            _empty_state(),
            _reset_slider(),
            "No reference stack generated yet.",
            None,
        )
    total = len(records)
    messages: list[str] = []
    for pos, record in enumerate(records):
        stl_path = record.get("stl_path")
        if not stl_path:
            messages.append(f"Shape {record['idx']}: skipped (no STL file).")
            continue

        def report_progress(cur: int, tot: int, offset: int = pos) -> None:
            progress((offset + cur / tot) / total, desc=f"Slicing shape {offset + 1} of {total}...")

        mesh = load_mesh(stl_path)
        scale_factors = _resolve_mesh_scale_factors(
            mesh,
            True,
            scale_mode,
            record.get("target_x"),
            record.get("target_y"),
            record.get("target_z"),
        )
        try:
            stack = slice_stl_to_tiffs(
                stl_path,
                layer_height=float(layer_height),
                pixel_size=float(pixel_size),
                progress_callback=report_progress,
                scale_factors=scale_factors,
            )
            record["tiff_state"] = _stack_to_state(stack)
            record["zip_path"] = str(stack.zip_path)
            messages.append(f"Shape {record['idx']}: wrote `{stack.zip_path.name}`.")
        except Exception as exc:
            messages.append(f"Shape {record['idx']}: failed ({exc}).")
    ref_state, ref_slider, ref_label, ref_preview = generate_dynamic_reference_stack(records, progress=progress)
    if (ref_state or {}).get("tiff_paths"):
        messages.append("Reference TIFF Stack: updated automatically.")
    else:
        messages.append("Reference TIFF Stack: skipped (no generated shape slices available).")
    selected_update, slider, label, preview = _tiff_preview_update(records)
    return (
        records,
        [record.get("zip_path") for record in records if record.get("zip_path")],
        "\n".join(messages),
        selected_update,
        slider,
        label,
        preview,
        ref_state,
        ref_slider,
        ref_label,
        ref_preview,
    )


def generate_dynamic_reference_stack(records: list[dict] | None, progress: gr.Progress = gr.Progress()) -> tuple:
    states = [record.get("tiff_state") or _empty_state() for record in (records or [])]
    return generate_reference_stack(*states, progress=progress)


def _split_piece_choice(piece: dict[str, Any]) -> str:
    return f"R{piece['row']} C{piece['col']}"


def _split_piece_dropdown_update(pieces: list[dict[str, Any]], selected: str | None = None) -> dict[str, Any]:
    choices = [_split_piece_choice(piece) for piece in pieces]
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _selected_split_piece(pieces: list[dict[str, Any]] | None, selected: str | None) -> dict[str, Any] | None:
    if not pieces:
        return None
    for piece in pieces:
        if _split_piece_choice(piece) == selected:
            return piece
    return pieces[0]


def preview_selected_split_piece(pieces: list[dict[str, Any]] | None, selected: str | None) -> tuple:
    piece = _selected_split_piece(pieces, selected)
    if not piece:
        return _reset_slider(), "No split stack generated yet.", None
    state = piece.get("state") or _empty_state()
    label, preview = _render_selected_slice(state, 0)
    slider = gr.update(
        minimum=0,
        maximum=max(0, len(state.get("tiff_paths", [])) - 1),
        value=0,
        step=1,
        interactive=len(state.get("tiff_paths", [])) > 1,
    )
    return slider, label, preview


def jump_to_selected_split_piece(pieces: list[dict[str, Any]] | None, selected: str | None, index: float) -> tuple:
    piece = _selected_split_piece(pieces, selected)
    if not piece:
        return "No split stack generated yet.", None
    return _render_selected_slice(piece.get("state") or _empty_state(), int(index))


def shift_selected_split_piece(pieces: list[dict[str, Any]] | None, selected: str | None, index: float, delta: int) -> tuple:
    piece = _selected_split_piece(pieces, selected)
    if not piece:
        return gr.update(value=0), "No split stack generated yet.", None
    state = piece.get("state") or _empty_state()
    new_index, label, preview = shift_slice(state, index, delta)
    return gr.update(value=new_index), label, preview


def split_selected_shape_for_grid(
    records: list[dict] | None,
    selected: str | None,
    settings_table: Any | None,
    existing_spacing: Any | None,
    use_individual_spacing: bool | None,
    columns: float,
    rows: float,
    overlapping_layers: bool,
    starting_nozzle: float,
    starting_valve: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    if not records:
        empty = _empty_state()
        return (
            records,
            _shape_settings_rows(records),
            _spacing_table_update(records, existing_spacing, use_individual_spacing),
            _dropdown_update(records),
            _reset_slider(),
            "No slice stack loaded yet.",
            None,
            [],
            [],
            _gcode_dropdown_update(records),
            _gcode_dropdown_update(records, include_upload=True),
            _dropdown_update(records),
            [],
            [],
            _split_piece_dropdown_update([]),
            _reset_slider(),
            "No split stack generated yet.",
            None,
            "Generate TIFF stacks for a shape before splitting it.",
        )

    pos = _selected_record_index(records, selected)
    if pos < 0:
        pos = 0
    source = records[pos]
    state = source.get("tiff_state") or _empty_state()
    try:
        pieces = split_tiff_stack_grid(
            state,
            base_name=str(source.get("name") or f"shape_{source.get('idx', pos + 1)}"),
            columns=columns,
            rows=rows,
            overlapping_layers=overlapping_layers,
            progress=progress,
        )
    except Exception as exc:
        empty = _empty_state()
        return (
            records,
            _shape_settings_rows(records),
            _spacing_table_update(records, existing_spacing, use_individual_spacing),
            _dropdown_update(records, selected),
            _reset_slider(),
            "No slice stack loaded yet.",
            None,
            [record.get("zip_path") for record in records if record.get("zip_path")],
            [record.get("gcode_path") for record in records if record.get("gcode_path")],
            _gcode_dropdown_update(records),
            _gcode_dropdown_update(records, include_upload=True),
            _dropdown_update(records, selected),
            [],
            [],
            _split_piece_dropdown_update([]),
            _reset_slider(),
            "No split stack generated yet.",
            None,
            f"Split failed: {exc}",
        )

    base_name = str(source.get("name") or f"Shape {source.get('idx', pos + 1)}")
    first_nozzle = max(1, _coerce_int(starting_nozzle, 1))
    first_valve = max(1, _coerce_int(starting_valve, _coerce_int(source.get("valve", 4), 4)))
    split_column_count = max(1, _coerce_int(columns, 2))
    split_row_count = max(1, _coerce_int(rows, 1))
    split_group_id = f"split-{int(time.time() * 1_000_000)}-{source.get('idx', pos + 1)}"
    split_records: list[dict] = []
    for index, piece in enumerate(pieces):
        piece_state = piece["state"]
        piece_width_mm = float(piece_state.get("image_width", 0) or 0) * float(piece_state.get("pixel_size", 0.0) or 0.0)
        piece_height_mm = float(piece_state.get("image_height", 0) or 0) * float(piece_state.get("pixel_size", 0.0) or 0.0)
        piece_record = dict(source)
        piece_record.update({
            "name": f"{base_name} - R{piece['row']}C{piece['col']}",
            "stl_path": None,
            "target_x": piece_width_mm or source.get("target_x", DEFAULT_TARGET_EXTENTS[0]),
            "target_y": piece_height_mm or source.get("target_y", DEFAULT_TARGET_EXTENTS[1]),
            "nozzle": first_nozzle + index,
            "valve": first_valve + index,
            "split_group_id": split_group_id,
            "split_index": index,
            "split_row": int(piece["row"]),
            "split_col": int(piece["col"]),
            "split_rows": split_row_count,
            "split_columns": split_column_count,
            "tiff_state": piece_state,
            "zip_path": str(piece["zip_path"]),
            "gcode_path": None,
        })
        split_records.append(piece_record)
    next_records = _reindex_shape_records([*records[:pos], *split_records, *records[pos + 1:]])
    split_selected = _shape_choice(next_records[pos]) if pos < len(next_records) else None
    selected_update, main_slider, main_label, main_preview = _tiff_preview_update(next_records, split_selected)
    slider, label, preview = preview_selected_split_piece(pieces, None)
    status = (
        f"Split Shape {source.get('idx', pos + 1)} into {len(pieces)} print-ready stacks "
        f"({max(1, _coerce_int(columns, 2))} columns x {max(1, _coerce_int(rows, 1))} rows).  \n"
        f"Nozzles {first_nozzle}-{first_nozzle + len(pieces) - 1}; valves {first_valve}-{first_valve + len(pieces) - 1}."
    )
    if overlapping_layers:
        status += "  \nOverlapping Layers is enabled: split boundaries alternate by 1 pixel per layer with small blank margins for alignment."
    return (
        next_records,
        _shape_settings_rows(next_records),
        _spacing_table_update(next_records, existing_spacing, use_individual_spacing),
        selected_update,
        main_slider,
        main_label,
        main_preview,
        [record.get("zip_path") for record in next_records if record.get("zip_path")],
        [record.get("gcode_path") for record in next_records if record.get("gcode_path")],
        _gcode_dropdown_update(next_records),
        _gcode_dropdown_update(next_records, include_upload=True),
        _dropdown_update(next_records),
        [str(piece["zip_path"]) for piece in pieces],
        pieces,
        _split_piece_dropdown_update(pieces),
        slider,
        label,
        preview,
        status,
    )


def _contour_tracing_sources(records: list[dict]) -> list[dict]:
    sources: list[dict] = []
    for record in records:
        if not record.get("contour_tracing"):
            continue
        state = record.get("tiff_state") or {}
        tiff_paths = state.get("tiff_paths") or []
        source = {
            "owner_idx": int(record.get("idx", len(sources) + 1)),
            "contour_mode": CONTOUR_MODE_ROW_ENVELOPE,
            "tiff_paths": list(tiff_paths),
            "zip_path": record.get("zip_path"),
        }
        if source["tiff_paths"] or source["zip_path"]:
            sources.append(source)
    return sources


def generate_dynamic_gcode(
    records: list[dict] | None,
    settings_table: Any,
    all_g1: bool,
    use_reference_motion: bool,
    raster_pattern: str | None,
    pressure_ramp_enabled: bool,
    lead_in_enabled: bool,
    lead_in_length: float,
    lead_in_clearance: float,
    lead_in_lines: float,
    ref_state: ViewerState | None,
    layer_height: float,
    pixel_size: float,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    motion_tiffs = (ref_state or {}).get("tiff_paths") if use_reference_motion else None
    contour_sources = _contour_tracing_sources(records)
    messages: list[str] = []
    if contour_sources:
        enabled = ", ".join(f"Shape {source['owner_idx']}" for source in contour_sources)
        messages.append(f"Shape-optimized contour tracing enabled for {enabled}.")
    for record in records:
        zip_path = record.get("zip_path")
        if not zip_path:
            messages.append(f"Shape {record['idx']}: skipped (no TIFF ZIP available).")
            continue
        if use_reference_motion and not motion_tiffs:
            messages.append(f"Shape {record['idx']}: skipped (Reference motion selected, but no Reference TIFF Stack exists).")
            continue
        shape_name = str(record.get("name") or Path(zip_path).stem).replace(" ", "_")
        try:
            gcode_path = generate_snake_path_gcode(
                zip_path=zip_path,
                shape_name=shape_name,
                pressure=float(record.get("pressure", 25.0)),
                valve=int(record.get("valve", 4)),
                port=int(record.get("port", 1)),
                layer_height=float(layer_height),
                fil_width=float(pixel_size),
                pressure_ramp_enabled=bool(pressure_ramp_enabled),
                all_g1=bool(all_g1),
                motion_tiffs=motion_tiffs,
                raster_pattern=raster_pattern,
                contour_tiff_sets=contour_sources,
                active_contour_owner=int(record.get("idx", 0)),
                lead_in_enabled=bool(lead_in_enabled),
                lead_in_length=float(lead_in_length),
                lead_in_clearance=float(lead_in_clearance),
                lead_in_lines=max(1, _coerce_int(lead_in_lines, 3)),
            )
            record["gcode_path"] = str(gcode_path)
            messages.append(f"Shape {record['idx']}: wrote `{gcode_path.name}`.")
        except Exception as exc:
            messages.append(f"Shape {record['idx']}: failed ({exc}).")
    return (
        records,
        [record.get("gcode_path") for record in records if record.get("gcode_path")],
        "\n".join(messages),
        _gcode_dropdown_update(records),
        _gcode_dropdown_update(records, include_upload=True),
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
            messages.append(f"Shape {idx}: no G-code (generate it on the TIFF Slices to GCode tab).")
            continue
        try:
            parsed = parse_gcode_path(Path(path).read_text())
        except OSError as exc:
            messages.append(f"Shape {idx}: failed to read ({exc}).")
            continue
        if not parsed.get("point_count"):
            messages.append(f"Shape {idx}: no G0/G1 moves found.")
            continue
        nozzle = _record_nozzle_number(record, idx)
        parts.append({"idx": idx, "nozzle": nozzle, "color": record.get("color", _default_color(idx)), "parsed": parsed})
        messages.append(f"Shape {idx} (Nozzle {nozzle}): {parsed['point_count']} moves, {parsed.get('layer_count', 0)} layer(s).")
    return parts, messages


def _spacing_args_from_table(spacing_table: Any, use_individual_spacing: bool | None = False) -> tuple[float, float, float, float, list[float]]:
    pairs = _spacing_pairs_from_table(spacing_table)
    if not pairs:
        pairs = [(5.0, 0.0)]
    if not use_individual_spacing:
        pairs = [pairs[0]]
    while len(pairs) < 2:
        pairs.append(pairs[0])
    extra = [value for pair in pairs[2:] for value in pair]
    return pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1], extra


def render_dynamic_nozzle_spacing(
    records: list[dict] | None,
    layout_mode: str | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
    use_individual_spacing: bool,
    spacing_table: Any,
) -> tuple[Any, str]:
    parts, _messages = _parts_from_records(records)
    if not parts:
        return None, "No shape G-code available. Generate G-code first."
    offsets, spacings = _resolve_layout_from_spacing_controls(
        parts,
        layout_mode,
        columns,
        rows,
        column_spacing,
        row_spacing,
        use_grid_individual_spacing,
        grid_spacing_table,
        use_individual_spacing,
        spacing_table,
    )
    return build_nozzle_spacing_figure(parts, offsets, spacings), _format_nozzle_spacing_status(parts, offsets, spacings)


def render_dynamic_toolpath(
    source: str | None,
    uploaded_path: str | None,
    records: list[dict] | None,
    travel_opacity: float,
    print_opacity: float,
    travel_color: str,
    print_color: str,
    print_width: float,
    travel_width: float,
    tube: bool = True,
) -> tuple[Any, str, dict]:
    if source == GCODE_SOURCE_UPLOAD:
        path = uploaded_path
        label = "uploaded file"
    else:
        pos = _selected_record_index(records or [], source)
        path = (records or [])[pos].get("gcode_path") if pos >= 0 else None
        label = source or "selected shape"
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
    return figure, (
        f"**{parsed['point_count']} moves parsed** - {len(parsed['print_segments'])} print segment(s), "
        f"{len(parsed['travel_segments'])} travel segment(s).  \n"
        f"Bounds: X [{x_min:.2f}, {x_max:.2f}], Y [{y_min:.2f}, {y_max:.2f}], Z [{z_min:.2f}, {z_max:.2f}] mm."
    ), parsed


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
    layout_mode: str | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
    use_individual_spacing: bool,
    spacing_table: Any,
    tube: bool = True,
) -> tuple[Any, str]:
    records = _apply_shape_settings(records or [], settings_table)
    parts, messages = _parts_from_records(records)
    if not parts:
        return None, "No shape G-code available. Generate G-code on the TIFF Slices to GCode tab first."
    offsets, spacings = _resolve_layout_from_spacing_controls(
        parts,
        layout_mode,
        columns,
        rows,
        column_spacing,
        row_spacing,
        use_grid_individual_spacing,
        grid_spacing_table,
        use_individual_spacing,
        spacing_table,
    )
    figure = build_parallel_figure(
        parts,
        part_offsets=offsets,
        filament_width=float(filament_width),
        travel_width=float(travel_width),
        travel_opacity=float(travel_opacity),
        tube=tube,
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
    layout_mode: str | None,
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
    use_individual_spacing: bool,
    spacing_table: Any,
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
    offsets, _spacings = _resolve_layout_from_spacing_controls(
        parts,
        layout_mode,
        columns,
        rows,
        column_spacing,
        row_spacing,
        use_grid_individual_spacing,
        grid_spacing_table,
        use_individual_spacing,
        spacing_table,
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
    with gr.Blocks(title="STL TIFF Slicer", css=APP_CSS, head=APP_HEAD + TOOLPATH_ANIM_HEAD + PARALLEL_ANIM_HEAD) as demo:
        shape_records = gr.State([])
        last_shape_delete_at = gr.State(0.0)
        ref_state = gr.State(_empty_state())
        split_piece_states = gr.State([])

        with gr.Tab("STL to TIFF Slicer"):
            gr.Markdown(
                """
                # STL to TIFF Slicer
                Upload any number of STL files, edit per-shape dimensions and print settings in the table, then generate TIFF stacks.
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
                    sync_uploads_button = gr.Button("Sync Uploaded STLs", variant="secondary", size="sm")
                    reset_dimensions_button = gr.Button("Reset Dimensions", variant="secondary", size="sm")
                    model_opacity = gr.Checkbox(label="Use 75% 3D Model Opacity", value=False)
                    scale_mode = gr.Radio(
                        choices=[SCALE_MODE_TARGET_DIMENSIONS, SCALE_MODE_UNIFORM_FACTOR],
                        value=SCALE_MODE_TARGET_DIMENSIONS,
                        label="Scaling Mode",
                    )

            shape_settings = gr.Dataframe(
                headers=SHAPE_SETTINGS_HEADERS,
                value=[],
                row_count=(0, "dynamic"),
                column_count=(len(SHAPE_SETTINGS_HEADERS), "fixed"),
                datatype=SHAPE_SETTINGS_DATATYPES,
                interactive=True,
                label="Shape Settings",
                elem_id="shape-settings-table",
            )
            with gr.Row():
                layer_height = gr.Number(label="Layer Height (mm)", value=0.8, minimum=0.0001, step=0.01)
                pixel_size = gr.Number(label="Pixel Size/Fill Width (mm)", value=0.8, minimum=0.0001, step=0.01)
                generate_button = gr.Button("Generate TIFF Stacks", variant="primary")

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
                split_button = gr.Button("Split Selected Shape into Grid Pieces", variant="primary")
                split_status = gr.Markdown("Generate a TIFF stack, then split it for multi-nozzle printing.")
                split_downloads = gr.File(label="Download Split TIFF ZIPs", file_count="multiple", interactive=False)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        split_piece_source = gr.Dropdown(label="Preview Generated Piece", choices=[], value=None, allow_custom_value=False)
                        with gr.Row():
                            split_piece_prev = gr.Button("Prev", scale=1, min_width=90, size="sm")
                            split_piece_next = gr.Button("Next", scale=1, min_width=90, size="sm")
                        split_piece_slider = gr.Slider(label="Piece Slice Index", minimum=0, maximum=0, value=0, step=1, interactive=False)
                        split_piece_label = gr.Markdown("No split stack generated yet.")
                    with gr.Column(scale=2, min_width=420):
                        split_piece_preview = gr.Image(label="Generated Piece Preview", type="pil", image_mode="RGB", height=330)

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
                    with gr.Column(scale=1, min_width=300):
                        model_details = gr.Markdown("No model loaded.")

                with gr.Row():
                    with gr.Column(scale=2, min_width=420):
                        slice_preview = gr.Image(label="Selected Slice Preview", type="pil", image_mode="RGB", height=320)
                    with gr.Column(scale=1, min_width=300):
                        slice_label = gr.Markdown("No slice stack loaded yet.")
                        with gr.Row():
                            prev_button = gr.Button("Prev", scale=1, min_width=90, size="sm")
                            next_button = gr.Button("Next", scale=1, min_width=90, size="sm")
                        slice_slider = gr.Slider(label="Slice Index", minimum=0, maximum=0, value=0, step=1, interactive=False)

            tiff_downloads = gr.File(label="Download TIFF ZIPs", file_count="multiple", interactive=False)
            slicer_status = gr.Markdown("")

            with gr.Accordion("Reference TIFF Stack Preview", open=False, elem_classes=["settings-accordion"]):
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        ref_generate_button = gr.Button("Generate Reference TIFF Stack", variant="primary")
                    with gr.Column(scale=3, min_width=250):
                        ref_slice_label = gr.Markdown("No reference stack generated yet.")
                        ref_slice_preview = gr.Image(label="Reference Slice Preview", type="pil", image_mode="RGB", height=270)
                        with gr.Row():
                            ref_prev_button = gr.Button("Prev", scale=1, min_width=90, size="sm")
                            ref_next_button = gr.Button("Next", scale=1, min_width=90, size="sm")
                        ref_slice_slider = gr.Slider(label="Slice Index", minimum=0, maximum=0, value=0, step=1, interactive=False)

        with gr.Tab("TIFF Slices to GCode"):
            gr.Markdown(
                """
                # TIFF Slices to GCode
                Generate G-code for every shape with a TIFF stack. Pressure, valve, nozzle, port, and color come from the Shape Settings table.
                """
            )
            gcode_use_ref_motion = gr.Checkbox(
                label="Use Reference Stack for motion (all shapes share one nozzle path; each dispenses only its own geometry).",
                value=True,
            )
            gcode_all_g1 = gr.Checkbox(label="Move at one constant speed (no fast travel moves)", value=True)
            gcode_pressure_ramp_enabled = gr.Checkbox(label="Increase Pressure Each Layer", value=True)
            gcode_raster_pattern = gr.Dropdown(
                label="Raster Pattern",
                choices=list(RASTER_PATTERN_CHOICES),
                value=RASTER_PATTERN_SAME_DIRECTION,
                allow_custom_value=False,
            )
            gcode_lead_in_enabled = gr.Checkbox(label="Lead In", value=False)
            with gr.Group(visible=False) as gcode_lead_in_options_group:
                with gr.Row():
                    gcode_lead_in_length = gr.Number(label="Lead In Length (mm)", value=5.0, minimum=0.1, step=0.1)
                    gcode_lead_in_clearance = gr.Number(label="Lead In Clearance (mm)", value=5.0, minimum=0.0, step=0.1)
                    gcode_lead_in_lines = gr.Number(label="Lead In Raster Lines", value=3, minimum=1, step=1)
            gcode_button = gr.Button("Generate G-Code", variant="primary")
            gcode_downloads = gr.File(label="Download G-Code Files", file_count="multiple", interactive=False, elem_classes=["gcode-download"])
            gcode_status = gr.Markdown("")
            with gr.Row():
                gcode_text_source = gr.Dropdown(label="Preview G-Code", choices=[], value=None, allow_custom_value=False)
                refresh_gcode_text_button = gr.Button("Refresh G-Code Preview", variant="secondary", size="sm")
            gcode_text = gr.Code(label="Selected G-Code", language=None, lines=18, max_lines=18, interactive=False, elem_classes=["gcode-view"])

            with gr.Accordion("Nozzle Spacing", open=False, elem_classes=["settings-accordion"]):
                nozzle_layout_mode = gr.Radio(
                    label="Spacing Mode",
                    choices=[NOZZLE_LAYOUT_GRID, NOZZLE_LAYOUT_PAIR_TABLE],
                    value=NOZZLE_LAYOUT_GRID,
                )
                with gr.Group(visible=True) as nozzle_grid_group:
                    with gr.Row():
                        nozzle_grid_preset = gr.Dropdown(
                            label="Common Layout",
                            choices=NOZZLE_LAYOUT_PRESETS,
                            value="Custom",
                            allow_custom_value=False,
                        )
                        nozzle_grid_columns = gr.Number(label="Grid Columns", value=2, minimum=1, step=1)
                        nozzle_grid_rows = gr.Number(label="Grid Rows", value=2, minimum=1, step=1)
                        nozzle_grid_column_spacing = gr.Number(label="Column Gap (X, mm)", value=0.0, step=0.1)
                        nozzle_grid_row_spacing = gr.Number(label="Row Gap (Y, mm)", value=0.0, step=0.1)
                    with gr.Row():
                        auto_align_split_parts_button = gr.Button("Auto Align Split Parts", variant="secondary", size="sm")
                        nozzle_grid_use_individual_spacing = gr.Checkbox(label="Use Different Grid Connection Gaps", value=False)
                    nozzle_grid_spacing_table = gr.Dataframe(
                        headers=ADVANCED_NOZZLE_SPACING_HEADERS,
                        value=[],
                        row_count=(0, "fixed"),
                        column_count=(len(ADVANCED_NOZZLE_SPACING_HEADERS), "fixed"),
                        interactive=True,
                        label="Advanced Grid Spacing",
                        visible=False,
                        elem_id="nozzle-grid-spacing-table",
                    )
                with gr.Group(visible=False) as nozzle_custom_group:
                    nozzle_use_individual_spacing = gr.Checkbox(label="Use Different Values for Each Nozzle Connection", value=False)
                    nozzle_spacing_table = gr.Dataframe(
                        headers=NOZZLE_SPACING_HEADERS,
                        value=[["Same spacing", "All neighboring nozzles", 5.0, 0.0]],
                        row_count=(1, "fixed"),
                        column_count=(len(NOZZLE_SPACING_HEADERS), "fixed"),
                        interactive=True,
                        label="Custom Spacing",
                        elem_id="nozzle-spacing-table",
                    )
                nozzle_preview_button = gr.Button("Visualize Nozzle Spacing", variant="secondary", elem_id="visualize-nozzle-spacing-button")
                with gr.Row():
                    with gr.Column(scale=3, min_width=420):
                        nozzle_spacing_plot = gr.Plot(label="Nozzle Spacing")
                    with gr.Column(scale=1, min_width=260):
                        nozzle_spacing_status = gr.Markdown("Generate G-code, then visualize nozzle spacing.")

        with gr.Tab("G-Code Visualization"):
            gr.Markdown("### 3D Tool-Path Viewer")
            with gr.Row():
                gcode_source = gr.Radio(choices=[GCODE_SOURCE_UPLOAD], value=GCODE_SOURCE_UPLOAD, label="G-Code source")
                with gr.Column(elem_id="gcode-upload-col"):
                    gcode_upload = gr.File(label="Upload G-Code", file_types=[".txt", ".gcode", ".nc"], interactive=True, height=110)

            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    render_line_button = gr.Button("Render Tool Path - Line Plot", variant="primary")
                    render_tube_button = gr.Button("Render Tool Path - Tube Plot with Animation", variant="primary")
                    gr.Markdown(
                        "&#9888;&#65039; For high-resolution models (small layer heights), the tube plot can take a while to build and render.",
                        elem_id="tube-render-warning",
                    )
                    anim_controls = gr.HTML(TOOLPATH_CONTROLS_HTML, visible=False)
                    with gr.Row():
                        travel_opacity_slider = gr.Slider(label="Travel (G0) opacity", minimum=0.0, maximum=1.0, value=0.2, step=0.05, min_width=150)
                        print_opacity_slider = gr.Slider(label="Print (G1) opacity", minimum=0.0, maximum=1.0, value=1.0, step=0.05, min_width=150)
                    with gr.Row():
                        travel_color_picker = gr.Dropdown(
                            label="Travel (G0) color",
                            choices=[("Grey", "#969696"), ("Orange", "#ff7f0e"), ("Green", "#2ca02c"), ("Red", "#d62728"), ("Purple", "#9467bd"), ("Pink", "#e377c2"), ("Black", "#000000"), ("White", "#ffffff")],
                            value="#969696",
                            allow_custom_value=False,
                            min_width=150,
                        )
                        print_color_picker = gr.Dropdown(
                            label="Print (G1) color",
                            choices=[("Blue", "#1f77b4"), ("Orange", "#ff7f0e"), ("Green", "#2ca02c"), ("Red", "#d62728"), ("Purple", "#9467bd"), ("Pink", "#e377c2"), ("Black", "#000000"), ("White", "#ffffff")],
                            value="#ff7f0e",
                            allow_custom_value=False,
                            min_width=150,
                        )
                    with gr.Row(visible=False) as width_row:
                        travel_width_slider = gr.Slider(label="Travel width (mm)", minimum=0.05, maximum=1.2, value=0.2, step=0.05, min_width=150)
                        print_width_slider = gr.Slider(label="Filament width (mm)", minimum=0.1, maximum=1.2, value=0.8, step=0.05, min_width=150)
                    toolpath_status = gr.Markdown("")
                with gr.Column(scale=3, min_width=500):
                    toolpath_plot = gr.Plot(label="Tool Path", elem_id="toolpath_plot")

            parsed_state = gr.State({})
            render_mode = gr.State("tube")

        with gr.Tab("Parallel Printing Visualization"):
            gr.Markdown(
                "### Parallel Printing Visualization\n"
                "Plots all generated shapes using the nozzle spacing configured on the TIFF Slices to GCode tab."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    parallel_line_button = gr.Button("Render Parallel Print - Line Plot", variant="primary")
                    parallel_render_button = gr.Button("Render Parallel Print - Tube Plot with Animation", variant="primary")
                    gr.Markdown("&#9888;&#65039; Building multiple tube plots can take a while for high-resolution models.", elem_id="parallel-render-warning")
                    parallel_anim_controls = gr.HTML(PARALLEL_CONTROLS_HTML, visible=False)
                    pp_travel_opacity = gr.Slider(label="Travel opacity (0 = hidden)", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
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

        shape_sync_outputs = [shape_records, shape_settings, nozzle_spacing_table, selected_shape, gcode_text_source, gcode_source, tiff_downloads, gcode_downloads]
        stl_upload.change(fn=sync_uploaded_shapes, inputs=[stl_upload, shape_records, shape_settings, nozzle_spacing_table, nozzle_use_individual_spacing], outputs=shape_sync_outputs).then(
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
        sync_uploads_button.click(fn=sync_uploaded_shapes, inputs=[stl_upload, shape_records, shape_settings, nozzle_spacing_table, nozzle_use_individual_spacing], outputs=shape_sync_outputs).then(
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
        load_samples_button.click(fn=load_sample_shapes, inputs=[stl_upload, shape_records, shape_settings, nozzle_spacing_table, nozzle_use_individual_spacing], outputs=[stl_upload, *shape_sync_outputs]).then(
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
        shape_settings.select(
            fn=delete_shape_from_settings,
            inputs=[shape_records, shape_settings, nozzle_spacing_table, nozzle_use_individual_spacing, last_shape_delete_at],
            outputs=[stl_upload, *shape_sync_outputs, last_shape_delete_at],
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

        preview_inputs = [shape_records, selected_shape, shape_settings, model_opacity, scale_mode]
        shape_settings.change(
            fn=normalize_shape_settings_and_spacing,
            inputs=[shape_records, shape_settings, scale_mode, nozzle_spacing_table, nozzle_use_individual_spacing],
            outputs=[shape_records, shape_settings, nozzle_spacing_table],
            queue=False,
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        selected_shape.change(fn=show_selected_model, inputs=preview_inputs, outputs=[model_viewer, model_details, slice_slider, slice_label, slice_preview])
        refresh_preview_button.click(fn=show_selected_model, inputs=preview_inputs, outputs=[model_viewer, model_details, slice_slider, slice_label, slice_preview])
        model_opacity.change(fn=show_selected_model, inputs=preview_inputs, outputs=[model_viewer, model_details, slice_slider, slice_label, slice_preview])
        scale_mode.change(
            fn=normalize_shape_dimensions_for_mode,
            inputs=[shape_records, shape_settings, scale_mode],
            outputs=[shape_records, shape_settings],
            queue=False,
        ).then(
            fn=show_selected_model,
            inputs=preview_inputs,
            outputs=[model_viewer, model_details, slice_slider, slice_label, slice_preview],
        )
        reset_dimensions_button.click(
            fn=reset_shape_dimensions,
            inputs=[shape_records, shape_settings],
            outputs=[shape_records, shape_settings],
        ).then(
            fn=show_selected_model,
            inputs=preview_inputs,
            outputs=[model_viewer, model_details, slice_slider, slice_label, slice_preview],
        )

        slice_slider.release(fn=jump_to_selected_slice, inputs=[shape_records, selected_shape, slice_slider], outputs=[slice_label, slice_preview], queue=False)
        prev_button.click(fn=lambda records, selected, idx: shift_selected_slice(records, selected, idx, -1), inputs=[shape_records, selected_shape, slice_slider], outputs=[slice_slider, slice_label, slice_preview], queue=False)
        next_button.click(fn=lambda records, selected, idx: shift_selected_slice(records, selected, idx, 1), inputs=[shape_records, selected_shape, slice_slider], outputs=[slice_slider, slice_label, slice_preview], queue=False)

        generate_button.click(
            fn=generate_dynamic_stacks,
            inputs=[shape_records, shape_settings, layer_height, pixel_size, scale_mode],
            outputs=[
                shape_records,
                tiff_downloads,
                slicer_status,
                selected_shape,
                slice_slider,
                slice_label,
                slice_preview,
                ref_state,
                ref_slice_slider,
                ref_slice_label,
                ref_slice_preview,
            ],
        ).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        )
        ref_generate_button.click(fn=generate_dynamic_reference_stack, inputs=[shape_records], outputs=[ref_state, ref_slice_slider, ref_slice_label, ref_slice_preview])
        ref_slice_slider.release(fn=jump_to_slice, inputs=[ref_state, ref_slice_slider], outputs=[ref_slice_label, ref_slice_preview], queue=False)
        ref_prev_button.click(fn=lambda sv, idx: shift_slice(sv, idx, -1), inputs=[ref_state, ref_slice_slider], outputs=[ref_slice_slider, ref_slice_label, ref_slice_preview], queue=False)
        ref_next_button.click(fn=lambda sv, idx: shift_slice(sv, idx, 1), inputs=[ref_state, ref_slice_slider], outputs=[ref_slice_slider, ref_slice_label, ref_slice_preview], queue=False)

        split_refresh_sources.click(fn=lambda records: _dropdown_update(records), inputs=[shape_records], outputs=[split_source], queue=False)
        split_button.click(
            fn=split_selected_shape_for_grid,
            inputs=[
                shape_records,
                split_source,
                shape_settings,
                nozzle_spacing_table,
                nozzle_use_individual_spacing,
                split_columns,
                split_rows,
                split_overlapping_layers,
                split_start_nozzle,
                split_start_valve,
            ],
            outputs=[
                shape_records,
                shape_settings,
                nozzle_spacing_table,
                selected_shape,
                slice_slider,
                slice_label,
                slice_preview,
                tiff_downloads,
                gcode_downloads,
                gcode_text_source,
                gcode_source,
                split_source,
                split_downloads,
                split_piece_states,
                split_piece_source,
                split_piece_slider,
                split_piece_label,
                split_piece_preview,
                split_status,
            ],
        ).then(
            fn=generate_dynamic_reference_stack,
            inputs=[shape_records],
            outputs=[ref_state, ref_slice_slider, ref_slice_label, ref_slice_preview],
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )
        split_piece_source.change(fn=preview_selected_split_piece, inputs=[split_piece_states, split_piece_source], outputs=[split_piece_slider, split_piece_label, split_piece_preview], queue=False)
        split_piece_slider.release(fn=jump_to_selected_split_piece, inputs=[split_piece_states, split_piece_source, split_piece_slider], outputs=[split_piece_label, split_piece_preview], queue=False)
        split_piece_prev.click(fn=lambda pieces, selected, idx: shift_selected_split_piece(pieces, selected, idx, -1), inputs=[split_piece_states, split_piece_source, split_piece_slider], outputs=[split_piece_slider, split_piece_label, split_piece_preview], queue=False)
        split_piece_next.click(fn=lambda pieces, selected, idx: shift_selected_split_piece(pieces, selected, idx, 1), inputs=[split_piece_states, split_piece_source, split_piece_slider], outputs=[split_piece_slider, split_piece_label, split_piece_preview], queue=False)

        gcode_button.click(
            fn=generate_dynamic_gcode,
            inputs=[
                shape_records,
                shape_settings,
                gcode_all_g1,
                gcode_use_ref_motion,
                gcode_raster_pattern,
                gcode_pressure_ramp_enabled,
                gcode_lead_in_enabled,
                gcode_lead_in_length,
                gcode_lead_in_clearance,
                gcode_lead_in_lines,
                ref_state,
                layer_height,
                pixel_size,
            ],
            outputs=[shape_records, gcode_downloads, gcode_status, gcode_text_source, gcode_source],
        ).then(
            fn=load_selected_gcode_text,
            inputs=[shape_records, gcode_text_source],
            outputs=[gcode_text],
        )
        gcode_lead_in_enabled.change(
            fn=update_lead_in_options_visibility,
            inputs=[gcode_lead_in_enabled],
            outputs=[gcode_lead_in_options_group],
            queue=False,
        )
        gcode_text_source.change(fn=load_selected_gcode_text, inputs=[shape_records, gcode_text_source], outputs=[gcode_text])
        refresh_gcode_text_button.click(fn=load_selected_gcode_text, inputs=[shape_records, gcode_text_source], outputs=[gcode_text])
        nozzle_layout_mode.change(
            fn=update_nozzle_spacing_mode,
            inputs=[nozzle_layout_mode],
            outputs=[nozzle_grid_group, nozzle_custom_group],
            queue=False,
        )
        auto_align_split_parts_button.click(
            fn=auto_align_split_parts_for_raster,
            inputs=[
                shape_records,
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
                gcode_raster_pattern,
            ],
            outputs=[
                nozzle_layout_mode,
                nozzle_grid_group,
                nozzle_custom_group,
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
            grid_spacing_control.change(
                fn=_grid_spacing_table_update,
                inputs=grid_spacing_refresh_inputs,
                outputs=[nozzle_grid_spacing_table],
                queue=False,
            )
        nozzle_use_individual_spacing.change(fn=update_nozzle_spacing_table_mode, inputs=[shape_records, nozzle_spacing_table, nozzle_use_individual_spacing], outputs=[nozzle_spacing_table], queue=False)
        nozzle_preview_button.click(
            fn=render_dynamic_nozzle_spacing,
            inputs=[
                shape_records,
                nozzle_layout_mode,
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
                nozzle_grid_use_individual_spacing,
                nozzle_grid_spacing_table,
                nozzle_use_individual_spacing,
                nozzle_spacing_table,
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
        render_inputs = [
            gcode_source,
            gcode_upload,
            shape_records,
            travel_opacity_slider,
            print_opacity_slider,
            travel_color_picker,
            print_color_picker,
            print_width_slider,
            travel_width_slider,
        ]
        render_line_button.click(fn=render_dynamic_toolpath_lines, inputs=render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state, render_mode, anim_controls, width_row])
        render_tube_button.click(fn=render_dynamic_toolpath_tubes, inputs=render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state, render_mode, anim_controls, width_row])
        travel_width_slider.release(fn=rerender_dynamic_toolpath_current_mode, inputs=[render_mode] + render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state])
        print_width_slider.release(fn=rerender_dynamic_toolpath_current_mode, inputs=[render_mode] + render_inputs, outputs=[toolpath_plot, toolpath_status, parsed_state])

        def sync_width_sliders(v: float):
            height = float(v or 0.8)
            travel = height / 4
            return (
                gr.update(value=height, minimum=min(0.1, height), maximum=height * 1.5),
                gr.update(value=travel, minimum=min(0.05, travel), maximum=height * 1.5),
            )

        layer_height.change(fn=sync_width_sliders, inputs=[layer_height], outputs=[print_width_slider, travel_width_slider], queue=False)

        parallel_render_inputs = [
            shape_records,
            shape_settings,
            pp_travel_opacity,
            pp_filament_width,
            pp_travel_width,
            nozzle_layout_mode,
            nozzle_grid_columns,
            nozzle_grid_rows,
            nozzle_grid_column_spacing,
            nozzle_grid_row_spacing,
            nozzle_grid_use_individual_spacing,
            nozzle_grid_spacing_table,
            nozzle_use_individual_spacing,
            nozzle_spacing_table,
        ]
        parallel_outputs = [parallel_plot, parallel_status, parallel_mode, parallel_anim_controls, pp_width_row, pp_export_group]
        parallel_line_button.click(fn=render_dynamic_parallel_lines, inputs=parallel_render_inputs, outputs=parallel_outputs)
        parallel_render_button.click(fn=render_dynamic_parallel_tubes, inputs=parallel_render_inputs, outputs=parallel_outputs)
        pp_filament_width.release(fn=rerender_dynamic_parallel_current_mode, inputs=[parallel_mode] + parallel_render_inputs, outputs=[parallel_plot, parallel_status])
        pp_travel_width.release(fn=rerender_dynamic_parallel_current_mode, inputs=[parallel_mode] + parallel_render_inputs, outputs=[parallel_plot, parallel_status])
        pp_export_button.click(
            fn=export_dynamic_parallel_gif,
            inputs=[
                shape_records,
                shape_settings,
                pp_gif_travel_opacity,
                nozzle_layout_mode,
                nozzle_grid_columns,
                nozzle_grid_rows,
                nozzle_grid_column_spacing,
                nozzle_grid_row_spacing,
                nozzle_grid_use_individual_spacing,
                nozzle_grid_spacing_table,
                nozzle_use_individual_spacing,
                nozzle_spacing_table,
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
