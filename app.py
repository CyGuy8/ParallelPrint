from __future__ import annotations

import math
import tempfile
import time
import warnings
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
    load_mesh,
    scale_factors_for_target_extents,
    scale_mesh,
    slice_stl_to_layers,
)
from vector_gcode import generate_vector_gcode
from vector_toolpath import (
    RASTER_PATTERN_CHOICES,
    RASTER_PATTERN_SAME_DIRECTION,
    RASTER_PATTERN_Y_DIRECTION,
    ContourSource,
    build_reference_stack,
    split_layer_stack_grid,
)


SAMPLE_STL_FILENAMES = ("Hollow_Pyramid.stl", "Rounded_Cube_Through_Holes.stl", "halfsphere.stl")
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


PARALLEL_COLOR_CHOICES = [
    ("Orange", "#ff7f0e"), ("Blue", "#1f77b4"), ("Green", "#2ca02c"),
    ("Red", "#d62728"), ("Purple", "#9467bd"), ("Pink", "#e377c2"),
    ("Teal", "#17becf"), ("Black", "#000000"),
]
DEFAULT_PARALLEL_COLORS = ("#ff7f0e", "#1f77b4", "#2ca02c")


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
    "Target X (mm)",
    "Target Y (mm)",
    "Target Z (mm)",
    "Pressure (psi)",
    "Valve",
    "Nozzle",
    "Port",
    "Color",
    "Infill %",
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
    "number",
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
            "infill": previous.get("infill", 100.0),
            "contour_tracing": previous.get("contour_tracing", False),
            "layer_stack": previous.get("layer_stack"),
            "slice_params": previous.get("slice_params"),
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
            _coerce_float(record.get("infill", 100.0), 100.0),
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
            if len(row) > color_pos and row[color_pos]:
                copy["color"] = str(row[color_pos])
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
    )


def load_sample_shapes(
    files: Any,
    records: list[dict] | None,
    settings_table: Any | None = None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    paths = [str(SAMPLE_STL_DIR / filename) for filename in SAMPLE_STL_FILENAMES if (SAMPLE_STL_DIR / filename).exists()]
    merged_paths = _append_file_paths(files, paths)
    return (
        gr.update(value=merged_paths),
        *sync_uploaded_shapes(merged_paths, records, None),
    )


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
        float(last_delete_at or 0.0),
    )


def delete_shape_from_settings(
    records: list[dict] | None,
    settings_table: Any | None,
    last_delete_at: float | None,
    evt: gr.SelectData,
) -> tuple:
    now = time.monotonic()
    rows = _normalise_rows(settings_table)
    selected = getattr(evt, "index", None)
    current_records = _apply_shape_settings(records or [], settings_table)
    if not isinstance(selected, (list, tuple)) or len(selected) < 2:
        return _shape_delete_outputs(current_records, last_delete_at)

    try:
        row_index, column_index = int(selected[0]), int(selected[1])
    except (TypeError, ValueError):
        return _shape_delete_outputs(current_records, last_delete_at)
    delete_column_index = len(SHAPE_SETTINGS_HEADERS) - 1
    if column_index != delete_column_index or row_index < 0 or row_index >= len(rows):
        return _shape_delete_outputs(current_records, last_delete_at)
    if last_delete_at and now - float(last_delete_at) < DELETE_SHAPE_COOLDOWN_SECONDS:
        return _shape_delete_outputs(current_records, last_delete_at)

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
        return _viewer_update(None), "No model loaded."
    record = records[pos]
    return load_single_model(
        record.get("stl_path"),
        opacity,
        True,
        scale_mode,
        record.get("target_x"),
        record.get("target_y"),
        record.get("target_z"),
    )


def _slice_params_snapshot(record: dict, layer_height: float, scale_mode: str | None) -> dict:
    return {
        "layer_height": float(layer_height),
        "scale_mode": _normalize_scale_mode(scale_mode),
        "target_x": record.get("target_x"),
        "target_y": record.get("target_y"),
        "target_z": record.get("target_z"),
    }


def _slice_record(
    record: dict,
    layer_height: float,
    scale_mode: str | None,
    progress_callback=None,
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
    )
    record["layer_stack"] = stack
    record["slice_params"] = _slice_params_snapshot(record, layer_height, scale_mode)
    return stack


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
            stack = _slice_record(record, layer_height, scale_mode, report_progress)
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
    return build_reference_stack(
        [record.get("layer_stack") for record in (records or [])],
        grid=float(fil_width) if fil_width else None,
    )


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
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)

    def _outputs(next_records: list[dict], selected_value: str | None, status: str) -> tuple:
        return (
            next_records,
            _shape_settings_rows(next_records),
            _dropdown_update(next_records, selected_value),
            [record.get("gcode_path") for record in next_records if record.get("gcode_path")],
            _gcode_dropdown_update(next_records),
            _gcode_dropdown_update(next_records, include_upload=True),
            _dropdown_update(next_records, selected_value),
            status,
        )

    if not records:
        return _outputs(records, None, "Slice a shape before splitting it.")

    pos = _selected_record_index(records, selected)
    if pos < 0:
        pos = 0
    source = records[pos]
    stack = source.get("layer_stack")
    if stack is None or not getattr(stack, "layers", None):
        return _outputs(
            records,
            selected,
            f"Split failed: Shape {source.get('idx', pos + 1)} has no sliced layers yet - press Slice Shapes first.",
        )

    split_column_count = max(1, _coerce_int(columns, 2))
    split_row_count = max(1, _coerce_int(rows, 1))
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
    resliced = False
    for record in records:
        stl_path = record.get("stl_path")
        if not stl_path:
            continue  # Split pieces carry their clipped layers; nothing to re-slice.
        current = _slice_params_snapshot(record, layer_height, scale_mode)
        if record.get("layer_stack") is not None and record.get("slice_params") == current:
            continue
        try:
            stack = _slice_record(record, layer_height, scale_mode)
            messages.append(
                f"Shape {record['idx']}: sliced automatically ({len(stack.layers)} layers)."
            )
        except Exception as exc:
            record["layer_stack"] = None
            messages.append(f"Shape {record['idx']}: slicing failed ({exc}).")
        resliced = True
    return resliced


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
    ref_layers: LayerStack | None,
    layer_height: float,
    fil_width: float,
    scale_mode: str | None,
) -> tuple:
    records = _apply_shape_settings(records or [], settings_table)
    messages: list[str] = []
    resliced = _ensure_records_sliced(records, layer_height, scale_mode, messages)
    if use_reference_motion:
        # Always rebuild with the CURRENT fil width: the reference stack's
        # alignment snap grid must match the fil the G-code is generated with.
        ref_layers = generate_dynamic_reference_stack(records, fil_width)
    elif resliced:
        ref_layers = generate_dynamic_reference_stack(records, fil_width)
    contour_sources = _contour_tracing_sources(records)
    if contour_sources:
        enabled = ", ".join(f"Shape {source.owner_idx}" for source in contour_sources)
        messages.append(f"Contour tracing enabled for {enabled}.")
    for record in records:
        stack = record.get("layer_stack")
        if stack is None or not getattr(stack, "layers", None):
            messages.append(f"Shape {record['idx']}: skipped (no sliced layers).")
            continue
        if use_reference_motion and ref_layers is None:
            messages.append(f"Shape {record['idx']}: skipped (Reference motion selected, but no shapes are sliced).")
            continue
        shape_name = str(record.get("name") or stack.name or f"shape_{record['idx']}").replace(" ", "_")
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
                all_g1=bool(all_g1),
                motion=ref_layers if use_reference_motion else None,
                raster_pattern=raster_pattern,
                contour_sources=contour_sources,
                active_contour_owner=int(record.get("idx", 0)),
                infill=_coerce_float(record.get("infill", 100.0), 100.0) / 100.0,
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
        ref_layers,
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
            messages.append(f"Shape {idx}: no G-code (generate it on the Generate G-Code tab).")
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
    columns: Any,
    rows: Any,
    column_spacing: Any,
    row_spacing: Any,
    use_grid_individual_spacing: bool,
    grid_spacing_table: Any,
    tube: bool = True,
) -> tuple[Any, str]:
    records = _apply_shape_settings(records or [], settings_table)
    parts, messages = _parts_from_records(records)
    if not parts:
        return None, "No shape G-code available. Generate G-code on the Generate G-Code tab first."
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

        with gr.Tab("Shapes & Slicing"):
            gr.Markdown(
                """
                # Shapes & Slicing
                Upload any number of STL files, edit per-shape dimensions and print settings in the table, then slice each shape into per-layer outlines.
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
                fil_width = gr.Number(label="Filament/Line Width (mm)", value=0.8, minimum=0.0001, step=0.01)
                generate_button = gr.Button("Slice Shapes", variant="primary")

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
                split_status = gr.Markdown("Slice a shape, then split it for multi-nozzle printing.")

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

            slicer_status = gr.Markdown("")

        with gr.Tab("Generate G-Code"):
            gr.Markdown(
                """
                # Generate G-Code
                Generate G-code for every sliced shape. Pressure, valve, nozzle, port, and color come from the Shape Settings table.
                """
            )
            gcode_use_ref_motion = gr.Checkbox(
                label="Use combined reference outline for motion (all shapes share one nozzle path; each dispenses only its own geometry).",
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
                        nozzle_grid_column_spacing = gr.Number(label="Column Gap (X, mm)", value=0.0, step=0.1)
                        nozzle_grid_row_spacing = gr.Number(label="Row Gap (Y, mm)", value=0.0, step=0.1)
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
                "Plots all generated shapes using the nozzle spacing configured on the Generate G-Code tab."
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

        shape_sync_outputs = [shape_records, shape_settings, selected_shape, gcode_text_source, gcode_source, gcode_downloads]
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
        load_samples_button.click(fn=load_sample_shapes, inputs=[stl_upload, shape_records, shape_settings], outputs=[stl_upload, *shape_sync_outputs]).then(
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
            inputs=[shape_records, shape_settings, last_shape_delete_at],
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
        model_opacity.change(fn=show_selected_model, inputs=preview_inputs, outputs=[model_viewer, model_details])
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

        generate_button.click(
            fn=generate_dynamic_layer_stacks,
            inputs=[shape_records, shape_settings, layer_height, scale_mode, fil_width],
            outputs=[shape_records, slicer_status, ref_layers],
        ).then(
            fn=lambda records: _dropdown_update(records),
            inputs=[shape_records],
            outputs=[split_source],
            queue=False,
        )

        split_refresh_sources.click(fn=lambda records: _dropdown_update(records), inputs=[shape_records], outputs=[split_source], queue=False)
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
            ],
        ).then(
            fn=generate_dynamic_reference_stack,
            inputs=[shape_records, fil_width],
            outputs=[ref_layers],
        ).then(
            fn=_grid_spacing_table_update,
            inputs=grid_spacing_refresh_inputs,
            outputs=[nozzle_grid_spacing_table],
            queue=False,
        )

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
                ref_layers,
                layer_height,
                fil_width,
                scale_mode,
            ],
            outputs=[shape_records, ref_layers, gcode_downloads, gcode_status, gcode_text_source, gcode_source],
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
            nozzle_grid_columns,
            nozzle_grid_rows,
            nozzle_grid_column_spacing,
            nozzle_grid_row_spacing,
            nozzle_grid_use_individual_spacing,
            nozzle_grid_spacing_table,
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
