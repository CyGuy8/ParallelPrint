from __future__ import annotations

import tempfile
import math
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
from PIL import Image, ImageDraw, ImageFont
import trimesh

from gcode_viewer import (
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
from tiff_to_gcode import generate_snake_path_gcode


ViewerState = dict[str, Any]
SAMPLE_STL_FILENAMES = ("Hollow_Pyramid.stl", "Rounded_Cube_Through_Holes.stl", "halfsphere.stl")
SAMPLE_STL_DIR = Path(__file__).resolve().parent / "sample_stls"
DEFAULT_TARGET_EXTENTS = (20.0, 20.0, 20.0)
DEFAULT_UNIFORM_SCALE = 1.0
SCALE_MODE_TARGET_DIMENSIONS = "Fit X/Y/Z Targets"
SCALE_MODE_UNIFORM_FACTOR = "Uniform Scale Factor"
FRONT_CAMERA = (90, 80, None)
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

#load-sample-stls-button,
#load-sample-stls-button button {
    background: #f97316 !important;
    border-color: #ea580c !important;
    color: #ffffff !important;
}

#load-sample-stls-button:hover,
#load-sample-stls-button button:hover {
    background: #ea580c !important;
    border-color: #c2410c !important;
}

#load-sample-stls-button:focus-visible,
#load-sample-stls-button button:focus-visible {
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.35) !important;
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
    function start() {
        enableUndoButtons();
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
        f"- Extents: `{extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f}`",
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


def _resolve_uniform_scale(
    scale_to_target: bool | None,
    uniform_scale: float | None,
) -> float | None:
    if not scale_to_target:
        return None

    if uniform_scale is None:
        raise ValueError("Uniform scale factor is required when uniform STL scaling is enabled.")

    scale = float(uniform_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Uniform scale factor must be greater than zero.")

    return scale


def _normalize_scale_mode(scale_mode: str | None) -> str:
    if scale_mode == SCALE_MODE_UNIFORM_FACTOR:
        return SCALE_MODE_UNIFORM_FACTOR
    return SCALE_MODE_TARGET_DIMENSIONS


def _shape_target_values(
    target1_x: float | None,
    target1_y: float | None,
    target1_z: float | None,
    target2_x: float | None,
    target2_y: float | None,
    target2_z: float | None,
    target3_x: float | None,
    target3_y: float | None,
    target3_z: float | None,
) -> tuple[tuple[float | None, float | None, float | None], ...]:
    return (
        (target1_x, target1_y, target1_z),
        (target2_x, target2_y, target2_z),
        (target3_x, target3_y, target3_z),
    )


def _shape_uniform_values(
    uniform1: float | None,
    uniform2: float | None,
    uniform3: float | None,
) -> tuple[float | None, float | None, float | None]:
    return (uniform1, uniform2, uniform3)


def _resolve_mesh_scale_factors(
    mesh: trimesh.Trimesh,
    scale_to_target: bool | None,
    scale_mode: str | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
    uniform_scale: float | None,
) -> tuple[float, float, float] | None:
    if not scale_to_target:
        return None

    if _normalize_scale_mode(scale_mode) == SCALE_MODE_UNIFORM_FACTOR:
        scale = _resolve_uniform_scale(True, uniform_scale)
        return (scale, scale, scale)

    target_extents = _resolve_target_extents(True, target_x, target_y, target_z)
    if target_extents is None:
        return None
    return scale_factors_for_target_extents(mesh, target_extents)


def _load_model_mesh(
    stl_file: str | Path,
    scale_to_target: bool | None = False,
    scale_mode: str | None = SCALE_MODE_TARGET_DIMENSIONS,
    target_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform_scale: float | None = DEFAULT_UNIFORM_SCALE,
) -> tuple[trimesh.Trimesh, tuple[float, float, float]]:
    mesh = load_mesh(stl_file)
    scale_factors = _resolve_mesh_scale_factors(
        mesh,
        scale_to_target,
        scale_mode,
        target_x,
        target_y,
        target_z,
        uniform_scale,
    )
    if scale_factors is None:
        return mesh, (1.0, 1.0, 1.0)
    return scale_mesh(mesh, scale_factors), scale_factors


def _viewer_update(model_path: str | None) -> dict[str, Any]:
    return gr.update(value=model_path, camera_position=FRONT_CAMERA)


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
    uniform_scale: float | None = DEFAULT_UNIFORM_SCALE,
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
        uniform_scale=uniform_scale,
    )
    glb_path = _build_annotated_scene(mesh, opacity=_resolve_model_opacity(opacity))
    return _viewer_update(glb_path), _format_model_details(Path(stl_file).name, mesh, scale_factors)


def preload_sample_models(
    opacity: float = 1.0,
    scale_to_target: bool | None = False,
    scale_mode: str | None = SCALE_MODE_TARGET_DIMENSIONS,
    target1_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target1_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target1_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform1: float | None = DEFAULT_UNIFORM_SCALE,
    target2_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target2_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target2_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform2: float | None = DEFAULT_UNIFORM_SCALE,
    target3_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target3_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target3_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform3: float | None = DEFAULT_UNIFORM_SCALE,
) -> tuple:
    outputs: list[Any] = []
    resolved_opacity = _resolve_model_opacity(opacity)
    target_values = _shape_target_values(
        target1_x,
        target1_y,
        target1_z,
        target2_x,
        target2_y,
        target2_z,
        target3_x,
        target3_y,
        target3_z,
    )
    uniform_values = _shape_uniform_values(uniform1, uniform2, uniform3)

    for index, filename in enumerate(SAMPLE_STL_FILENAMES):
        stl_path = SAMPLE_STL_DIR / filename
        if not stl_path.exists():
            outputs.extend([
                None,
                _viewer_update(None),
                f"Sample file not found: {stl_path}",
            ])
            continue

        try:
            mesh, scale_factors = _load_model_mesh(
                stl_path,
                scale_to_target=scale_to_target,
                scale_mode=scale_mode,
                target_x=target_values[index][0],
                target_y=target_values[index][1],
                target_z=target_values[index][2],
                uniform_scale=uniform_values[index],
            )
        except Exception as exc:
            outputs.extend([
                str(stl_path),
                _viewer_update(None),
                f"Failed to load sample model: {stl_path.name} ({exc})",
            ])
            continue

        outputs.extend([
            str(stl_path),
            _viewer_update(_build_annotated_scene(mesh, opacity=resolved_opacity)),
            _format_model_details(stl_path.name, mesh, scale_factors),
        ])

    return tuple(outputs)


def refresh_all_model_viewers(
    stl1: str | None,
    stl2: str | None,
    stl3: str | None,
    opacity: float,
    scale_to_target: bool | None = False,
    scale_mode: str | None = SCALE_MODE_TARGET_DIMENSIONS,
    target1_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target1_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target1_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform1: float | None = DEFAULT_UNIFORM_SCALE,
    target2_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target2_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target2_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform2: float | None = DEFAULT_UNIFORM_SCALE,
    target3_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target3_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target3_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform3: float | None = DEFAULT_UNIFORM_SCALE,
) -> tuple:
    outputs: list[Any] = []
    resolved_opacity = _resolve_model_opacity(opacity)
    target_values = _shape_target_values(
        target1_x,
        target1_y,
        target1_z,
        target2_x,
        target2_y,
        target2_z,
        target3_x,
        target3_y,
        target3_z,
    )
    uniform_values = _shape_uniform_values(uniform1, uniform2, uniform3)

    for stl_file, values, uniform_scale in zip((stl1, stl2, stl3), target_values, uniform_values):
        if not stl_file:
            outputs.extend([_viewer_update(None), "No model loaded."])
            continue
        outputs.extend(
            load_single_model(
                stl_file,
                resolved_opacity,
                scale_to_target,
                scale_mode,
                values[0],
                values[1],
                values[2],
                uniform_scale,
            )
        )
    return tuple(outputs)


def generate_all_stacks(
    stl1: str | None,
    stl2: str | None,
    stl3: str | None,
    layer_height: float,
    pixel_size: float,
    scale_to_target: bool | None = False,
    scale_mode: str | None = SCALE_MODE_TARGET_DIMENSIONS,
    target1_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target1_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target1_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform1: float | None = DEFAULT_UNIFORM_SCALE,
    target2_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target2_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target2_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform2: float | None = DEFAULT_UNIFORM_SCALE,
    target3_x: float | None = DEFAULT_TARGET_EXTENTS[0],
    target3_y: float | None = DEFAULT_TARGET_EXTENTS[1],
    target3_z: float | None = DEFAULT_TARGET_EXTENTS[2],
    uniform3: float | None = DEFAULT_UNIFORM_SCALE,
    progress: gr.Progress = gr.Progress(),
):
    files = [stl1, stl2, stl3]
    target_values = _shape_target_values(
        target1_x,
        target1_y,
        target1_z,
        target2_x,
        target2_y,
        target2_z,
        target3_x,
        target3_y,
        target3_z,
    )
    uniform_values = _shape_uniform_values(uniform1, uniform2, uniform3)
    valid_count = max(1, sum(1 for f in files if f))
    results: list = []
    completed = 0

    for stl_file, values, uniform_scale in zip(files, target_values, uniform_values):
        if not stl_file:
            results.extend([
                _empty_state(),
                _reset_slider(),
                "No slice stack loaded yet.",
                None,
                None,
            ])
            continue

        slot_offset = completed

        def report_progress(cur: int, tot: int, offset: int = slot_offset) -> None:
            progress(
                (offset + cur / tot) / valid_count,
                desc=f"Slicing object {offset + 1} of {valid_count}\u2026",
            )

        scale_factors = None
        if scale_to_target:
            mesh = load_mesh(stl_file)
            scale_factors = _resolve_mesh_scale_factors(
                mesh,
                scale_to_target,
                scale_mode,
                values[0],
                values[1],
                values[2],
                uniform_scale,
            )

        stack = slice_stl_to_tiffs(
            stl_file,
            layer_height=layer_height,
            pixel_size=pixel_size,
            progress_callback=report_progress,
            scale_factors=scale_factors,
        )
        state = _stack_to_state(stack)
        label, preview = _render_selected_slice(state, 0)
        slider = gr.update(
            minimum=0,
            maximum=max(0, len(stack.tiff_paths) - 1),
            value=0,
            step=1,
            interactive=len(stack.tiff_paths) > 1,
        )
        results.extend([
            state,
            slider,
            label,
            preview,
            str(stack.zip_path),
        ])
        completed += 1

    return tuple(results)


def jump_to_slice(state: ViewerState, index: float) -> tuple[str, Image.Image | None]:
    return _render_selected_slice(state, int(index))


def run_all_tiff_to_gcode(
    zip1: str | None,
    zip2: str | None,
    zip3: str | None,
    pressure1: float,
    valve1: float,
    port1: float,
    pressure2: float,
    valve2: float,
    port2: float,
    pressure3: float,
    valve3: float,
    port3: float,
    all_g1: bool = False,
    use_reference_motion: bool = False,
    ref_state: ViewerState | None = None,
    layer_height: float = 0.8,
    pixel_size: float = 0.8,
) -> tuple[str | None, str | None, str | None, str]:
    specs = [
        (1, zip1, pressure1, valve1, port1),
        (2, zip2, pressure2, valve2, port2),
        (3, zip3, pressure3, valve3, port3),
    ]

    # When reference-driven motion is requested, every shape's nozzle path comes
    # from the combined Reference TIFF Stack; the valve still follows each shape.
    motion_tiffs: list[str] | None = None
    if use_reference_motion:
        motion_tiffs = (ref_state or {}).get("tiff_paths") or None

    outputs: list[str | None] = [None, None, None]
    messages: list[str] = []

    for idx, zip_path, pressure, valve, port in specs:
        if not zip_path:
            messages.append(f"Shape {idx}: skipped (no TIFF ZIP available).")
            continue

        if use_reference_motion and not motion_tiffs:
            messages.append(
                f"Shape {idx}: skipped (Reference motion selected, but no Reference "
                f"TIFF Stack has been generated on the first tab)."
            )
            continue

        zip_name = Path(zip_path).stem
        default_shape_name = f"shape{idx}"
        shape_name = zip_name.replace("_tiff_slices", "") or default_shape_name

        try:
            gcode_path = generate_snake_path_gcode(
                zip_path=zip_path,
                shape_name=shape_name,
                pressure=float(pressure),
                valve=int(valve),
                port=int(port),
                layer_height=float(layer_height),
                fil_width=float(pixel_size),
                all_g1=bool(all_g1),
                motion_tiffs=motion_tiffs,
            )
            outputs[idx - 1] = str(gcode_path)
            messages.append(f"Shape {idx}: wrote `{gcode_path.name}`.")
        except Exception as exc:  # surface errors in the UI
            outputs[idx - 1] = None
            messages.append(f"Shape {idx}: failed ({exc}).")

    return outputs[0], outputs[1], outputs[2], "\n".join(messages)


def load_all_gcode_text(
    path1: str | None,
    path2: str | None,
    path3: str | None,
) -> tuple[str, str, str]:
    """Return the raw text of each shape's generated G-code file for display."""
    def read(path: str | None, shape_num: int) -> str:
        if not path:
            return f"# No G-code generated for Shape {shape_num} yet."
        try:
            return Path(path).read_text()
        except OSError as exc:
            return f"# Failed to read G-code file: {exc}"

    return read(path1, 1), read(path2, 2), read(path3, 3)


GCODE_SOURCE_SHAPE1 = "Use Shape 1 G-Code"
GCODE_SOURCE_SHAPE2 = "Use Shape 2 G-Code"
GCODE_SOURCE_SHAPE3 = "Use Shape 3 G-Code"
GCODE_SOURCE_UPLOAD = "Upload G-Code file"


def render_toolpath(
    source: str,
    uploaded_path: str | None,
    shape1_path: str | None,
    shape2_path: str | None,
    shape3_path: str | None,
    travel_opacity: float = 0.2,
    print_opacity: float = 1.0,
    travel_color: str = "#969696",
    print_color: str = "#ff7f0e",
    print_width: float = 0.8,
    travel_width: float = 0.2,
    tube: bool = True,
) -> tuple[Any, str, dict]:
    if source == GCODE_SOURCE_UPLOAD:
        path = uploaded_path
        if not path:
            return None, "No G-code file uploaded yet.", {}
    else:
        shape_paths = {
            GCODE_SOURCE_SHAPE1: (shape1_path, 1),
            GCODE_SOURCE_SHAPE2: (shape2_path, 2),
            GCODE_SOURCE_SHAPE3: (shape3_path, 3),
        }
        path, shape_num = shape_paths.get(source, (shape1_path, 1))
        if not path:
            return None, f"No Shape {shape_num} G-code available yet. Generate it on the TIFF Slices to GCode tab first.", {}

    try:
        text = Path(path).read_text()
    except OSError as exc:
        return None, f"Failed to read G-code file: {exc}", {}

    parsed = parse_gcode_path(text)
    if parsed["point_count"] == 0:
        return None, "No G0/G1 movement lines found in the file.", {}

    figure = build_toolpath_figure(parsed, travel_opacity=travel_opacity, print_opacity=print_opacity, travel_color=travel_color, print_color=print_color, print_width=print_width, travel_width=travel_width, tube=tube)
    (x_min, y_min, z_min), (x_max, y_max, z_max) = parsed["bounds"]
    summary = (
        f"**{parsed['point_count']} moves parsed** — "
        f"{len(parsed['print_segments'])} print segment(s), "
        f"{len(parsed['travel_segments'])} travel segment(s).  \n"
        f"Bounds: X ∈ [{x_min:.2f}, {x_max:.2f}], "
        f"Y ∈ [{y_min:.2f}, {y_max:.2f}], "
        f"Z ∈ [{z_min:.2f}, {z_max:.2f}] mm."
    )
    return figure, summary, parsed


def render_toolpath_lines(
    source: str,
    uploaded_path: str | None,
    shape1_path: str | None,
    shape2_path: str | None,
    shape3_path: str | None,
    travel_opacity: float,
    print_opacity: float,
    travel_color: str,
    print_color: str,
    print_width: float,
    travel_width: float,
) -> tuple[Any, str, dict, str, dict[str, Any], dict[str, Any]]:
    figure, status, parsed = render_toolpath(
        source, uploaded_path, shape1_path, shape2_path, shape3_path,
        travel_opacity, print_opacity, travel_color, print_color,
        print_width, travel_width, tube=False,
    )
    return figure, status, parsed, "line", gr.update(visible=False), gr.update(visible=False)


def render_toolpath_tubes(
    source: str,
    uploaded_path: str | None,
    shape1_path: str | None,
    shape2_path: str | None,
    shape3_path: str | None,
    travel_opacity: float,
    print_opacity: float,
    travel_color: str,
    print_color: str,
    print_width: float,
    travel_width: float,
) -> tuple[Any, str, dict, str, dict[str, Any], dict[str, Any]]:
    figure, status, parsed = render_toolpath(
        source, uploaded_path, shape1_path, shape2_path, shape3_path,
        travel_opacity, print_opacity, travel_color, print_color,
        print_width, travel_width, tube=True,
    )
    # Playback controls and mm width sliders only apply to the tube figure.
    has_animation = bool(parsed.get("point_count"))
    return figure, status, parsed, "tube", gr.update(visible=has_animation), gr.update(visible=has_animation)


def rerender_toolpath_current_mode(
    mode: str,
    source: str,
    uploaded_path: str | None,
    shape1_path: str | None,
    shape2_path: str | None,
    shape3_path: str | None,
    travel_opacity: float,
    print_opacity: float,
    travel_color: str,
    print_color: str,
    print_width: float,
    travel_width: float,
) -> tuple[Any, str, dict]:
    return render_toolpath(
        source, uploaded_path, shape1_path, shape2_path, shape3_path,
        travel_opacity, print_opacity, travel_color, print_color,
        print_width, travel_width, tube=(mode != "line"),
    )


PARALLEL_COLOR_CHOICES = [
    ("Orange", "#ff7f0e"), ("Blue", "#1f77b4"), ("Green", "#2ca02c"),
    ("Red", "#d62728"), ("Purple", "#9467bd"), ("Pink", "#e377c2"),
    ("Teal", "#17becf"), ("Black", "#000000"),
]


def render_parallel(
    path1: str | None,
    path2: str | None,
    path3: str | None,
    color1: str,
    color2: str,
    color3: str,
    travel_opacity: float,
    filament_width: float,
    travel_width: float,
    gap: float,
    tube: bool = True,
) -> tuple[Any, str]:
    specs = [(1, path1, color1), (2, path2, color2), (3, path3, color3)]
    parts: list[dict] = []
    messages: list[str] = []

    for idx, path, color in specs:
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
        parts.append({"idx": idx, "color": color, "parsed": parsed})
        messages.append(f"Shape {idx}: {parsed['point_count']} moves, {parsed.get('layer_count', 0)} layer(s).")

    if not parts:
        return None, "No shape G-code available. Generate G-code on the TIFF Slices to GCode tab first."

    figure = build_parallel_figure(
        parts,
        gap=float(gap),
        filament_width=float(filament_width),
        travel_width=float(travel_width),
        travel_opacity=float(travel_opacity),
        tube=tube,
    )
    return figure, "  \n".join(messages)


def render_parallel_lines(
    *args: Any,
) -> tuple[Any, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    figure, status = render_parallel(*args, tube=False)
    # Line mode: no playback animation and no mm width sliders, but the GIF
    # export (server-side, line-style) is available whenever there's data.
    has_data = figure is not None
    return (
        figure, status, "line",
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=has_data),
    )


def render_parallel_tubes(
    *args: Any,
) -> tuple[Any, str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    figure, status = render_parallel(*args, tube=True)
    has_anim = figure is not None
    return (
        figure, status, "tube",
        gr.update(visible=has_anim), gr.update(visible=has_anim), gr.update(visible=has_anim),
    )


def rerender_parallel_current_mode(mode: str, *args: Any) -> tuple[Any, str]:
    return render_parallel(*args, tube=(mode != "line"))


def export_parallel_gif(
    path1: str | None,
    path2: str | None,
    path3: str | None,
    color1: str,
    color2: str,
    color3: str,
    travel_opacity: float,
    gap: float,
    duration: float,
    fps: float,
    elev: float,
    azim: float,
    progress: gr.Progress = gr.Progress(),
) -> str | None:
    """Render the parallel print to an animated GIF server-side via Matplotlib.

    CPU-only (Agg backend) — no WebGL or headless browser — so it works locally
    and on Hugging Face. Each shape grows as colored lines on a shared timeline.
    """
    specs = [(path1, color1), (path2, color2), (path3, color3)]
    parts: list[dict] = []
    for idx, (path, color) in enumerate(specs, start=1):
        if not path:
            continue
        try:
            parsed = parse_gcode_path(Path(path).read_text())
        except OSError:
            continue
        if parsed.get("point_count"):
            parts.append({"idx": idx, "color": color, "parsed": parsed})

    if not parts:
        return None

    def report(frame: int, total: int) -> None:
        progress(frame / total, desc=f"Rendering frame {frame + 1}/{total}")

    out_path = Path(tempfile.mkdtemp(prefix="parallel_gif_")) / "parallel_print.gif"
    result = build_parallel_gif(
        parts,
        out_path=out_path,
        gap=float(gap),
        duration=float(duration),
        fps=int(fps),
        travel_opacity=float(travel_opacity),
        elev=float(elev),
        azim=float(azim),
        progress_cb=report,
    )
    return str(result) if result else None


def update_toolpath_opacity(
    parsed: dict,
    travel_opacity: float,
    print_opacity: float,
) -> Any:
    if not parsed or not parsed.get("point_count"):
        return None
    return build_toolpath_figure(parsed, travel_opacity=travel_opacity, print_opacity=print_opacity)


def shift_slice(state: ViewerState, index: float, delta: int) -> tuple[int, str, Image.Image | None]:
    tiff_paths = state.get("tiff_paths", [])
    if not tiff_paths:
        return 0, "No slice stack loaded yet.", None

    new_index = max(0, min(int(index) + delta, len(tiff_paths) - 1))
    label, preview = _render_selected_slice(state, new_index)
    return new_index, label, preview


def generate_reference_stack(
    state1: ViewerState,
    state2: ViewerState,
    state3: ViewerState,
    progress: gr.Progress = gr.Progress(),
) -> tuple:
    """Combine all available TIFF stacks into a single reference stack.

    For each pixel in each layer the result is black (0) when *any* source
    stack has a black pixel at that position, and white (255) only when *all*
    sources are white.  Images of different sizes are centred on a canvas
    sized to the largest dimensions.
    """
    active_states = [s for s in [state1, state2, state3] if s.get("tiff_paths")]

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

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="STL TIFF Slicer", css=APP_CSS, head=APP_HEAD + TOOLPATH_ANIM_HEAD + PARALLEL_ANIM_HEAD) as demo:
        with gr.Tab("STL to TIFF Slicer"):
            gr.Markdown(
                """
                # STL to TIFF Slicer
                Upload up to three STL files, choose per-shape STL dimensions, layer height, and XY pixel size, then generate TIFF stacks for all uploaded models.
                """
            )

            with gr.Row():
                load_samples_button = gr.Button(
                    "Load Sample STLs",
                    variant="secondary",
                    size="sm",
                    min_width=140,
                    scale=0,
                    elem_id="load-sample-stls-button",
                )
                with gr.Column(scale=0, min_width=240):
                    model_opacity = gr.Checkbox(
                        label="Use 75% 3D Model Opacity",
                        value=False,
                    )
                with gr.Column(scale=0, min_width=260):
                    scale_to_target = gr.Checkbox(
                        label="Apply STL Scaling",
                        value=False,
                    )
                with gr.Column(scale=0, min_width=260):
                    scale_mode = gr.Radio(
                        choices=[SCALE_MODE_TARGET_DIMENSIONS, SCALE_MODE_UNIFORM_FACTOR],
                        value=SCALE_MODE_TARGET_DIMENSIONS,
                        label="Scaling Mode",
                    )

            # --- Upload + 3D viewer row ---
            stl_files: list[gr.File] = []
            model_viewers: list[gr.Model3D] = []
            model_details_list: list[gr.Markdown] = []
            target_xs: list[gr.Number] = []
            target_ys: list[gr.Number] = []
            target_zs: list[gr.Number] = []
            uniform_scales: list[gr.Number] = []

            with gr.Row():
                for i in range(3):
                    with gr.Column(min_width=250):
                        stl_file = gr.File(
                            label=f"STL File {i + 1}",
                            file_types=[".stl"],
                            type="filepath",
                        )
                        model_viewer = gr.Model3D(
                            label=f"3D Viewer {i + 1}",
                            display_mode="solid",
                            clear_color=(0.94, 0.95, 0.97, 1.0),
                            camera_position=FRONT_CAMERA,
                            height=270,
                        )
                        model_details = gr.Markdown(f"No model {i + 1} loaded.")
                        with gr.Row():
                            target_x = gr.Number(
                                label="Target X (mm)",
                                value=DEFAULT_TARGET_EXTENTS[0],
                                minimum=0.0001,
                                step=0.1,
                            )
                            target_y = gr.Number(
                                label="Target Y (mm)",
                                value=DEFAULT_TARGET_EXTENTS[1],
                                minimum=0.0001,
                                step=0.1,
                            )
                            target_z = gr.Number(
                                label="Target Z (mm)",
                                value=DEFAULT_TARGET_EXTENTS[2],
                                minimum=0.0001,
                                step=0.1,
                            )
                        uniform_scale = gr.Number(
                            label="Uniform Scale",
                            value=DEFAULT_UNIFORM_SCALE,
                            minimum=0.0001,
                            step=0.01,
                        )
                        stl_files.append(stl_file)
                        model_viewers.append(model_viewer)
                        model_details_list.append(model_details)
                        target_xs.append(target_x)
                        target_ys.append(target_y)
                        target_zs.append(target_z)
                        uniform_scales.append(uniform_scale)

            # --- Shared slicing controls ---
            with gr.Row():
                layer_height = gr.Number(label="Layer Height", value=0.8, minimum=0.0001, step=0.01)
                pixel_size = gr.Number(
                    label="Pixel Size/Fill Width",
                    value=0.8,
                    minimum=0.0001,
                    step=0.01,
                )
                generate_button = gr.Button("Generate TIFF Stacks", variant="primary")

            # --- Per-object slice browsers ---
            states: list[gr.State] = []
            sliders: list[gr.Slider] = []
            slice_labels: list[gr.Markdown] = []
            slice_previews: list[gr.Image] = []
            download_zips: list[gr.File] = []

            with gr.Row():
                for i in range(3):
                    with gr.Column(min_width=250):
                        slice_label = gr.Markdown("No slice stack loaded yet.")
                        slice_preview = gr.Image(
                            label=f"Slice Preview {i + 1}",
                            type="pil",
                            image_mode="RGB",
                            height=270,
                        )
                        with gr.Row():
                            prev_button = gr.Button("\u25c4 Prev", scale=1, min_width=90, size="sm")
                            next_button = gr.Button("Next \u25ba", scale=1, min_width=90, size="sm")
                        slice_slider = gr.Slider(
                            label="Slice Index",
                            minimum=0,
                            maximum=0,
                            value=0,
                            step=1,
                            interactive=False,
                        )
                        download_zip = gr.File(label=f"Download TIFF ZIP {i + 1}", interactive=False)
                        state = gr.State(_empty_state())

                        slice_labels.append(slice_label)
                        slice_previews.append(slice_preview)
                        sliders.append(slice_slider)
                        download_zips.append(download_zip)
                        states.append(state)

                        slice_slider.release(
                            fn=jump_to_slice,
                            inputs=[state, slice_slider],
                            outputs=[slice_label, slice_preview],
                            queue=False,
                        )
                        prev_button.click(
                            fn=lambda sv, idx: shift_slice(sv, idx, -1),
                            inputs=[state, slice_slider],
                            outputs=[slice_slider, slice_label, slice_preview],
                            queue=False,
                        )
                        next_button.click(
                            fn=lambda sv, idx: shift_slice(sv, idx, 1),
                            inputs=[state, slice_slider],
                            outputs=[slice_slider, slice_label, slice_preview],
                            queue=False,
                        )

            # --- Reference TIFF Stack ---
            gr.Markdown("---")
            gr.Markdown("### Reference TIFF Stack")

            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    ref_generate_button = gr.Button(
                        "Generate Reference TIFF Stack",
                        variant="primary",
                    )
                with gr.Column(scale=3, min_width=250):
                    ref_slice_label = gr.Markdown("No reference stack generated yet.")
                    ref_slice_preview = gr.Image(
                        label="Reference Slice Preview",
                        type="pil",
                        image_mode="RGB",
                        height=270,
                    )
                    with gr.Row():
                        ref_prev_button = gr.Button("\u25c4 Prev", scale=1, min_width=90, size="sm")
                        ref_next_button = gr.Button("Next \u25ba", scale=1, min_width=90, size="sm")
                    ref_slice_slider = gr.Slider(
                        label="Slice Index",
                        minimum=0,
                        maximum=0,
                        value=0,
                        step=1,
                        interactive=False,
                    )
            ref_state = gr.State(_empty_state())

            ref_slice_slider.release(
                fn=jump_to_slice,
                inputs=[ref_state, ref_slice_slider],
                outputs=[ref_slice_label, ref_slice_preview],
                queue=False,
            )
            ref_prev_button.click(
                fn=lambda sv, idx: shift_slice(sv, idx, -1),
                inputs=[ref_state, ref_slice_slider],
                outputs=[ref_slice_slider, ref_slice_label, ref_slice_preview],
                queue=False,
            )
            ref_next_button.click(
                fn=lambda sv, idx: shift_slice(sv, idx, 1),
                inputs=[ref_state, ref_slice_slider],
                outputs=[ref_slice_slider, ref_slice_label, ref_slice_preview],
                queue=False,
            )

            # --- File upload handlers ---
            all_scale_inputs = [
                target_xs[0],
                target_ys[0],
                target_zs[0],
                uniform_scales[0],
                target_xs[1],
                target_ys[1],
                target_zs[1],
                uniform_scales[1],
                target_xs[2],
                target_ys[2],
                target_zs[2],
                uniform_scales[2],
            ]

            for i in range(3):
                stl_files[i].change(
                    fn=load_single_model,
                    inputs=[
                        stl_files[i],
                        model_opacity,
                        scale_to_target,
                        scale_mode,
                        target_xs[i],
                        target_ys[i],
                        target_zs[i],
                        uniform_scales[i],
                    ],
                    outputs=[model_viewers[i], model_details_list[i]],
                )

            # --- Generate button ---
            generate_outputs: list = []
            for i in range(3):
                generate_outputs.extend([
                    states[i],
                    sliders[i],
                    slice_labels[i],
                    slice_previews[i],
                    download_zips[i],
                ])

            preload_outputs: list = []
            for i in range(3):
                preload_outputs.extend([
                    stl_files[i],
                    model_viewers[i],
                    model_details_list[i],
                ])

            load_samples_button.click(
                fn=preload_sample_models,
                inputs=[model_opacity, scale_to_target, scale_mode, *all_scale_inputs],
                outputs=preload_outputs,
            )

            refresh_outputs: list = []
            for i in range(3):
                refresh_outputs.extend([model_viewers[i], model_details_list[i]])

            refresh_inputs = [
                stl_files[0],
                stl_files[1],
                stl_files[2],
                model_opacity,
                scale_to_target,
                scale_mode,
                *all_scale_inputs,
            ]

            model_opacity.change(
                fn=refresh_all_model_viewers,
                inputs=refresh_inputs,
                outputs=refresh_outputs,
            )
            for scale_control in (scale_to_target, scale_mode, *all_scale_inputs):
                scale_control.change(
                    fn=refresh_all_model_viewers,
                    inputs=refresh_inputs,
                    outputs=refresh_outputs,
                )

            generate_button.click(
                fn=generate_all_stacks,
                inputs=[
                    stl_files[0],
                    stl_files[1],
                    stl_files[2],
                    layer_height,
                    pixel_size,
                    scale_to_target,
                    scale_mode,
                    *all_scale_inputs,
                ],
                outputs=generate_outputs,
            )

            ref_generate_button.click(
                fn=generate_reference_stack,
                inputs=[states[0], states[1], states[2]],
                outputs=[ref_state, ref_slice_slider, ref_slice_label, ref_slice_preview],
            )

        with gr.Tab("TIFF Slices to GCode"):
            gr.Markdown(
                """
                # TIFF Slices to GCode
                Uses TIFF ZIP outputs from the first tab. Set pressure, valve,
                and port for each shape, then generate G-code files in one run.
                """
            )

            with gr.Row():
                with gr.Column(min_width=250):
                    with gr.Group(elem_classes=["gcode-shape-card"]):
                        gr.Markdown("### Shape 1")
                        with gr.Row():
                            with gr.Column(min_width=70):
                                gr.Markdown("Pressure (psi)", elem_classes=["gcode-param-label"])
                                gcode_pressure_1 = gr.Number(
                                    show_label=False,
                                    value=25.0,
                                    minimum=0.0,
                                    step=0.5,
                                )
                            with gr.Column(min_width=70):
                                gr.Markdown("Valve", elem_classes=["gcode-param-label"])
                                gcode_valve_1 = gr.Number(
                                    show_label=False,
                                    value=4,
                                    minimum=0,
                                    step=1,
                                    precision=0,
                                )
                            with gr.Column(min_width=70):
                                gr.Markdown("Port", elem_classes=["gcode-param-label"])
                                gcode_port_1 = gr.Number(
                                    show_label=False,
                                    value=1,
                                    minimum=1,
                                    step=1,
                                    precision=0,
                                )
                with gr.Column(min_width=250):
                    with gr.Group(elem_classes=["gcode-shape-card"]):
                        gr.Markdown("### Shape 2")
                        with gr.Row():
                            with gr.Column(min_width=70):
                                gr.Markdown("Pressure (psi)", elem_classes=["gcode-param-label"])
                                gcode_pressure_2 = gr.Number(
                                    show_label=False,
                                    value=25.0,
                                    minimum=0.0,
                                    step=0.5,
                                )
                            with gr.Column(min_width=70):
                                gr.Markdown("Valve", elem_classes=["gcode-param-label"])
                                gcode_valve_2 = gr.Number(
                                    show_label=False,
                                    value=4,
                                    minimum=0,
                                    step=1,
                                    precision=0,
                                )
                            with gr.Column(min_width=70):
                                gr.Markdown("Port", elem_classes=["gcode-param-label"])
                                gcode_port_2 = gr.Number(
                                    show_label=False,
                                    value=1,
                                    minimum=1,
                                    step=1,
                                    precision=0,
                                )
                with gr.Column(min_width=250):
                    with gr.Group(elem_classes=["gcode-shape-card"]):
                        gr.Markdown("### Shape 3")
                        with gr.Row():
                            with gr.Column(min_width=70):
                                gr.Markdown("Pressure (psi)", elem_classes=["gcode-param-label"])
                                gcode_pressure_3 = gr.Number(
                                    show_label=False,
                                    value=25.0,
                                    minimum=0.0,
                                    step=0.5,
                                )
                            with gr.Column(min_width=70):
                                gr.Markdown("Valve", elem_classes=["gcode-param-label"])
                                gcode_valve_3 = gr.Number(
                                    show_label=False,
                                    value=4,
                                    minimum=0,
                                    step=1,
                                    precision=0,
                                )
                            with gr.Column(min_width=70):
                                gr.Markdown("Port", elem_classes=["gcode-param-label"])
                                gcode_port_3 = gr.Number(
                                    show_label=False,
                                    value=1,
                                    minimum=1,
                                    step=1,
                                    precision=0,
                                )

            gcode_use_ref_motion = gr.Checkbox(
                label="Use Reference Stack for motion (all shapes share one nozzle path; each dispenses only its own geometry). Generate the Reference TIFF Stack on the first tab first.",
                value=True,
            )
            gcode_all_g1 = gr.Checkbox(
                label="Move at one constant speed (no fast travel moves)",
                info=(
                    "Every move — including repositioning between deposits — uses the "
                    "slower printing-speed command (G1) instead of a rapid travel command "
                    "(G0). The valve still controls where material is actually dispensed. "
                    "Applies to all shapes."
                ),
                value=True,
            )
            gcode_button = gr.Button("Generate G-Code", variant="primary")

            with gr.Row():
                gcode_file_1 = gr.File(label="Download G-Code Shape 1", interactive=False, elem_classes=["gcode-download"])
                gcode_file_2 = gr.File(label="Download G-Code Shape 2", interactive=False, elem_classes=["gcode-download"])
                gcode_file_3 = gr.File(label="Download G-Code Shape 3", interactive=False, elem_classes=["gcode-download"])

            # Per-shape G-code preview, aligned under each download box. Fixed
            # height with internal scrolling so the files don't fill the page.
            with gr.Row():
                gcode_view_1 = gr.Code(
                    label="Shape 1 G-Code", language=None, lines=15, max_lines=15,
                    interactive=False, elem_classes=["gcode-view"], elem_id="gcode-view-1",
                )
                gcode_view_2 = gr.Code(
                    label="Shape 2 G-Code", language=None, lines=15, max_lines=15,
                    interactive=False, elem_classes=["gcode-view"], elem_id="gcode-view-2",
                )
                gcode_view_3 = gr.Code(
                    label="Shape 3 G-Code", language=None, lines=15, max_lines=15,
                    interactive=False, elem_classes=["gcode-view"], elem_id="gcode-view-3",
                )

            gcode_status = gr.Markdown("")

            gcode_button.click(
                fn=run_all_tiff_to_gcode,
                inputs=[
                    download_zips[0],
                    download_zips[1],
                    download_zips[2],
                    gcode_pressure_1,
                    gcode_valve_1,
                    gcode_port_1,
                    gcode_pressure_2,
                    gcode_valve_2,
                    gcode_port_2,
                    gcode_pressure_3,
                    gcode_valve_3,
                    gcode_port_3,
                    gcode_all_g1,
                    gcode_use_ref_motion,
                    ref_state,
                    layer_height,
                    pixel_size,
                ],
                outputs=[gcode_file_1, gcode_file_2, gcode_file_3, gcode_status],
            ).then(
                fn=load_all_gcode_text,
                inputs=[gcode_file_1, gcode_file_2, gcode_file_3],
                outputs=[gcode_view_1, gcode_view_2, gcode_view_3],
            ).then(
                # gr.Code's built-in download button hardcodes "file.txt"; stamp
                # the real filename (from each File value's orig_name) onto the
                # download link so it matches the Download G-Code Shape box.
                fn=None,
                inputs=[gcode_file_1, gcode_file_2, gcode_file_3],
                outputs=[],
                js="""(f1, f2, f3) => {
                    const files = [f1, f2, f3];
                    const nameOf = (f) => {
                        if (!f) return null;
                        if (f.orig_name) return f.orig_name;
                        const p = f.path || f.url || "";
                        return p ? p.split(/[\\\\/]/).pop() : null;
                    };
                    const apply = (tries) => {
                        let pending = false;
                        files.forEach((f, i) => {
                            const name = nameOf(f);
                            if (!name) return;
                            const a = document.querySelector(`#gcode-view-${i + 1} a[download]`);
                            if (a) {
                                if (a.getAttribute("download") !== name) a.setAttribute("download", name);
                            } else {
                                pending = true;
                            }
                        });
                        if (pending && tries < 20) setTimeout(() => apply(tries + 1), 150);
                    };
                    apply(0);
                    return [];
                }""",
            )

        with gr.Tab("G-Code Visualization"):
            gr.Markdown(
                "### 3D Tool-Path Viewer\n"
                "Choose a G-code source, then click **Render Tool Path** to visualize the nozzle path."
            )
            # --- G-code source selection, above the controls/chart area ---
            with gr.Row():
                gcode_source = gr.Radio(
                    choices=[
                        GCODE_SOURCE_SHAPE1,
                        GCODE_SOURCE_SHAPE2,
                        GCODE_SOURCE_SHAPE3,
                        GCODE_SOURCE_UPLOAD,
                    ],
                    value=GCODE_SOURCE_SHAPE1,
                    label="G-Code source",
                )
                # Shown only when "Upload G-Code file" is selected. Visibility is
                # toggled client-side (see gcode_source.change below) instead of
                # via a server round-trip, which raced and intermittently left
                # the box hidden. Hidden initially by CSS (#gcode-upload-col).
                with gr.Column(elem_id="gcode-upload-col") as upload_col:
                    gcode_upload = gr.File(
                        label="Upload G-Code",
                        file_types=[".txt", ".gcode", ".nc"],
                        interactive=True,
                        height=110,
                    )

            with gr.Row():
                # --- Left column: render buttons and plot controls ---
                with gr.Column(scale=1, min_width=340):
                    render_line_button = gr.Button(
                        "Render Tool Path - Line Plot", variant="primary"
                    )
                    render_tube_button = gr.Button(
                        "Render Tool Path - Tube Plot with Animation", variant="primary"
                    )
                    gr.Markdown(
                        "&#9888;&#65039; For high-resolution models (small layer "
                        "heights), the tube plot can take a while to build and render.",
                        elem_id="tube-render-warning",
                    )
                    anim_controls = gr.HTML(TOOLPATH_CONTROLS_HTML, visible=False)
                    with gr.Row():
                        travel_opacity_slider = gr.Slider(
                            label="Travel (G0) opacity",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.05,
                            min_width=150,
                        )
                        print_opacity_slider = gr.Slider(
                            label="Print (G1) opacity",
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            min_width=150,
                        )
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
                        travel_width_slider = gr.Slider(
                            label="Travel width (mm)",
                            minimum=0.05,
                            maximum=1.2,
                            value=0.2,
                            step=0.05,
                            min_width=150,
                        )
                        print_width_slider = gr.Slider(
                            label="Filament width (mm)",
                            minimum=0.1,
                            maximum=1.2,
                            value=0.8,
                            step=0.05,
                            min_width=150,
                        )
                    toolpath_status = gr.Markdown("")

                # --- Right column: the chart ---
                with gr.Column(scale=3, min_width=500):
                    toolpath_plot = gr.Plot(label="Tool Path", elem_id="toolpath_plot")

            parsed_state = gr.State({})
            render_mode = gr.State("tube")

            gcode_source.change(
                fn=None,
                inputs=[gcode_source],
                outputs=[],
                js="""(src) => {
                    const col = document.getElementById('gcode-upload-col');
                    if (col) col.style.display = (src === '"""
                + GCODE_SOURCE_UPLOAD
                + """') ? 'flex' : 'none';
                    return [];
                }""",
            )
            render_inputs = [gcode_source, gcode_upload, gcode_file_1, gcode_file_2, gcode_file_3, travel_opacity_slider, print_opacity_slider, travel_color_picker, print_color_picker, print_width_slider, travel_width_slider]
            render_line_button.click(
                fn=render_toolpath_lines,
                inputs=render_inputs,
                outputs=[toolpath_plot, toolpath_status, parsed_state, render_mode, anim_controls, width_row],
            )
            render_tube_button.click(
                fn=render_toolpath_tubes,
                inputs=render_inputs,
                outputs=[toolpath_plot, toolpath_status, parsed_state, render_mode, anim_controls, width_row],
            )
            # Changing the travel width rebuilds the figure in the last-used mode.
            travel_width_slider.release(
                fn=rerender_toolpath_current_mode,
                inputs=[render_mode] + render_inputs,
                outputs=[toolpath_plot, toolpath_status, parsed_state],
            )
            # Changing the filament width rebuilds the figure in the last-used mode.
            print_width_slider.release(
                fn=rerender_toolpath_current_mode,
                inputs=[render_mode] + render_inputs,
                outputs=[toolpath_plot, toolpath_status, parsed_state],
            )
            # Keep the filament and travel width sliders in sync with the Layer
            # Height chosen on the slicing tab: filament width equals the layer
            # height, travel width is a quarter of it, both capped 50% above the
            # layer height.
            def sync_width_sliders(v: float):
                height = float(v or 0.8)
                travel = height / 4
                print_update = gr.update(
                    value=height, minimum=min(0.1, height), maximum=height * 1.5
                )
                travel_update = gr.update(
                    value=travel, minimum=min(0.05, travel), maximum=height * 1.5
                )
                return print_update, travel_update

            layer_height.change(
                fn=sync_width_sliders,
                inputs=[layer_height],
                outputs=[print_width_slider, travel_width_slider],
                queue=False,
            )
            travel_opacity_slider.release(
                fn=None,
                inputs=[travel_opacity_slider],
                outputs=[],
                js="""(opacity_val) => {
                    const container = document.getElementById("toolpath_plot");
                    if (!container) return [];
                    const plotDiv = container.querySelector(".js-plotly-plot");
                    if (!plotDiv || !plotDiv.data) return [];
                    const indices = plotDiv.data
                        .map((t, i) => t.name === "Travel (G0)" ? i : -1)
                        .filter(i => i >= 0);
                    if (indices.length > 0) Plotly.restyle(plotDiv, {opacity: opacity_val}, indices);
                    return [];
                }"""
            )
            print_opacity_slider.release(
                fn=None,
                inputs=[print_opacity_slider],
                outputs=[],
                js="""(opacity_val) => {
                    const container = document.getElementById("toolpath_plot");
                    if (!container) return [];
                    const plotDiv = container.querySelector(".js-plotly-plot");
                    if (!plotDiv || !plotDiv.data) return [];
                    const indices = plotDiv.data
                        .map((t, i) => t.name === "Print (G1)" ? i : -1)
                        .filter(i => i >= 0);
                    if (indices.length > 0) Plotly.restyle(plotDiv, {opacity: opacity_val}, indices);
                    return [];
                }"""
            )
            travel_color_picker.change(
                fn=None,
                inputs=[travel_color_picker],
                outputs=[],
                js="""(color) => {
                    const container = document.getElementById("toolpath_plot");
                    if (!container) return [];
                    const plotDiv = container.querySelector(".js-plotly-plot");
                    if (!plotDiv || !plotDiv.data) return [];
                    plotDiv.data.forEach((t, i) => {
                        if (t.name !== "Travel (G0)") return;
                        const attr = t.type === "mesh3d" ? {"color": color} : {"line.color": color};
                        Plotly.restyle(plotDiv, attr, [i]);
                    });
                    return [];
                }"""
            )
            print_color_picker.change(
                fn=None,
                inputs=[print_color_picker],
                outputs=[],
                js="""(color) => {
                    const container = document.getElementById("toolpath_plot");
                    if (!container) return [];
                    const plotDiv = container.querySelector(".js-plotly-plot");
                    if (!plotDiv || !plotDiv.data) return [];
                    plotDiv.data.forEach((t, i) => {
                        if (t.name !== "Print (G1)") return;
                        const attr = t.type === "mesh3d" ? {"color": color} : {"line.color": color};
                        Plotly.restyle(plotDiv, attr, [i]);
                    });
                    return [];
                }"""
            )

        with gr.Tab("Parallel Printing Visualization"):
            gr.Markdown(
                "### Parallel Printing Visualization\n"
                "Plots all three shapes side by side (offset in X) and animates "
                "them printing in parallel. Uses the G-code generated on the "
                "**TIFF Slices to GCode** tab."
            )
            with gr.Row():
                # --- Left column: controls ---
                with gr.Column(scale=1, min_width=340):
                    parallel_line_button = gr.Button(
                        "Render Parallel Print - Line Plot", variant="primary"
                    )
                    parallel_render_button = gr.Button(
                        "Render Parallel Print - Tube Plot with Animation", variant="primary"
                    )
                    gr.Markdown(
                        "&#9888;&#65039; Building three tube plots can take a while "
                        "for high-resolution models.",
                        elem_id="parallel-render-warning",
                    )
                    parallel_anim_controls = gr.HTML(PARALLEL_CONTROLS_HTML, visible=False)
                    with gr.Row():
                        pp_color_1 = gr.Dropdown(
                            label="Shape 1 color", choices=PARALLEL_COLOR_CHOICES,
                            value="#ff7f0e", allow_custom_value=False, min_width=120,
                        )
                        pp_color_2 = gr.Dropdown(
                            label="Shape 2 color", choices=PARALLEL_COLOR_CHOICES,
                            value="#1f77b4", allow_custom_value=False, min_width=120,
                        )
                        pp_color_3 = gr.Dropdown(
                            label="Shape 3 color", choices=PARALLEL_COLOR_CHOICES,
                            value="#2ca02c", allow_custom_value=False, min_width=120,
                        )
                    pp_travel_opacity = gr.Slider(
                        label="Travel opacity (0 = hidden)",
                        minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                    )
                    with gr.Row(visible=False) as pp_width_row:
                        pp_filament_width = gr.Slider(
                            label="Filament width (mm)", minimum=0.1, maximum=3.0,
                            value=0.8, step=0.05, min_width=150,
                        )
                        pp_travel_width = gr.Slider(
                            label="Travel width (mm)", minimum=0.05, maximum=3.0,
                            value=0.2, step=0.05, min_width=150,
                        )
                    pp_gap = gr.Slider(
                        label="Gap between parts (mm)",
                        minimum=0.0, maximum=50.0, value=5.0, step=0.5,
                    )
                    parallel_status = gr.Markdown("")

                    with gr.Group(visible=False) as pp_export_group:
                        gr.Markdown(
                            "**Export animation (GIF)** — a server-side line "
                            "animation of the parallel print. Set the viewing "
                            "angle below."
                        )
                        with gr.Row():
                            pp_gif_duration = gr.Slider(
                                label="Duration (s)", minimum=2.0, maximum=20.0,
                                value=6.0, step=1.0, min_width=150,
                            )
                            pp_gif_fps = gr.Slider(
                                label="Frames per second", minimum=5, maximum=30,
                                value=10, step=1, min_width=150,
                            )
                        with gr.Row():
                            pp_elev = gr.Slider(
                                label="Elevation angle", minimum=0, maximum=90,
                                value=22, step=1, min_width=150,
                            )
                            pp_azim = gr.Slider(
                                label="Azimuth angle", minimum=-180, maximum=180,
                                value=-60, step=5, min_width=150,
                            )
                        pp_gif_travel_opacity = gr.Slider(
                            label="Travel opacity in GIF (0 = hidden)",
                            minimum=0.0, maximum=1.0, value=0.15, step=0.05,
                        )
                        pp_export_button = gr.Button("Export Animation as GIF", variant="primary")
                        pp_gif_file = gr.File(label="Download GIF", interactive=False)
                # --- Right column: the plot ---
                with gr.Column(scale=3, min_width=500):
                    parallel_plot = gr.Plot(label="Parallel Tool Paths", elem_id="parallel_plot")

            parallel_mode = gr.State("tube")
            parallel_render_inputs = [
                gcode_file_1, gcode_file_2, gcode_file_3,
                pp_color_1, pp_color_2, pp_color_3,
                pp_travel_opacity, pp_filament_width, pp_travel_width, pp_gap,
            ]
            parallel_outputs = [
                parallel_plot, parallel_status, parallel_mode,
                parallel_anim_controls, pp_width_row, pp_export_group,
            ]
            parallel_line_button.click(
                fn=render_parallel_lines,
                inputs=parallel_render_inputs,
                outputs=parallel_outputs,
            )
            parallel_render_button.click(
                fn=render_parallel_tubes,
                inputs=parallel_render_inputs,
                outputs=parallel_outputs,
            )
            # Width and gap changes rebuild the figure in the last-used mode.
            for _slider in (pp_filament_width, pp_travel_width, pp_gap):
                _slider.release(
                    fn=rerender_parallel_current_mode,
                    inputs=[parallel_mode] + parallel_render_inputs,
                    outputs=[parallel_plot, parallel_status],
                )
            # Color changes recolor a part's print/travel/nozzle traces client-side.
            for _picker, _idx in ((pp_color_1, 1), (pp_color_2, 2), (pp_color_3, 3)):
                _picker.change(
                    fn=None,
                    inputs=[_picker],
                    outputs=[],
                    js="""(color) => {
                        const c = document.getElementById("parallel_plot");
                        if (!c) return [];
                        const pd = c.querySelector(".js-plotly-plot");
                        if (!pd || !pd.data) return [];
                        pd.data.forEach((t, i) => {
                            if (t.name === "Shape %d" || t.name === "Travel %d") {
                                const attr = t.type === "mesh3d" ? {"color": color} : {"line.color": color};
                                Plotly.restyle(pd, attr, [i]);
                            } else if (t.name === "Nozzle %d") {
                                Plotly.restyle(pd, {"marker.color": color}, [i]);
                            }
                        });
                        return [];
                    }""" % (_idx, _idx, _idx),
                )
            # Travel opacity changes apply to all travel meshes client-side.
            pp_travel_opacity.release(
                fn=None,
                inputs=[pp_travel_opacity],
                outputs=[],
                js="""(op) => {
                    const c = document.getElementById("parallel_plot");
                    if (!c) return [];
                    const pd = c.querySelector(".js-plotly-plot");
                    if (!pd || !pd.data) return [];
                    pd.data.forEach((t, i) => {
                        if (/^Travel \\d+$/.test(t.name)) Plotly.restyle(pd, {"opacity": op}, [i]);
                    });
                    return [];
                }""",
            )
            # Export: render the GIF server-side with Matplotlib (CPU/Agg) —
            # no WebGL or headless browser, so it works locally and on HF.
            pp_export_button.click(
                fn=export_parallel_gif,
                inputs=[
                    gcode_file_1, gcode_file_2, gcode_file_3,
                    pp_color_1, pp_color_2, pp_color_3,
                    pp_gif_travel_opacity, pp_gap, pp_gif_duration, pp_gif_fps,
                    pp_elev, pp_azim,
                ],
                outputs=[pp_gif_file],
            )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
