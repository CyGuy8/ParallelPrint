---
title: ParallelPrint
sdk: gradio
sdk_version: 6.10.0
python_version: "3.12"
app_file: app.py
fullWidth: true
short_description: Upload STLs, slice to vector outlines, and generate G-code.
---

# STL to G-Code Gradio App

This project provides a Gradio app that takes any number of uploaded STL files, shows a selected-shape 3D viewer, slices each model along the Z axis into per-layer vector outlines (shapely polygons), generates parallel-nozzle G-code directly from those outlines, previews the resulting tool path (a fast line plot or an animated 3D tube plot), and can visualize the shapes printing in parallel and export that animation as a GIF.

## Prerequisites

- Python 3.11 or newer for local development
- `uv` for dependency management and script execution
- Git LFS for the bundled `.stl` sample files

## Run

```powershell
uv sync --all-groups
uv run python app.py
```

For reload mode during development, run:

```powershell
uv run gradio app.py
```

When `app.py` changes, Gradio will automatically rerun the file and refresh the demo.

Then open the local Gradio URL in your browser, upload STL files or load the bundled samples, and slice the shapes.

## What the app does

- Uploads any number of `.stl` files with a single multi-file uploader
- Loads bundled sample STL files and merges them with already uploaded STLs
- Syncs the uploaded STL list back into Shape Settings if the table and uploader get out of step
- Shows an interactive selected-shape 3D viewer for rotating each model
- Shows model extents, face count, vertex count, and watertight status
- Scales loaded STLs from editable target X/Y/Z dimensions in the Shape Settings table; new rows default to the STL's original dimensions, **Reset Dimensions** restores them, and **Keep Proportions** updates the other target sides from the edited side
- Lets you choose layer height and filament/line width
- Slices each shape into per-layer vector outlines held in memory (no intermediate image files)
- Automatically unions the sliced shapes into a combined reference layer set whenever shapes are sliced
- Splits one sliced shape's geometry into an editable row/column grid for multi-nozzle printing of one large shape
- Converts sliced layers into G-code files with pressure, valve, nozzle, port, and infill % settings per shape from the Shape Settings table
- **Infill %** per shape skips dispensing on evenly-distributed raster lines (rings/revolutions for spiral patterns) — at 50% every other line prints — while the motion path stays exactly the same, so parallel shapes with different infill still share one print path
- Appends a shape outline contour after each enabled shape layer by tracing that layer's polygon boundary
- Offers G-code generation options for raster pattern, **Use G1 for all moves** (no rapid travel command), and **Use combined reference outline for motion** (all shapes share one nozzle path; each dispenses only its own geometry)
- Re-slices shapes automatically during G-code generation when their slices are missing or stale, so "upload, then Generate G-Code" works in one click
- Calculates X/Y nozzle spacing from a grid layout (columns/rows plus gaps), with an optional per-connection Advanced Grid Spacing table, then visualizes the resulting nozzle layout
- Previews selected generated G-code inline
- Visualizes generated or uploaded G-code tool paths, with the source selectable from any active generated shape or an uploaded file
- Renders the tool path as a fast line plot or an animated 3D tube plot (play/pause, speed, scrub, frame-step, nozzle marker)
- Plots the generated shapes using the configured nozzle spacing and animates them printing in parallel, with a server-side GIF export of that animation

## Behavior and Implementation Notes

### Vector Slicing

Slicing uses `trimesh` cross-sections composed into shapely polygons (`slice_stl_to_layers` in `stl_slicer.py`). Each layer is a `MultiPolygon` in world-XY millimetres; the whole shape is a `LayerStack` (layers, z-values, bounds, layer height) held in the Gradio session state — no files are written until G-code is generated.

### Reference Layer Union

When you click **Slice Shapes**, the app automatically unions the sliced shapes layer-by-layer into a combined reference layer set (used for shared-motion G-code).

- Shapes are aligned by centering each shape's XY bounding box on a common center before the union.
- Alignment is centered placement (in exact millimetres), not bottom-left anchoring.

### Multi-Nozzle Split

The **Multi-Nozzle Split** accordion on the **Shapes & Slicing** tab can split one sliced shape into a grid of print-ready piece stacks. Choose a source shape that has been sliced, set the number of columns and rows, choose the starting nozzle and valve numbers, then click **Split Selected Shape into Grid Pieces**.

- Each layer's geometry is clipped against equal-size grid cells, so every piece keeps exact vector outlines.
- The selected shape is replaced in Shape Settings by one generated record per grid cell, named by row and column.
- Nozzle and valve numbers are assigned sequentially from the starting values, and the **Generate G-Code** tab can generate separate G-code for each piece.
- **Overlapping Layers** alternates the interior cut lines by one filament width per layer so neighbouring pieces interlock.

### G-code XY Step Size

- G-code generation uses the slicer tab's `Filament/Line Width` as the raster line spacing by passing `fil_width` into `generate_vector_gcode()`.

### G-code Output

- Generated G-code starts in relative coordinate mode (`G91`).
- `G0` is travel and `G1` is print/feed.
- The app generates print/feed moves where the tool path is inside a shape's layer polygons and travel moves elsewhere.
- Generated files include pressure preset commands and WAGO valve commands based on the selected pressure, valve, and port; the nozzle number controls layout/spacing assignment.
- Pressure increases by `0.1` psi per layer by default.
- **Use G1 for all moves**: when enabled, every movement line is emitted as `G1` (no `G0` rapid travel); the WAGO valve still marks where material is dispensed. Applies to all shapes.
- **Use combined reference outline for motion**: when enabled, every shape's *motion* is taken from the combined reference layer union while each shape's *valve/dispensing* comes from its own layer polygons — so parallel print heads share one synchronized nozzle path and each deposits only its own geometry. The reference union is rebuilt automatically when shapes are sliced. Contour tracing stays synchronized too: every shape traces every traced shape's contour, opening its valve only on its own outline.
- Every generated file starts with a `; PathOrigin X.. Y..` comment: the world position (in the shape's own frame) that the relative toolpath starts from. Tools use it to place parallel parts so split pieces reassemble.
- **Raster Pattern**: `X-direction raster` sweeps every layer back-and-forth in X. `Y-direction raster` rasters every layer in Y. `Woodpile raster` alternates the raster axis by layer, switching between X-direction and Y-direction sweeps. `Rectangular Spiral raster` walks each layer from the outer layer bounds toward the center, then reverses from center to edge on the next layer. `Circle Spiral raster` uses a shrinking circular spiral from the layer bounds toward the center, then reverses outward on the next layer. Spiral motion covers the layer bounds; the valve opens only where the path is inside material.
- **Auto Align Split Parts**: in Nozzle Spacing, computes exact per-connection grid gaps from the split pieces' generated G-code (`PathOrigin` anchors + toolpath bounds), sets the grid columns/rows from the split, and fills the Advanced Grid Spacing table. Works for every raster pattern, filament width, reference-motion setting, and overlapping-layer split. Requires the pieces' G-code to be generated first.
- **Contour Tracing**: enabled per row in Shape Settings. The app traces the layer polygon's boundary rings (holes traced separately), travels from the layer raster end to the nearest contour point, prints the contour, then returns to the raster endpoint before the next layer.

### Print vs Travel Classification

When parsing G-code for visualization, the app decides print vs travel as follows:

- If the file contains `WAGO_ValveCommands`, the valve state (open/closed) determines print vs travel. This overrides `G0`/`G1`, because some generators emit every move as `G1`, or invert `G0`/`G1` relative to the valve.
- Otherwise it falls back to the convention `G1` = print, `G0` = travel.

The parser also handles standard slicer G-code: single-axis and Z-only moves, axes in any order, and `F`/`E` tokens (feed rate, extrusion) are ignored for geometry.

### G-code Visualization

The G-code visualization tab renders generated shape G-code or an uploaded `.txt`, `.gcode`, or `.nc` file. It parses `G0`/`G1` movement lines, supports relative (`G91`) and absolute (`G90`) positioning, and offers two render modes:

- **Line Plot** — fast thin scatter lines (print and travel), with color/opacity controls.
- **Tube Plot with Animation** — mm-width filament tubes (circular, capped, lit) with a client-side build animation (play/pause, speed, scrub, frame-step) and a moving nozzle marker. Filament/travel widths default to the layer height and its quarter.

### Parallel Printing Visualization

The fourth tab plots the generated shapes' G-code at once using the nozzle spacing configured on the Generate G-Code tab, each in its own color. Shape Settings maps each STL to a nozzle number, so multiple shapes can share one nozzle offset while valves remain independent. Like the visualization tab it has a fast **Line Plot** and an animated **Tube Plot**; the animation advances all parts on a shared cumulative-path-length timeline, so a shorter part finishes first.

It can also **export the animation as a GIF**, rendered server-side with Matplotlib (the `Agg` CPU backend — no WebGL, no headless browser, and no `ffmpeg`, so it works locally and on Hugging Face). The GIF is line-style with faint grey travel and white, black-outlined nozzle markers drawn on top; controls cover duration, frames per second, elevation/azimuth viewing angle, and travel opacity (0 hides travel).

## Dependency Updates

The parallel-print GIF export requires `matplotlib` (rendered with the CPU `Agg` backend so it runs on Hugging Face).

When dependencies change, update the lockfile and refresh the Hugging Face `requirements.txt` export — the Space installs from `requirements.txt`, not from the lockfile:

```powershell
uv sync --all-groups
uv export --format requirements.txt --no-hashes --no-dev --frozen --output-file requirements.txt
```

## Test

```powershell
uv run pytest
```

## Hugging Face Deployment

This repository tracks `.stl` files with Git LFS (see `.gitattributes`).

Before your first push on a machine:

```powershell
git lfs install
git lfs pull
```

Recommended push flow:

```powershell
git push origin main
git push hf-space main
```

If Hugging Face rejects a push for binary files, verify LFS setup first:

```powershell
git lfs version
git lfs ls-files
```

Warning: `git lfs migrate` rewrites commit history. Use it only when you intentionally want history rewritten and all collaborators are aligned.
