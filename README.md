---
title: ParallelPrint
sdk: gradio
sdk_version: 6.10.0
python_version: "3.12"
app_file: app.py
fullWidth: true
short_description: Upload STLs, export TIFF stacks, and generate G-code.
---

# STL to G-Code Gradio App

This project provides a Gradio app that takes any number of uploaded STL files, shows a selected-shape 3D viewer, slices each model along the Z axis, saves slices as TIFF images, generates G-code from those TIFF stacks, previews the resulting tool path (a fast line plot or an animated 3D tube plot), and can visualize the shapes printing in parallel and export that animation as a GIF.

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

Then open the local Gradio URL in your browser, upload STL files or load the bundled samples, and generate the TIFF stacks.

## What the app does

- Uploads any number of `.stl` files with a single multi-file uploader
- Loads bundled sample STL files and merges them with already uploaded STLs
- Syncs the uploaded STL list back into Shape Settings if the table and uploader get out of step
- Shows an interactive selected-shape 3D viewer for rotating each model
- Shows model extents, face count, vertex count, and watertight status
- Scales loaded STLs from editable target X/Y/Z dimensions in the Shape Settings table; new rows default to the STL's original dimensions, **Reset Dimensions** restores them, and **Keep Proportions** updates the other target sides from the edited side
- Keeps optional shape, TIFF, reference-stack, and nozzle-spacing previews in closed accordions to reduce clutter
- Lets you choose layer height and XY pixel size
- Produces one `.tif` image per slice
- Encodes material as black (`0`) and empty space as white (`255`) in each TIFF slice
- Lets you step through the slice stack in the browser
- Exports a ZIP containing the generated TIFF images
- Automatically combines generated stacks into a reference TIFF stack when TIFF stacks are generated
- Splits one generated TIFF stack into an editable row/column grid for multi-nozzle printing of one large shape
- Converts generated TIFF ZIPs into G-code files with pressure, valve, nozzle, and port settings per shape from the Shape Settings table
- Appends a shape-optimized outer contour after each enabled shape layer by tracing that layer's active row envelope
- Offers G-code generation options for raster pattern, **Use G1 for all moves** (no rapid travel command), and **Use Reference Stack for motion** (all shapes share one nozzle path; each dispenses only its own geometry)
- Calculates X/Y nozzle spacing from an editable adjacent-pair spacing table, then visualizes the resulting nozzle layout
- Previews selected generated G-code inline
- Visualizes generated or uploaded G-code tool paths, with the source selectable from any active generated shape or an uploaded file
- Renders the tool path as a fast line plot or an animated 3D tube plot (play/pause, speed, scrub, frame-step, nozzle marker)
- Plots the generated shapes using the configured nozzle spacing and animates them printing in parallel, with a server-side GIF export of that animation

## Behavior and Implementation Notes

### Reference TIFF Stack Alignment

When you click **Generate TIFF Stacks**, the app automatically combines available TIFF stacks layer-by-layer into the Reference TIFF Stack. The **Generate Reference TIFF Stack** button can still rebuild it manually from the current shape stacks.

- If source TIFFs have different dimensions, each layer is placed on a canvas using the largest width and height.
- Layers are centered in X and Y before merging.
- Pixel merge uses a black-wins rule: a pixel is black in the reference if any source has black at that pixel.
- Alignment is centered image placement, not bottom-left anchoring.
- If image-size differences are odd, centering may produce a one-pixel shift due to integer rounding.

### Multi-Nozzle Split

The **Multi-Nozzle Split** accordion on the **STL to TIFF Slicer** tab can split one generated shape stack into a grid of print-ready stacks. Choose a source shape that already has TIFF slices, set the number of columns and rows, choose the starting nozzle and valve numbers, then click **Split Selected Shape into Grid Pieces**.

- Each slice is padded with white pixels as needed, then split into equal-width columns and equal-height rows so every generated piece in the grid has a matching TIFF canvas.
- The selected shape is replaced in Shape Settings by one generated record per grid cell, named by row and column.
- Nozzle and valve numbers are assigned sequentially from the starting values, and the existing **TIFF Slices to GCode** tab can generate separate G-code for each piece.

### G-code XY Step Size

- G-code generation uses the slicer's `Pixel Size/Fill Width` for XY step distance by passing `fil_width=pixel_size` into `generate_snake_path_gcode()`.

### G-code Output

- Generated G-code starts in relative coordinate mode (`G91`).
- `G0` is travel and `G1` is print/feed.
- The app generates print/feed moves from material pixels and travel moves between material regions.
- Generated files include pressure preset commands and WAGO valve commands based on the selected pressure, valve, and port; the nozzle number controls layout/spacing assignment.
- Pressure increases by `0.1` psi per layer by default.
- **Use G1 for all moves**: when enabled, every movement line is emitted as `G1` (no `G0` rapid travel); the WAGO valve still marks where material is dispensed. Applies to all shapes.
- **Use Reference Stack for motion**: when enabled, every shape's snake-path *motion* is taken from the combined Reference TIFF Stack while each shape's *valve/dispensing* comes from its own slices — so parallel print heads share one synchronized nozzle path and each deposits only its own geometry. The reference stack is generated automatically with TIFF stacks; shapes are skipped with a message if it is missing.
- **Raster Pattern**: `X-direction raster` keeps the existing X-direction back-and-forth raster on every layer. `Y-direction raster` rasters every layer in Y. `Woodpile raster` alternates the raster axis by layer, switching between X-direction and Y-direction sweeps. `Rectangular Spiral raster` walks each layer from the outer layer bounds toward the center, then reverses from center to edge on the next layer. `Circle Spiral raster` uses a shrinking circular spiral from the layer bounds toward the center, then reverses outward on the next layer.
- **Auto Align Split Parts**: in Nozzle Spacing, fills Grid Layout gaps for split-piece alignment. X-direction raster uses X `-3.2` mm and Y `-0.8` mm; Y-direction raster switches those values.
- **Contour Tracing**: enabled per row in Shape Settings. The app uses the shape-optimized row-envelope tracer, travels from the layer raster end to the nearest contour point, prints the contour, then returns to the raster endpoint before the next layer.

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

The fourth tab plots the generated shapes' G-code at once using the nozzle spacing configured on the TIFF-to-G-code tab, each in its own color. Shape Settings maps each STL to a nozzle number, so multiple shapes can share one nozzle offset while valves remain independent. Like the visualization tab it has a fast **Line Plot** and an animated **Tube Plot**; the animation advances all parts on a shared cumulative-path-length timeline, so a shorter part finishes first.

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
