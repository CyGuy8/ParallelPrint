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

Then open the local Gradio URL in your browser, upload STL files or load the bundled samples, and press **Generate G-Code**.

## What the app does

- Uploads any number of `.stl` files with a single multi-file uploader
- Loads bundled sample STL files and merges them with already uploaded STLs — a **Sample Set** dropdown under the button picks between the standard shapes (hollow pyramid, rounded cube, half sphere) and the simple flat shapes (circle, square, triangle)
- Syncs the uploaded STL list back into Shape Settings if the table and uploader get out of step
- Shows an interactive selected-shape 3D viewer for rotating each model
- Shows model extents, face count, vertex count, and watertight status
- Scales loaded STLs from editable target X/Y/Z dimensions in the Shape Settings table; new rows default to the STL's original dimensions, **Reset Dimensions** restores them, and **Keep Proportions** updates the other target sides from the edited side
- Lets you choose layer height and filament/line width
- Slices each shape into per-layer vector outlines held in memory (no intermediate image files)
- Shapes that share a **nozzle number** are treated automatically as one multi-material assembly: sliced on one shared Z grid and kept exactly where they were modeled, while shapes alone on their nozzle behave as ordinary independent parts
- Nozzle groups are visible right in the table: rows sharing a nozzle get a shared tint and accent stripe, and a summary under the table names each assembly ("Nozzle 1: red_base + black_stripes print as one assembly")
- Valve safety check: valve cells used by more than one shape turn red with a warning under the table — shapes sharing a valve dispense simultaneously, which is almost never intended. New shapes default to their own unused valve number
- Port groups are marked too: shapes sharing a Port get a matching underline on their Pressure and Port cells and a summary line ("Port 1: A + B share one pressure regulator (25 psi)") — one regulator per serial port is why their pressures stay in sync
- Automatically unions the sliced shapes into a combined reference layer set whenever shapes are sliced
- Shows a sliced-layer preview in the Selected Shape Preview accordion (layer slider through the shape's polygon outlines, drawn in its print color; assembly parts sharing the nozzle are drawn together so multi-material slicing can be checked before generating G-code)
- Bundled sample sets include a **Multi-Material Demo** (checkerboard cube, wrapped egg, space helmet — two STLs each) that loads with the parts of each model already grouped onto shared nozzles, forming three assemblies in one click
- **Save / Load Settings**: exports every table setting plus the generation options as a small JSON keyed by STL filename; re-upload the same STLs later (or after a Space restart) and import to restore the whole setup — files in the export that aren't loaded yet are listed so they can be added. Split pieces are derived geometry and don't round-trip
- Generate G-Code reports live progress (slicing, reference building, then shape-by-shape generation)
- Splits one sliced shape's geometry into an editable row/column grid for multi-nozzle printing of one large shape
- Converts sliced layers into G-code files with pressure, valve, nozzle, port, and infill % settings per shape from the Shape Settings table
- **Pressure is a port property** (one pressure regulator per serial port): shapes sharing a Port always share one pressure — editing one shape's pressure updates every shape on that port, moving a shape onto a port adopts that port's pressure, and newly added shapes join at their port's existing pressure
- **Infill %** per shape skips dispensing on evenly-distributed raster lines (rings for the circle spiral) — at 50% every other line prints. Parallel shapes with different infill still share one print path, and lines that **no** shape dispenses on are dropped from the motion entirely instead of being swept valve-off (every shape at 50% roughly halves the print path; with mixed infills a line survives if any shape prints it). The rectangular spiral keeps its full continuous walk; circle-spiral perimeter walls always stay
- Appends a shape outline contour after each enabled shape layer by tracing that layer's polygon boundary
- Offers a choice of raster pattern for G-code generation; all shapes always share one combined reference outline for motion (one nozzle path; each dispenses only its own geometry), and every move is emitted as `G1` at one constant speed (no `G0` rapid travel)
- **Sweep Buffer (mm)** sets the valve-settle travel before and after each raster line (default 0.8, 0 disables it); it applies to the axis rasters and both woodpiles, and the same value drives every shape so the shared motion stays in sync
- Slicing is automatic: **Generate G-Code** slices every shape (fresh or stale) before writing G-code, and **Split Selected Shape into Grid Pieces** slices before splitting — "upload, then Generate G-Code" works in one click with no separate slice step
- The generation status leads with the shared print path length and a time estimate at the Visualization tab's nozzle speed, and generation auto-renders the parallel view so switching to the Visualization tab always shows the current print
- A **stale-G-code banner** appears above the Generate button whenever the table or any generation option changes after files were generated (each file carries a settings fingerprint), so outdated G-code is never downloaded or printed unnoticed
- **Assign Unique Valves** resolves valve collisions in one click (first occurrence keeps its number; duplicates and unset valves move to the smallest unused), and **Undo Split** unwinds splits one at a time, newest first (each split pushes a snapshot onto an undo stack, kept for the last 10 splits)
- Calculates X/Y nozzle spacing from a grid layout (columns/rows plus gaps) in the Visualization tab's Nozzle Spacing accordion, with an optional per-connection Advanced Grid Spacing table, then visualizes the resulting nozzle layout
- Previews selected generated G-code inline
- One Visualization tab that defaults to the parallel print of every generated shape (configured nozzle spacing, animated, with a server-side GIF export), and can switch to a single tool path from any generated shape or an uploaded G-code file
- Renders as a fast line plot or an animated 3D tube plot (play/pause, speed, scrub, frame-step, nozzle marker); print colors come from the Shape Settings table (uploads render orange, travel grey)
- Estimates print time from an inputted **Nozzle Speed (mm/s)**: every move runs at one constant speed, so time = tool-path length ÷ speed — shown with the path length in the render status for both the parallel view (longest head's path; all heads move together) and the single-tool-path view
- Each shape's plot color is set with one click on a palette chip embedded in the Shape Settings **Color** column (Orange, Blue, Green, Red, Purple, Pink, Teal, Yellow, White, Black) — the cell shows the current color's name and highlights its chip

## Behavior and Implementation Notes

### Vector Slicing

Slicing uses `trimesh` cross-sections composed into shapely polygons (`slice_stl_to_layers` in `stl_slicer.py`). Each layer is a `MultiPolygon` in world-XY millimetres; the whole shape is a `LayerStack` (layers, z-values, bounds, layer height) held in the Gradio session state — no files are written until G-code is generated.

### Reference Layer Union

Whenever shapes are sliced (automatically, during G-code generation or a split), the app unions the sliced shapes layer-by-layer into a combined reference layer set (used for shared-motion G-code).

- Shapes are aligned by centering each shape's XY bounding box on a common center before the union.
- Alignment is centered placement (in exact millimetres), not bottom-left anchoring.

### Multi-Material Assemblies (shared nozzle numbers)

For a multi-material object exported as separate STLs (one per material), give every part the **same nozzle number** in Shape Settings — parts sharing a nozzle print from the same physical position, so the app treats them automatically as one assembly:

- Group members are sliced on **one shared Z grid** spanning the whole assembly, so a part that starts higher in the model simply has empty lower layers — it travels the shared path but dispenses nothing until the print reaches its height.
- The group is aligned into the reference union as **one rigid unit**: each part keeps exactly the position it was modeled at relative to the others, so asymmetric assemblies line up the way they do in the 3D preview.
- Shapes alone on their nozzle keep the normal behavior (centered onto the common reference), so regular shapes and multi-material assemblies can print together in one job.
- **Contour Tracing** on assembly parts outlines only the assembly's true outer surface: edges where one material meets (or nearly meets, within half a bead — fit tolerances included) another material are internal interfaces and are skipped, exactly like the cut seams of grid-split pieces.
- Motion is always the combined reference outline, so all heads share one synchronized path while each dispenses only its own part.
- Nozzle renumbering in the table takes effect on the next slice or G-code generation — groups are re-detected automatically.
- **Dimension edits scale the whole assembly**: changing a dimension of one group member applies the same scale factor (target ÷ original, per axis) to every shape on that nozzle, in both scaling modes — in Keep Proportions one edit rescales every member proportionally; in Independent X/Y/Z the edited axis's factor propagates to the group. Factors (not absolute values) keep differently-sized parts proportional to each other, and group members are scaled about the assembly's shared corner so the parts stay assembled at any size.

### Multi-Nozzle Split

The **Multi-Nozzle Split** accordion on the **Shapes & G-Code** tab can split one shape into a grid of print-ready piece stacks. Choose a source shape, set the number of columns and rows, choose the starting nozzle and valve numbers, then click **Split Selected Shape into Grid Pieces** — the shapes are sliced automatically first if needed.

- Each layer's geometry is clipped against equal-size grid cells, so every piece keeps exact vector outlines.
- The selected shape is replaced in Shape Settings by one generated record per grid cell, named by row and column.
- Nozzle and valve numbers are assigned sequentially from the starting values, and **Generate G-Code** produces separate G-code for each piece.
- **Overlapping Layers** alternates the interior cut lines by one filament width per layer so neighbouring pieces interlock.
- **Multi-material assemblies split as one shape**: if the selected shape shares its nozzle with other shapes, the whole group is split together — every material is clipped by the same cell grid over the group's combined bounds. Pieces are emitted cell by cell: each cell's pieces share a nozzle (so every cell is itself a multi-material group, keeping the alignment and seam-free contour behavior), each piece gets its own valve, and cells where a material has no geometry are skipped. Auto Align works on the result like any other split.

### G-code XY Step Size

- G-code generation uses the slicer tab's `Filament/Line Width` as the raster line spacing by passing `fil_width` into `generate_vector_gcode()`.

### G-code Output

- Generated G-code starts in relative coordinate mode (`G91`).
- The app opens the valve where the tool path is inside a shape's layer polygons and closes it elsewhere.
- Generated files include pressure preset commands and WAGO valve commands based on the selected pressure, valve, and port; the nozzle number controls layout/spacing assignment.
- Pressure commands are written **once per port**: when shapes share a serial port, only the first shape's file carries the preset, toggle, and per-layer ramp — the print host compiles every file onto one timeline, and duplicated toggles would flip the regulator on/off/on at start. Valve commands stay per shape in every file.
- Pressure increases by `0.1` psi per layer by default.
- Every movement line is emitted as `G1` at one constant speed (no `G0` rapid travel); the WAGO valve commands mark where material is dispensed vs where the head just travels.
- **Lead In**: enabled per shape via the **Lead In** column in Shape Settings; the Lead In Options accordion sets the patch geometry. Prints a purge patch before layer 1: **Lead In Position** (Left/Right/Up/Down) picks which side of the shape it sits on, **Lead In Clearance** how far away, and **Lead In Line Direction** which way the purge strokes run — Auto points them at the shape (the historical behavior), or force Horizontal/Vertical (e.g. a patch below the shape with horizontal strokes). When strokes run across the approach, lines step further away from the shape. The return route exits the patch laterally and comes home through the clearance lane, so the primed nozzle never drags back across the wet purge lines. For grid-split pieces the clearance is automatically extended by the assembly's remaining extent along the purge axis (reported in the G-code status), so under shared reference motion every nozzle's purge patch lands clear of the whole assembled part instead of on a neighbor's print area. The **Lead In** column in Shape Settings controls dispensing per shape: an opted-out head still travels the shared patch (keeping parallel heads in sync) but keeps its valve shut, and skips the lead-in moves entirely when printing without shared motion.
- **Combined reference outline for motion** (always on): every shape's *motion* is taken from the combined reference layer union while each shape's *valve/dispensing* comes from its own layer polygons — so parallel print heads share one synchronized nozzle path and each deposits only its own geometry. The reference union is rebuilt automatically whenever shapes are sliced or G-code is generated. Contour tracing stays synchronized too: every shape traces every traced shape's contour, opening its valve only on its own outline.
- Generated files contain only machine commands (no metadata comments). The toolpath's world anchor — the position, in the shape's own frame, that the relative moves start from — is kept on the shape's session record and used to place parallel parts so split pieces reassemble.
- **Raster Pattern**: `X-direction raster` sweeps every layer back-and-forth in X. `Y-direction raster` rasters every layer in Y. `90° Woodpile raster` alternates the raster axis by layer, switching between X-direction and Y-direction sweeps. `45° Woodpile raster` rotates the sweep 45 degrees per layer, cycling 0, 45, 90, 135 degrees. `Rectangular Spiral raster` walks rectangular loops from the outside toward the center, then reverses from center to edge on the next layer; the loops live on one family anchored to the shape frame, so every layer (and every split sibling) walks the same rectangles and the walls stack — outer loops that cannot touch a layer's material are skipped instead of traveled. `Circle Spiral raster` prints concentric circles stepping inward by one line width per revolution — each revolution stays at a constant radius, so the walls are smooth true circles — then reverses outward on the next layer. The outermost revolution is a perimeter wall hugging the layer's material edge (half a bead inside its farthest boundary point), so the printed silhouette follows the shape smoothly; a matching inner wall hugs a central hole when there is one. The fill rings between the walls come from one global radii grid anchored at the shape frame's center, so interior rings stack exactly across layers, and rings that cannot touch a layer's material (or would overlap a wall bead) are skipped instead of traveled. Walls always dispense even under partial infill, like contour tracing; elsewhere the valve opens only where the path is inside material. Under shared reference motion every shape's own wall radius joins the one shared ring set (all heads travel all walls, each dispenses only its own), and rings that would graze a shape's boundary are suppressed for that shape — so each parallel shape keeps a smooth, complete outer circle regardless of its dimensions.
- **Auto Align Split Parts**: in Nozzle Spacing, computes exact per-connection grid gaps from the split pieces' generated G-code (world toolpath anchors + toolpath bounds), sets the grid columns/rows from the split, and fills the Advanced Grid Spacing table. Works for every raster pattern, filament width, reference-motion setting, and overlapping-layer split. Requires the pieces' G-code to be generated first.
- **Contour Tracing**: enabled per row in Shape Settings. The app traces the layer polygon's boundary rings (holes traced separately), travels from the layer raster end to the nearest contour point, prints the contour, then returns to the raster endpoint before the next layer. For grid-split pieces only the parent shape's true outer surface is traced — the cut seams between sibling pieces are excluded (open arcs are printed end-to-end without closing the loop; fully interior pieces get no contour). Multi-material assembly parts (shapes sharing a nozzle) work the same way: boundary within half a bead of a sibling material counts as an internal interface and is not contoured — only the assembled shape's true outer surface is traced, and a part fully embedded in the assembly gets no contour at all.

### Print vs Travel Classification

When parsing G-code for visualization, the app decides print vs travel as follows:

- If the file contains `WAGO_ValveCommands`, the valve state (open/closed) determines print vs travel. This overrides `G0`/`G1`, because some generators emit every move as `G1`, or invert `G0`/`G1` relative to the valve.
- Otherwise it falls back to the convention `G1` = print, `G0` = travel.

The parser also handles standard slicer G-code: single-axis and Z-only moves, axes in any order, and `F`/`E` tokens (feed rate, extrusion) are ignored for geometry.

### Visualization

The Visualization tab has one source selector. The default, **Parallel print (all shapes)**, plots every generated shape's G-code at once using the spacing configured in the tab's own Nozzle Spacing accordion, each in its Shape Settings color. Shape Settings maps each STL to a nozzle number, so multiple shapes can share one nozzle offset while valves remain independent; the animation advances all parts on a shared cumulative-path-length timeline, so a shorter part finishes first.

Selecting a single shape (or **Upload G-Code file** for a `.txt`, `.gcode`, or `.nc` file) switches to a single-tool-path view. The parser handles `G0`/`G1` movement lines and relative (`G91`) / absolute (`G90`) positioning. Colors are fixed: a generated shape prints in its Shape Settings color, an uploaded file prints in orange, and travel is always grey (opacity sliders control how visible each is).

Both views offer two render modes:

- **Line Plot** — fast thin scatter lines (print and travel).
- **Tube Plot with Animation** — mm-width filament tubes (circular, capped, lit) with a client-side build animation (play/pause, speed, scrub, frame-step) and a moving nozzle marker. Filament/travel widths automatically follow the slicer's Filament/Line Width and its quarter.

The parallel view can also **export the animation as a GIF**, rendered server-side with Matplotlib (the `Agg` CPU backend — no WebGL, no headless browser, and no `ffmpeg`, so it works locally and on Hugging Face). The GIF is line-style with faint grey travel and white, black-outlined nozzle markers drawn on top; controls cover duration, frames per second, elevation/azimuth viewing angle, and travel opacity (0 hides travel).

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
