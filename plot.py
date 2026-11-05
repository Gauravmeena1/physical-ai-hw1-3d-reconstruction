"""Panel dashboards for measured experiment factors and reconstruction runs.

Imports Panel/Bokeh lazily so the core Turtle and SPARQL CLI remains usable
without dashboard dependencies.
"""
import math
import os


def _experiment_data(path):
    import api
    from rdflib import RDF

    exp = api.read_experiment(path)
    graph, exp_iri = exp["graph"], exp["exp_iri"]
    batch = graph.value(exp_iri, api.HW1.onBatch)
    indices = {}
    frame_stems = []
    frame_images = {}
    for frame in graph.objects(batch, api.HW1.hasFrame):
        idx = graph.value(frame, api.HW1.frameIndex)
        if idx is not None:
            stem = int(idx)
            frame_stems.append(stem)
            rgb_node = graph.value(frame, api.HW1.hasRGBImage)
            depth_node = graph.value(frame, api.HW1.hasDepthImage)
            rgb_url = graph.value(rgb_node, api.SCHEMA.contentUrl) if rgb_node is not None else None
            depth_url = graph.value(depth_node, api.SCHEMA.contentUrl) if depth_node is not None else None
            frame_images[stem] = {
                "rgb": str(rgb_url) if rgb_url is not None else None,
                "depth": str(depth_url) if depth_url is not None else None,
            }
        for image in (graph.value(frame, api.HW1.hasRGBImage),
                      graph.value(frame, api.HW1.hasDepthImage)):
            if image is not None and idx is not None:
                indices[image] = int(idx)

    definitions = api.load_quality_factors()
    factors = {name: [] for name in definitions}
    for node in graph.subjects(api.HW1.inExperiment, exp_iri):
        definition = graph.value(node, api.HW1.hasDefinition)
        if definition is None:
            continue
        name = api._local(definition)
        if name not in factors:
            continue
        current = graph.value(node, api.HW1.hasCurrentFrame)
        previous = graph.value(node, api.HW1.hasPrevious)
        idx = indices.get(current)
        if idx is None:
            continue
        raw = graph.value(node, api.HW1.value)
        nonfinite = False
        try:
            value = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            value = None
        if value is not None and not math.isfinite(value):
            nonfinite = True
            value = None
        factors[name].append({
            "frame": idx, "previous": indices.get(previous), "value": value,
            "pair": previous is not None,
            "nonfinite": nonfinite,
        })

    runs = []
    for mode, run in api._run_nodes(graph, exp_iri):
        val = graph.value(run, api.HW1.mapMeanL2)
        count = graph.value(run, api.HW1.runFrameCount)
        try:
            val = float(val) if val is not None else None
        except (TypeError, ValueError):
            val = None
        try:
            count = int(count) if count is not None else None
        except (TypeError, ValueError):
            count = None
        runs.append({"mode": mode, "frames": count, "mapMeanL2": val})
    frame_count = len(list(graph.objects(batch, api.HW1.hasFrame)))
    return exp, definitions, factors, runs, frame_count, sorted(frame_stems), frame_images


_PLACEHOLDER_IMG = (
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='100' "
    "style='background:%23222;display:flex;align-items:center;justify-content:center;'>"
    "<text x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23aaa' "
    "font-size='12' font-family='sans-serif'>No Image</text></svg>"
)


def _resolve_image_path(raw_path, exp_file_path):
    if not raw_path:
        return None
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path
    if os.path.exists(raw_path):
        return os.path.abspath(raw_path)
    base_dir = os.path.dirname(os.path.abspath(exp_file_path))
    cand = os.path.join(base_dir, raw_path)
    if os.path.exists(cand):
        return cand
    cand_parent = os.path.join(os.path.dirname(base_dir), raw_path)
    if os.path.exists(cand_parent):
        return cand_parent
    return raw_path


def _encode_rgb_thumbnail(raw_path, exp_file_path, max_dim=(140, 105)):
    if not raw_path:
        return _PLACEHOLDER_IMG
    resolved = _resolve_image_path(raw_path, exp_file_path)
    if not resolved or not os.path.exists(resolved):
        return _PLACEHOLDER_IMG
    try:
        import cv2
        img = cv2.imread(resolved)
        if img is not None:
            h, w = img.shape[:2]
            scale = min(max_dim[0] / max(1, w), max_dim[1] / max(1, h))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                import base64
                return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
    except Exception:
        pass
    try:
        from PIL import Image
        import io, base64
        with Image.open(resolved) as im:
            im = im.convert("RGB")
            im.thumbnail(max_dim)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return _PLACEHOLDER_IMG


def _encode_depth_thumbnail(raw_path, exp_file_path, max_dim=(140, 105)):
    if not raw_path:
        return _PLACEHOLDER_IMG
    resolved = _resolve_image_path(raw_path, exp_file_path)
    if not resolved or not os.path.exists(resolved):
        return _PLACEHOLDER_IMG
    try:
        import cv2
        import numpy as np
        depth = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            valid = (depth > 0) & np.isfinite(depth)
            norm = np.zeros_like(depth, dtype=np.uint8)
            if valid.any():
                d_min, d_max = float(depth[valid].min()), float(depth[valid].max())
                if d_max > d_min:
                    norm[valid] = np.clip((depth[valid] - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
                else:
                    norm[valid] = 255
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            colored[~valid] = [0, 0, 0]
            h, w = colored.shape[:2]
            scale = min(max_dim[0] / max(1, w), max_dim[1] / max(1, h))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = cv2.resize(colored, (nw, nh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                import base64
                return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
    except Exception:
        pass
    try:
        from PIL import Image
        import numpy as np
        import io, base64
        with Image.open(resolved) as im:
            arr = np.array(im).astype(float)
            valid = (arr > 0) & np.isfinite(arr)
            norm = np.zeros_like(arr, dtype=np.uint8)
            if valid.any():
                d_min, d_max = float(arr[valid].min()), float(arr[valid].max())
                if d_max > d_min:
                    norm[valid] = np.clip((arr[valid] - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)
                else:
                    norm[valid] = 255
            im_norm = Image.fromarray(norm, mode="L")
            im_norm.thumbnail(max_dim)
            buf = io.BytesIO()
            im_norm.save(buf, format="JPEG", quality=80)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return _PLACEHOLDER_IMG

def _plots(path):
    import api
    from bokeh.events import MouseLeave
    from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Span
    from bokeh.plotting import figure
    import panel as pn

    exp, definitions, factors, runs, batch_size, frame_stems, frame_images = _experiment_data(path)
    name = exp["exp_name"]
    units = {
        "RawPointCount": "points", "VoxelizedPointYield": "points",
        "CoarseRANSACFitness": "ratio", "CoarseRANSACRMSE": "m",
        "CoarseRANSACCorrespondences": "matches",
        "ICPInlierFitness": "ratio", "ICPInlierRMSE": "m",
        "ICPInlierCorrespondences": "matches",
    }

    frame_thumbnails = {}
    for stem in frame_stems:
        info = frame_images.get(stem, {})
        frame_thumbnails[stem] = {
            "rgb": _encode_rgb_thumbnail(info.get("rgb"), path),
            "depth": _encode_depth_thumbnail(info.get("depth"), path),
        }

    timelines = []
    histograms = []
    for factor in definitions:
        rows = sorted(factors[factor], key=lambda row: (row["frame"], row["pair"]))
        pair = any(row["pair"] for row in rows)
        title = factor + (" (pair measurement at current frame; previous frame shown on hover)" if pair
                          else " (per-frame measurement)")
        frame_indices = frame_stems
        present = {row["frame"]: row for row in rows}
        expected_indices = frame_indices[1:] if pair else frame_indices
        previous_for = {frame_indices[i]: frame_indices[i - 1]
                        for i in range(1, len(frame_indices))} if pair else {}
        plotted = [present.get(index, {"frame": index, "previous": previous_for.get(index),
                                      "value": None,
                                      "pair": pair, "nonfinite": False})
                   for index in expected_indices]
        valid = [row for row in plotted if row["value"] is not None]
        cds_data = {
            "frame": [r["frame"] for r in plotted],
            "previous": [r["previous"] if r["previous"] is not None else "—" for r in plotted],
            "value": [r["value"] if r["value"] is not None else float("nan") for r in plotted],
            "rgb_img": [frame_thumbnails.get(r["frame"], {}).get("rgb", _PLACEHOLDER_IMG) for r in plotted],
            "depth_img": [frame_thumbnails.get(r["frame"], {}).get("depth", _PLACEHOLDER_IMG) for r in plotted],
        }
        if pair:
            cds_data["prev_rgb_img"] = [
                frame_thumbnails.get(r["previous"], {}).get("rgb", _PLACEHOLDER_IMG)
                if r["previous"] is not None else _PLACEHOLDER_IMG
                for r in plotted
            ]
            cds_data["prev_depth_img"] = [
                frame_thumbnails.get(r["previous"], {}).get("depth", _PLACEHOLDER_IMG)
                if r["previous"] is not None else _PLACEHOLDER_IMG
                for r in plotted
            ]
        source = ColumnDataSource(cds_data)
        p = figure(title=title, width=820, height=230, tools="pan,wheel_zoom,box_zoom,reset,save",
                   x_axis_label="Frame index", y_axis_label=f"{factor} ({units.get(factor, 'value')})")
        if valid:
            # Vertical dashed guide snapped to the hovered frame. Hovering
            # anywhere along the vertical line (e.g. down at the x-axis)
            # highlights the point at that frame and raises the guide to it.
            guide = Span(location=0, dimension="height",
                         line_color="#496a81", line_alpha=0.7,
                         line_width=1.5, line_dash="dashed", visible=False)
            p.add_layout(guide)
            show_guide = CustomJS(args=dict(source=source, guide=guide), code="""
                const indices = cb_data.index.indices;
                if (indices.length > 0) {
                    guide.location = source.data['frame'][indices[0]];
                    guide.visible = true;
                } else {
                    guide.visible = false;
                }
            """)
            hide_guide = CustomJS(args=dict(guide=guide), code="guide.visible = false;")
            p.js_on_event(MouseLeave, hide_guide)
            line_r = p.line("frame", "value", source=source, color="#496a81", alpha=.6, line_width=2)
            scatter_r = p.scatter("frame", "value", source=source, color="#496a81", size=7)
            if pair:
                tooltips = """
                <div style="padding: 6px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    <div style="font-weight: 600; font-size: 12px; margin-bottom: 2px;">Pair: Frame @previous &rarr; Frame @frame</div>
                    <div style="font-size: 11px; margin-bottom: 4px; color: #333;"><b>Value:</b> @value{0.0000}</div>
                    <div style="font-size: 10px; font-weight: 600; color: #444; margin-top: 4px;">Current Frame (@frame):</div>
                    <table style="border-collapse: collapse;">
                        <tr>
                            <th style="font-size: 9px; font-weight: 600; color: #666; text-align: center; padding: 1px;">RGB</th>
                            <th style="font-size: 9px; font-weight: 600; color: #666; text-align: center; padding: 1px;">Depth</th>
                        </tr>
                        <tr>
                            <td style="padding: 1px;"><img src="@rgb_img" style="width: 110px; height: auto; border: 1px solid #bbb; border-radius: 3px;" /></td>
                            <td style="padding: 1px;"><img src="@depth_img" style="width: 110px; height: auto; border: 1px solid #bbb; border-radius: 3px;" /></td>
                        </tr>
                    </table>
                    <div style="font-size: 10px; font-weight: 600; color: #444; margin-top: 4px;">Previous Frame (@previous):</div>
                    <table style="border-collapse: collapse;">
                        <tr>
                            <th style="font-size: 9px; font-weight: 600; color: #666; text-align: center; padding: 1px;">RGB</th>
                            <th style="font-size: 9px; font-weight: 600; color: #666; text-align: center; padding: 1px;">Depth</th>
                        </tr>
                        <tr>
                            <td style="padding: 1px;"><img src="@prev_rgb_img" style="width: 110px; height: auto; border: 1px solid #bbb; border-radius: 3px;" /></td>
                            <td style="padding: 1px;"><img src="@prev_depth_img" style="width: 110px; height: auto; border: 1px solid #bbb; border-radius: 3px;" /></td>
                        </tr>
                    </table>
                </div>
                """
            else:
                tooltips = """
                <div style="padding: 6px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    <div style="font-weight: 600; font-size: 12px; margin-bottom: 2px;">Frame @frame</div>
                    <div style="font-size: 11px; margin-bottom: 4px; color: #333;"><b>Value:</b> @value{0.0000}</div>
                    <table style="border-collapse: collapse;">
                        <tr>
                            <th style="font-size: 10px; font-weight: 600; color: #555; text-align: center; padding: 2px;">RGB</th>
                            <th style="font-size: 10px; font-weight: 600; color: #555; text-align: center; padding: 2px;">Depth</th>
                        </tr>
                        <tr>
                            <td style="padding: 2px;"><img src="@rgb_img" style="width: 130px; height: auto; border: 1px solid #bbb; border-radius: 3px;" /></td>
                            <td style="padding: 2px;"><img src="@depth_img" style="width: 130px; height: auto; border: 1px solid #bbb; border-radius: 3px;" /></td>
                        </tr>
                    </table>
                </div>
                """
            p.add_tools(HoverTool(tooltips=tooltips, mode="vline", renderers=[line_r, scatter_r],
                                  callback=show_guide))
        else:
            p.text(x=[0], y=[0], text=["No measured values yet"])
        timelines.append(p)

        finite = [row["value"] for row in rows if row["value"] is not None]
        counts, edges = __import__("numpy").histogram(finite, bins=min(20, max(1, len(finite)))) if finite else ([], [])
        hp = figure(title=factor, width=390, height=240, tools="pan,wheel_zoom,reset,save",
                    x_axis_label="Recorded value", y_axis_label="Count")
        if finite:
            hp.quad(top=counts, bottom=0, left=edges[:-1], right=edges[1:],
                    fill_color="#5b8db8", line_color="white")
        else:
            hp.text(x=[0], y=[0], text=["No finite measurements"])
        missing = sum(row["value"] is None and not row["nonfinite"] for row in rows)
        nonfinite = sum(row["nonfinite"] for row in rows)
        hp.title.text += f" — n={len(finite)} finite, {missing} missing, {nonfinite} non-finite"
        histograms.append(hp)

    trade = figure(title="Measured reconstruction outcomes", width=820, height=340,
                   x_axis_label="Retained frame fraction", y_axis_label="Map mean L2 (m)",
                   tools="pan,wheel_zoom,box_zoom,reset,save")
    measured = [r for r in runs if r["frames"] is not None and r["mapMeanL2"] is not None
                and math.isfinite(r["mapMeanL2"])]
    if measured:
        ts = ColumnDataSource({"fraction": [r["frames"] / float(batch_size or 1) for r in measured],
                               "frames": [r["frames"] for r in measured],
                               "error": [r["mapMeanL2"] for r in measured],
                               "mode": [r["mode"] for r in measured]})
        trade.scatter("fraction", "error", source=ts, size=12, color="#7651a8",
                      legend_field="mode")
        trade.x_range.start = 0
        trade.x_range.end = 1
        trade.add_tools(HoverTool(tooltips=[("run", "@mode"), ("frames", "@frames"),
                                            ("map mean L2", "@error{0.0000}")]))
    else:
        trade.text(x=[0], y=[0], text=["No comparable measured run outcomes yet"])
    if not measured:
        trade.text(x=[0], y=[0], text=["Runs are measured observations; no sweep is synthesized."])
    note = pn.pane.Markdown(
        f"### Experiment `{name}`\n"
        f"Batch contains {batch_size} frames. "
        "The tradeoff plot uses only recorded baseline/selected runs. The schema stores one run "
        "per mode, so one experiment has at most two points; no reconstruction sweep is run or inferred.\n\n"
        "**Tip:** Hover over any point in the factor timelines to inspect the corresponding RGB and depth image pair.",
        sizing_mode="stretch_width")
    timeline_tab = pn.Column(note, *timelines, sizing_mode="stretch_width")
    distribution_tab = pn.Column(note, pn.GridBox(*histograms, ncols=2), sizing_mode="stretch_width")
    trade_tab = pn.Column(note, trade, sizing_mode="stretch_width")
    dashboard = pn.Tabs(("Factor timelines", timeline_tab),
                        ("Factor distributions", distribution_tab), ("Selection tradeoff", trade_tab),
                        sizing_mode="stretch_width")
    return dashboard, {"timelines": timeline_tab,
                       "distributions": distribution_tab, "tradeoff": trade_tab}


def _save_dashboard(path, output_dir):
    import panel as pn
    from bokeh.resources import INLINE

    os.makedirs(output_dir, exist_ok=True)
    dashboard, pages = _plots(path)
    artifacts = []
    for filename, page in pages.items():
        artifact = os.path.join(output_dir, f"{filename}.html")
        pn.io.save.save(page, artifact, resources=INLINE, title=f"Experiment plots: {filename}")
        artifacts.append(artifact)
    dashboard_file = os.path.join(output_dir, "dashboard.html")
    pn.io.save.save(dashboard, dashboard_file, resources=INLINE,
                    title="Experiment factor dashboard")
    artifacts.insert(0, dashboard_file)
    return dashboard, artifacts

def generate(path, output_dir, launch_dashboard=False, port=0):
    """Write three interactive plot pages plus dashboard.html; optionally serve dashboard."""
    import panel as pn

    dashboard, artifacts = _save_dashboard(path, output_dir)
    if launch_dashboard:
        for artifact in artifacts:
            print(f"[explore] wrote plot: {artifact}")
        pn.serve(dashboard, address="localhost", port=port, show=True,
                 title="Experiment factor dashboard")
    return artifacts


def generate_many(items, launch_dashboard=False, port=0):
    """Save each experiment's artifacts and optionally serve a tabbed comparison dashboard."""
    import panel as pn

    dashboards, artifacts = [], []
    for path, output_dir in items:
        dashboard, files = _save_dashboard(path, output_dir)
        name = os.path.splitext(os.path.basename(path))[0]
        dashboards.append((name, dashboard))
        artifacts.extend(files)
    if launch_dashboard and dashboards:
        for artifact in artifacts:
            print(f"[explore] wrote plot: {artifact}")
        combined = pn.Tabs(*dashboards, sizing_mode="stretch_width")
        pn.serve(combined, address="localhost", port=port, show=True,
                 title="Experiment factor dashboards")
    return artifacts
