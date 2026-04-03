"""
FiftyOne panel: repimprov_dashboard

Category browser with video previews.
Derives exercise category directly from filepath so matching always works,
even if exercise_folder field was never written.
"""

import logging
import os
import re
from collections import defaultdict

import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone import ViewField as F

logger = logging.getLogger(__name__)


def _cat_from_path(filepath: str) -> str:
    """Return the parent folder name of a filepath (the exercise category)."""
    if not filepath:
        return "unknown"
    # Works for both / and \ separators
    return os.path.basename(os.path.dirname(filepath)).strip() or "unknown"


def _collect_categories(dataset):
    """
    Derives category from filepath (always present) so the panel works even
    when exercise_folder was never written to the dataset.

    Returns a list of dicts sorted by category name:
      [{ "raw": "bench press", "display": "Bench Press",
         "total": 61, "analyzed": 0, "pending": 61 }, ...]
    """
    try:
        paths  = dataset.values("filepath") or []
        schema = dataset.get_field_schema()
        scores = dataset.values("form_score") if "form_score" in schema else []
        logger.debug(
            "_collect_categories: %d filepaths, form_score in schema=%s",
            len(paths), "form_score" in schema,
        )
    except Exception as exc:
        logger.exception("_collect_categories: failed to read dataset — %s", exc)
        return []

    counts = defaultdict(lambda: {"total": 0, "analyzed": 0})

    try:
        for i, path in enumerate(paths):
            raw   = _cat_from_path(path)
            score = scores[i] if i < len(scores) else None
            counts[raw]["total"] += 1
            if score is not None:
                counts[raw]["analyzed"] += 1
    except Exception as exc:
        logger.exception("_collect_categories: error building counts — %s", exc)

    result = []
    for raw, c in sorted(counts.items()):
        result.append({
            "raw":      raw,
            "display":  raw.title(),
            "total":    c["total"],
            "analyzed": c["analyzed"],
            "pending":  c["total"] - c["analyzed"],
        })

    logger.info("_collect_categories: %d categories from %d filepaths", len(result), len(paths))
    return result


def _select_by_folder(dataset, raw: str):
    """
    Return a dataset view containing only samples whose filepath has `raw`
    as the immediate parent folder name.

    Uses pure Python matching on filepaths, then dataset.select_by_filepath()
    — no regex, no MongoDB expression, no ID lookup.
    """
    try:
        paths = dataset.values("filepath") or []
    except Exception as exc:
        logger.exception("_select_by_folder: failed to read filepaths — %s", exc)
        return dataset.view()

    search = raw.lower()
    matching_paths = []
    for path in paths:
        if not path:
            continue
        norm   = path.replace("\\", "/")
        parts  = norm.split("/")
        parent = parts[-2].lower() if len(parts) >= 2 else ""
        if parent == search:
            matching_paths.append(path)

    logger.info(
        "_select_by_folder: raw=%r → %d/%d matches",
        raw, len(matching_paths), len(paths),
    )

    if not matching_paths:
        logger.warning("_select_by_folder: no matches for %r", raw)
        return dataset.view()

    return dataset.select_by("filepath", matching_paths)


class RepImprovDashboard(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="repimprov_dashboard",
            label="RepImprov Dashboard",
            icon="/assets/icon.svg",
        )

    # ── State ─────────────────────────────────────────────────────────────────

    def on_load(self, ctx):
        try:
            logger.info("on_load — dataset=%s", ctx.dataset.name)
            self._refresh_state(ctx)
        except Exception as exc:
            logger.exception("on_load failed: %s", exc)

    def on_change_dataset(self, ctx):
        try:
            logger.info("on_change_dataset — dataset=%s", ctx.dataset.name)
            self._refresh_state(ctx)
        except Exception as exc:
            logger.exception("on_change_dataset failed: %s", exc)

    def _refresh_state(self, ctx):
        try:
            cats       = _collect_categories(ctx.dataset)
            total_vids = sum(c["total"]    for c in cats)
            total_done = sum(c["analyzed"] for c in cats)
            ctx.panel.set_state("cats",       cats)
            ctx.panel.set_state("total_vids", total_vids)
            ctx.panel.set_state("total_done", total_done)
            logger.info(
                "_refresh_state: %d categories — %d/%d analyzed",
                len(cats), total_done, total_vids,
            )
        except Exception as exc:
            logger.exception("_refresh_state failed: %s", exc)

    # ── Filter actions ────────────────────────────────────────────────────────

    def on_show_all(self, ctx):
        try:
            logger.info("on_show_all")
            ctx.ops.set_view(ctx.dataset.view())
        except Exception as exc:
            logger.exception("on_show_all failed: %s", exc)

    def on_show_analyzed(self, ctx):
        try:
            logger.info("on_show_analyzed")
            schema = ctx.dataset.get_field_schema()
            if "form_score" not in schema:
                logger.warning("on_show_analyzed: form_score not in schema yet")
                ctx.ops.set_view(ctx.dataset.limit(0))
                return
            paths  = ctx.dataset.values("filepath")   or []
            scores = ctx.dataset.values("form_score") or []
            matching = [p for i, p in enumerate(paths) if i < len(scores) and scores[i] is not None]
            logger.info("on_show_analyzed: %d matches", len(matching))
            ctx.ops.set_view(ctx.dataset.select_by("filepath", matching) if matching else ctx.dataset.limit(0))
        except Exception as exc:
            logger.exception("on_show_analyzed failed: %s", exc)

    def on_show_unanalyzed(self, ctx):
        try:
            logger.info("on_show_unanalyzed")
            schema = ctx.dataset.get_field_schema()
            paths  = ctx.dataset.values("filepath")                                    or []
            scores = ctx.dataset.values("form_score") if "form_score" in schema else []
            matching = [p for i, p in enumerate(paths) if i >= len(scores) or scores[i] is None]
            logger.info("on_show_unanalyzed: %d matches", len(matching))
            ctx.ops.set_view(ctx.dataset.select_by("filepath", matching) if matching else ctx.dataset.limit(0))
        except Exception as exc:
            logger.exception("on_show_unanalyzed failed: %s", exc)

    def on_open_category(self, ctx):
        try:
            raw = ctx.params.get("raw", "")
            if not raw:
                logger.warning("on_open_category: empty raw param")
                return
            logger.info("on_open_category: raw=%r", raw)
            view = _select_by_folder(ctx.dataset, raw)
            logger.info("on_open_category: view has %d samples", len(view))
            ctx.ops.set_view(view)
        except Exception as exc:
            logger.exception("on_open_category failed (raw=%r): %s", ctx.params.get("raw"), exc)

    def on_refresh(self, ctx):
        try:
            logger.info("on_refresh")
            self._refresh_state(ctx)
        except Exception as exc:
            logger.exception("on_refresh failed: %s", exc)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, ctx):
        panel = types.Object()
        try:
            return self._render_content(ctx, panel)
        except Exception as exc:
            logger.exception("render failed: %s", exc)
            panel.str(
                "render_error", label="",
                default=f"# RepImprov Dashboard\n\n**Render error:** {exc}\n\nCheck server logs.",
                view=types.MarkdownView(read_only=True),
            )
            return types.Property(panel)

    def _render_content(self, ctx, panel):
        cats       = ctx.panel.state.get("cats", [])
        total_vids = ctx.panel.state.get("total_vids", 0)
        total_done = ctx.panel.state.get("total_done", 0)
        pending    = total_vids - total_done

        # ── Header ────────────────────────────────────────────────────────────
        panel.str(
            "header", label="",
            default="\n".join([
                "# RepImprov  AI Workout Form Analyzer",
                "**Team:** Hiroaki Okumura · Hutch Turner · Laxmi Balcha · Ethan Lee",
                f"**{total_done}** analyzed · **{pending}** pending · **{total_vids}** total",
                "---",
                "*Click any category to load those videos in the grid, then click a video to preview it.*",
            ]),
            view=types.MarkdownView(read_only=True),
        )

        # ── Quick filters ─────────────────────────────────────────────────────
        panel.str(
            "qf_label", label="",
            default="### Quick Filters",
            view=types.MarkdownView(read_only=True),
        )
        panel.btn("show_all",        label=f"All Videos ({total_vids})", on_click=self.on_show_all)
        panel.btn("show_analyzed",   label=f"Analyzed ({total_done})",   on_click=self.on_show_analyzed)
        panel.btn("show_unanalyzed", label=f"Not Analyzed ({pending})",  on_click=self.on_show_unanalyzed)

        panel.str("div1", label="", default="---", view=types.MarkdownView(read_only=True))

        # ── Category list ─────────────────────────────────────────────────────
        panel.str(
            "cat_label", label="",
            default=f"### Exercise Categories  ({len(cats)})",
            view=types.MarkdownView(read_only=True),
        )

        for cat in cats:
            raw      = cat["raw"]
            display  = cat["display"]
            total    = cat["total"]
            analyzed = cat["analyzed"]
            pending_cat = cat["pending"]

            if analyzed == 0:
                status = f"{total} videos  ·  not analyzed"
            elif pending_cat == 0:
                status = f"{total} videos  ·  all analyzed"
            else:
                status = f"{total} videos  ·  {analyzed} analyzed  ·  {pending_cat} pending"

            safe_key = "cat_" + re.sub(r"[^a-z0-9]", "_", raw.lower())
            panel.btn(
                safe_key,
                label=f"{display}  —  {status}",
                on_click=self.on_open_category,
                params={"raw": raw},
            )

        panel.str("div2", label="", default="---", view=types.MarkdownView(read_only=True))
        panel.btn("refresh", label="Refresh", on_click=self.on_refresh)

        return types.Property(panel)


def register(plugin):
    plugin.register(RepImprovDashboard)
