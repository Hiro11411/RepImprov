"""
FiftyOne panel: repimprov_dashboard

Features:
- Summary report with grade distribution, exercise leaderboard, top issues/strengths
- Per-video form issue timestamps (click a category to see them)
- Side-by-side folder comparison (set "Mine" vs "Model" to compare two people)
"""

import logging
import os
import re
from collections import Counter, defaultdict

import fiftyone.operators as foo
import fiftyone.operators.types as types

logger = logging.getLogger(__name__)


# ── Filepath helpers ───────────────────────────────────────────────────────────

def _cat_from_path(filepath: str) -> str:
    if not filepath:
        return "unknown"
    return os.path.basename(os.path.dirname(filepath)).strip() or "unknown"


def _select_by_folder(dataset, raw: str):
    try:
        paths = dataset.values("filepath") or []
    except Exception as exc:
        logger.exception("_select_by_folder failed: %s", exc)
        return dataset.view()

    search = raw.lower()
    matching = []
    for path in paths:
        if not path:
            continue
        norm   = path.replace("\\", "/")
        parts  = norm.split("/")
        parent = parts[-2].lower() if len(parts) >= 2 else ""
        if parent == search:
            matching.append(path)

    logger.info("_select_by_folder: %r -> %d/%d", raw, len(matching), len(paths))
    return dataset.select_by("filepath", matching) if matching else dataset.view()


def _collect_categories(dataset):
    try:
        paths  = dataset.values("filepath") or []
        schema = dataset.get_field_schema()
        scores = dataset.values("form_score") if "form_score" in schema else []
    except Exception as exc:
        logger.exception("_collect_categories failed: %s", exc)
        return []

    counts = defaultdict(lambda: {"total": 0, "analyzed": 0})
    for i, path in enumerate(paths):
        raw   = _cat_from_path(path)
        score = scores[i] if i < len(scores) else None
        counts[raw]["total"] += 1
        if score is not None:
            counts[raw]["analyzed"] += 1

    return [
        {
            "raw":      raw,
            "display":  raw.title(),
            "total":    c["total"],
            "analyzed": c["analyzed"],
            "pending":  c["total"] - c["analyzed"],
        }
        for raw, c in sorted(counts.items())
    ]


# ── Analytics helpers ──────────────────────────────────────────────────────────

def _compute_summary(dataset):
    empty = {
        "analyzed": 0, "avg_score": None, "grade_counts": {},
        "exercise_scores": {}, "top_issues": [], "top_strengths": [],
        "top5": [], "bottom5": [],
    }
    try:
        schema = dataset.get_field_schema()
        if "form_score" not in schema:
            return empty

        paths     = dataset.values("filepath")          or []
        scores    = dataset.values("form_score")        or []
        grades    = dataset.values("form_grade")        or []
        exercises = dataset.values("exercise_detected") if "exercise_detected" in schema else []
        strengths = dataset.values("strengths")         if "strengths"         in schema else []

        issue_labels = []
        if "form_issues" in schema:
            try:
                raw_issues = dataset.values("form_issues.detections.label", unwind=True)
                issue_labels = [l for l in (raw_issues or []) if l]
            except Exception:
                pass

        analyzed_scores  = []
        grade_counts     = Counter()
        exercise_bucket  = defaultdict(list)
        strength_words   = []
        scored_samples   = []

        for i, score in enumerate(scores):
            if score is None:
                continue
            fp       = paths[i]     if i < len(paths)     else ""
            grade    = grades[i]    if i < len(grades)     else "?"
            exercise = exercises[i] if i < len(exercises)  else ""
            if not exercise:
                exercise = _cat_from_path(fp)

            analyzed_scores.append(score)
            grade_counts[grade or "?"] += 1
            exercise_bucket[exercise].append(score)
            scored_samples.append((score, fp, exercise or "?", grade or "?"))

            if i < len(strengths) and strengths[i]:
                for s in (strengths[i] if isinstance(strengths[i], list) else [strengths[i]]):
                    if s:
                        strength_words.append(str(s)[:80])

        if not analyzed_scores:
            return empty

        avg_score = sum(analyzed_scores) / len(analyzed_scores)
        exercise_scores = {
            ex: round(sum(sc) / len(sc), 1)
            for ex, sc in exercise_bucket.items()
        }
        exercise_scores = dict(sorted(exercise_scores.items(), key=lambda x: x[1], reverse=True))

        top_issues    = Counter(issue_labels).most_common(10)
        short_str     = [" ".join(s.split()[:5]) for s in strength_words]
        top_strengths = Counter(short_str).most_common(8)

        scored_samples.sort(key=lambda x: x[0], reverse=True)

        return {
            "analyzed":        len(analyzed_scores),
            "avg_score":       round(avg_score, 1),
            "grade_counts":    dict(grade_counts),
            "exercise_scores": exercise_scores,
            "top_issues":      top_issues,
            "top_strengths":   top_strengths,
            "top5":            scored_samples[:5],
            "bottom5":         scored_samples[-5:][::-1],
        }
    except Exception as exc:
        logger.exception("_compute_summary failed: %s", exc)
        return empty


def _compute_folder_stats(dataset, cat_raw):
    """Per-folder stats for comparison mode."""
    try:
        view   = _select_by_folder(dataset, cat_raw)
        schema = dataset.get_field_schema()

        scores = [s for s in (view.values("form_score") if "form_score" in schema else []) if s is not None]
        grades = [g for g in (view.values("form_grade") if "form_grade" in schema else []) if g]

        if not scores:
            return None

        issue_labels = []
        if "form_issues" in schema:
            try:
                raw = view.values("form_issues.detections.label", unwind=True)
                issue_labels = [l for l in (raw or []) if l]
            except Exception:
                pass

        grade_counts = Counter(grades)
        top_grade    = grade_counts.most_common(1)[0][0] if grade_counts else "?"

        return {
            "total":      len(view),
            "analyzed":   len(scores),
            "avg_score":  round(sum(scores) / len(scores), 1),
            "top_grade":  top_grade,
            "top_issues": Counter(issue_labels).most_common(5),
            "grade_counts": dict(grade_counts),
        }
    except Exception as exc:
        logger.exception("_compute_folder_stats(%r) failed: %s", cat_raw, exc)
        return None


def _get_timestamp_report(dataset, cat_raw, max_videos=10):
    """
    Returns per-video list of bad-form issues AND good-form highlights with
    human-readable timestamps.
    Each entry: {
        name, score, grade,
        issues:     [(ts_str, label, severity, fix)],
        highlights: [(ts_str, label, description)]
    }
    """
    try:
        view   = _select_by_folder(dataset, cat_raw)
        schema = dataset.get_field_schema()
        if "form_score" not in schema:
            return []

        def _safe_get(sample, field):
            try:
                return sample.get_field(field)
            except Exception:
                return None

        def _det_to_ts(det, frame_rate):
            """Convert a TemporalDetection to a timestamp string."""
            try:
                frame  = (det.support or [0])[0]
                ts_sec = float(frame) / frame_rate
                return f"{int(ts_sec // 60)}:{int(ts_sec % 60):02d}"
            except Exception:
                return "0:00"

        results = []
        for sample in view.iter_samples():
            try:
                score = _safe_get(sample, "form_score")
                if score is None:
                    continue

                grade            = _safe_get(sample, "form_grade") or "?"
                coaching_summary = _safe_get(sample, "coaching_summary") or ""
                top_priority_fix = _safe_get(sample, "top_priority_fix") or ""
                raw_strengths    = _safe_get(sample, "strengths") or []
                fp         = sample.filepath or ""
                name       = os.path.basename(fp)
                frame_rate = getattr(sample.metadata, "frame_rate", None) or 30.0

                # ── Bad form issues ───────────────────────────────────────────
                issues = []
                form_issues = _safe_get(sample, "form_issues")
                if form_issues:
                    for det in (form_issues.detections or []):
                        try:
                            ts_str   = _det_to_ts(det, frame_rate)
                            label    = (det.label or "unknown").replace("_", " ").title()
                            severity = _safe_get(det, "severity") or "minor"
                            fix      = _safe_get(det, "fix") or ""
                            issues.append((ts_str, label, severity, fix))
                        except Exception:
                            pass
                    issues.sort(key=lambda x: x[0])

                # ── Good form highlights ──────────────────────────────────────
                highlights = []
                form_highlights = _safe_get(sample, "form_highlights")
                if form_highlights:
                    for det in (form_highlights.detections or []):
                        try:
                            ts_str = _det_to_ts(det, frame_rate)
                            label  = (det.label or "good_form").replace("_", " ").title()
                            desc   = _safe_get(det, "description") or label
                            highlights.append((ts_str, label, desc))
                        except Exception:
                            pass
                    highlights.sort(key=lambda x: x[0])

                # Fall back to plain strength strings if no timestamped highlights
                strength_texts = (
                    [s for s in raw_strengths if isinstance(s, str)]
                    if not highlights else []
                )

                results.append({
                    "name":            name,
                    "score":           score,
                    "grade":           grade,
                    "issues":          issues,
                    "highlights":      highlights,
                    "strength_texts":  strength_texts,
                    "coaching_summary": coaching_summary,
                    "top_priority_fix": top_priority_fix,
                })

            except Exception as exc:
                logger.warning("_get_timestamp_report: skipping sample — %s", exc)

            if len(results) >= max_videos:
                break

        results.sort(key=lambda x: x["score"])  # worst first
        return results

    except Exception as exc:
        logger.exception("_get_timestamp_report failed: %s", exc)
        return []


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _score_bar(score, width=20):
    filled = int(round(score / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _grade_emoji(grade):
    return {"A": "🏆", "B": "✅", "C": "⚠️", "D": "🔻", "F": "❌"}.get(str(grade).upper(), "❓")


def _severity_icon(severity):
    return {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(str(severity).lower(), "⚪")


# ── Panel ──────────────────────────────────────────────────────────────────────

class RepImprovDashboard(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="repimprov_dashboard",
            label="RepImprov Dashboard",
            icon="/assets/icon.svg",
        )

    # ── State ──────────────────────────────────────────────────────────────────

    def on_load(self, ctx):
        try:
            self._refresh_state(ctx)
        except Exception as exc:
            logger.exception("on_load failed: %s", exc)

    def on_change_dataset(self, ctx):
        try:
            self._refresh_state(ctx)
        except Exception as exc:
            logger.exception("on_change_dataset failed: %s", exc)

    def _refresh_state(self, ctx):
        try:
            cats    = _collect_categories(ctx.dataset)
            summary = _compute_summary(ctx.dataset)
            total_vids = sum(c["total"]    for c in cats)
            total_done = sum(c["analyzed"] for c in cats)
            ctx.panel.set_state("cats",       cats)
            ctx.panel.set_state("total_vids", total_vids)
            ctx.panel.set_state("total_done", total_done)
            ctx.panel.set_state("summary",    summary)
            # preserve compare selections across refresh
            if not ctx.panel.state.get("compare_a"):
                ctx.panel.set_state("compare_a", "")
            if not ctx.panel.state.get("compare_b"):
                ctx.panel.set_state("compare_b", "")
        except Exception as exc:
            logger.exception("_refresh_state failed: %s", exc)

    # ── Grid filter actions ────────────────────────────────────────────────────

    def on_show_all(self, ctx):
        try:
            ctx.ops.set_view(ctx.dataset.view())
        except Exception as exc:
            logger.exception("on_show_all failed: %s", exc)

    def on_show_analyzed(self, ctx):
        try:
            schema = ctx.dataset.get_field_schema()
            if "form_score" not in schema:
                ctx.ops.set_view(ctx.dataset.limit(0))
                return
            paths  = ctx.dataset.values("filepath")   or []
            scores = ctx.dataset.values("form_score") or []
            ok = [p for i, p in enumerate(paths) if i < len(scores) and scores[i] is not None]
            ctx.ops.set_view(ctx.dataset.select_by("filepath", ok) if ok else ctx.dataset.limit(0))
        except Exception as exc:
            logger.exception("on_show_analyzed failed: %s", exc)

    def on_show_unanalyzed(self, ctx):
        try:
            schema = ctx.dataset.get_field_schema()
            paths  = ctx.dataset.values("filepath") or []
            scores = ctx.dataset.values("form_score") if "form_score" in schema else []
            ok = [p for i, p in enumerate(paths) if i >= len(scores) or scores[i] is None]
            ctx.ops.set_view(ctx.dataset.select_by("filepath", ok) if ok else ctx.dataset.limit(0))
        except Exception as exc:
            logger.exception("on_show_unanalyzed failed: %s", exc)

    def on_open_category(self, ctx):
        """Filter grid + load timestamps for clicked category."""
        try:
            raw = ctx.params.get("raw", "")
            if not raw:
                return
            ctx.ops.set_view(_select_by_folder(ctx.dataset, raw))
            # Load timestamp report into state (up to 10 videos)
            report = _get_timestamp_report(ctx.dataset, raw, max_videos=10)
            ctx.panel.set_state("detail_cat",    raw)
            ctx.panel.set_state("detail_report", report)
        except Exception as exc:
            logger.exception("on_open_category failed: %s", exc)

    def on_clear_detail(self, ctx):
        try:
            ctx.panel.set_state("detail_cat",    "")
            ctx.panel.set_state("detail_report", [])
            ctx.ops.set_view(ctx.dataset.view())
        except Exception as exc:
            logger.exception("on_clear_detail failed: %s", exc)

    # ── Comparison actions ─────────────────────────────────────────────────────

    def on_set_compare_a(self, ctx):
        try:
            raw = ctx.params.get("raw", "")
            ctx.panel.set_state("compare_a", raw)
            logger.info("compare_a set to %r", raw)
        except Exception as exc:
            logger.exception("on_set_compare_a failed: %s", exc)

    def on_set_compare_b(self, ctx):
        try:
            raw = ctx.params.get("raw", "")
            ctx.panel.set_state("compare_b", raw)
            logger.info("compare_b set to %r", raw)
        except Exception as exc:
            logger.exception("on_set_compare_b failed: %s", exc)

    def on_clear_compare(self, ctx):
        try:
            ctx.panel.set_state("compare_a", "")
            ctx.panel.set_state("compare_b", "")
        except Exception as exc:
            logger.exception("on_clear_compare failed: %s", exc)

    def on_load_compare_a(self, ctx):
        try:
            raw = ctx.panel.state.get("compare_a", "")
            if raw:
                ctx.ops.set_view(_select_by_folder(ctx.dataset, raw))
        except Exception as exc:
            logger.exception("on_load_compare_a failed: %s", exc)

    def on_load_compare_b(self, ctx):
        try:
            raw = ctx.panel.state.get("compare_b", "")
            if raw:
                ctx.ops.set_view(_select_by_folder(ctx.dataset, raw))
        except Exception as exc:
            logger.exception("on_load_compare_b failed: %s", exc)

    # ── Best/worst shortcuts ───────────────────────────────────────────────────

    def on_open_top(self, ctx):
        try:
            fps = [fp for _, fp, _, _ in ctx.panel.state.get("summary", {}).get("top5", []) if fp]
            if fps:
                ctx.ops.set_view(ctx.dataset.select_by("filepath", fps))
        except Exception as exc:
            logger.exception("on_open_top failed: %s", exc)

    def on_open_bottom(self, ctx):
        try:
            fps = [fp for _, fp, _, _ in ctx.panel.state.get("summary", {}).get("bottom5", []) if fp]
            if fps:
                ctx.ops.set_view(ctx.dataset.select_by("filepath", fps))
        except Exception as exc:
            logger.exception("on_open_bottom failed: %s", exc)

    def on_refresh(self, ctx):
        try:
            self._refresh_state(ctx)
        except Exception as exc:
            logger.exception("on_refresh failed: %s", exc)

    # ── Render ─────────────────────────────────────────────────────────────────

    def render(self, ctx):
        panel = types.Object()
        try:
            return self._render_content(ctx, panel)
        except Exception as exc:
            logger.exception("render failed: %s", exc)
            panel.str("render_error", label="", default=f"**Render error:** {exc}", view=types.MarkdownView(read_only=True))
            return types.Property(panel)

    def _render_content(self, ctx, panel):
        cats       = ctx.panel.state.get("cats", [])
        total_vids = ctx.panel.state.get("total_vids", 0)
        total_done = ctx.panel.state.get("total_done", 0)
        pending    = total_vids - total_done
        summary    = ctx.panel.state.get("summary", {})
        analyzed   = summary.get("analyzed", 0)
        avg_score  = summary.get("avg_score")

        detail_cat    = ctx.panel.state.get("detail_cat", "")
        detail_report = ctx.panel.state.get("detail_report", [])
        compare_a     = ctx.panel.state.get("compare_a", "")
        compare_b     = ctx.panel.state.get("compare_b", "")

        # ── Header ─────────────────────────────────────────────────────────────
        panel.str("header", label="", default="\n".join([
            "# RepImprov — AI Workout Form Analyzer",
            "**Team:** Hiroaki Okumura · Hutch Turner · Laxmi Balcha · Ethan Lee",
            f"**{total_done}** analyzed · **{pending}** pending · **{total_vids}** total",
            "---",
        ]), view=types.MarkdownView(read_only=True))

        # ── Quick filters ───────────────────────────────────────────────────────
        panel.str("qf_label", label="", default="### Quick Filters", view=types.MarkdownView(read_only=True))
        panel.btn("show_all",        label=f"All ({total_vids})",        on_click=self.on_show_all)
        panel.btn("show_analyzed",   label=f"Analyzed ({total_done})",   on_click=self.on_show_analyzed)
        panel.btn("show_unanalyzed", label=f"Pending ({pending})",       on_click=self.on_show_unanalyzed)
        panel.str("div0", label="", default="---", view=types.MarkdownView(read_only=True))

        # ── Summary report ─────────────────────────────────────────────────────
        if analyzed > 0 and avg_score is not None:
            self._render_summary(panel, summary, analyzed, avg_score)
            panel.str("div_s", label="", default="---", view=types.MarkdownView(read_only=True))

        # ── Comparison section ─────────────────────────────────────────────────
        self._render_comparison(ctx, panel, cats, compare_a, compare_b)
        panel.str("div_cmp", label="", default="---", view=types.MarkdownView(read_only=True))

        # ── Timestamp detail ───────────────────────────────────────────────────
        if detail_cat and detail_report:
            self._render_timestamps(panel, detail_cat, detail_report)
            panel.str("div_ts", label="", default="---", view=types.MarkdownView(read_only=True))

        # ── Category browser ───────────────────────────────────────────────────
        ex_scores = summary.get("exercise_scores", {})
        panel.str("cat_label", label="", default=f"### Exercise Categories ({len(cats)})", view=types.MarkdownView(read_only=True))
        panel.str("cat_hint",  label="", default=(
            "*Click a category name to load its videos and see form timestamps.*\n"
            "*Use **Mine / Model** buttons to set up a comparison.*"
        ), view=types.MarkdownView(read_only=True))

        for cat in cats:
            raw          = cat["raw"]
            display      = cat["display"]
            total        = cat["total"]
            cat_analyzed = cat["analyzed"]
            cat_pending  = cat["pending"]

            score_tag = f" · avg **{ex_scores[raw]}**" if raw in ex_scores else ""
            a_tag = " ★Mine"  if compare_a == raw else ""
            b_tag = " ★Model" if compare_b == raw else ""

            if cat_analyzed == 0:
                status = f"{total} videos · not analyzed"
            elif cat_pending == 0:
                status = f"{total} videos · all analyzed{score_tag}"
            else:
                status = f"{total} videos · {cat_analyzed} done · {cat_pending} pending"

            safe = "cat_" + re.sub(r"[^a-z0-9]", "_", raw.lower())
            panel.btn(
                safe,
                label=f"{display}{a_tag}{b_tag}  —  {status}",
                on_click=self.on_open_category,
                params={"raw": raw},
            )
            # Mine / Model assignment buttons
            panel.btn(
                safe + "_mine",
                label="Set as Mine",
                on_click=self.on_set_compare_a,
                params={"raw": raw},
            )
            panel.btn(
                safe + "_model",
                label="Set as Model",
                on_click=self.on_set_compare_b,
                params={"raw": raw},
            )

        panel.str("div2", label="", default="---", view=types.MarkdownView(read_only=True))
        panel.btn("refresh", label="Refresh Stats", on_click=self.on_refresh)

        return types.Property(panel)

    # ── Sub-renderers ──────────────────────────────────────────────────────────

    def _render_summary(self, panel, summary, analyzed, avg_score):
        grade_counts  = summary.get("grade_counts", {})
        ex_scores     = summary.get("exercise_scores", {})
        top_issues    = summary.get("top_issues", [])
        top_strengths = summary.get("top_strengths", [])
        top5          = summary.get("top5", [])
        bottom5       = summary.get("bottom5", [])

        # Score card
        bar = _score_bar(avg_score)
        panel.str("score_card", label="", default="\n".join([
            "### Overall Form Score",
            "```",
            f"  Average Score   {avg_score:>5.1f} / 100",
            f"  {bar}",
            f"  Videos Analyzed {analyzed:>5}",
            "```",
        ]), view=types.MarkdownView(read_only=True))

        # Grade distribution
        if grade_counts:
            total_graded = sum(grade_counts.values())
            lines = ["### Grade Distribution", ""]
            for g in ["A", "B", "C", "D", "F"]:
                count = grade_counts.get(g, 0)
                if count == 0:
                    continue
                pct  = count / total_graded * 100
                bar2 = "█" * int(pct / 5)
                lines.append(f"{_grade_emoji(g)} **{g}**  {bar2:<20} {count:>3} videos ({pct:.0f}%)")
            panel.str("grade_dist", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))

        # Exercise leaderboard
        if ex_scores:
            lines = ["### Exercise Leaderboard", "", "| Exercise | Avg Score | Bar |", "|---|---|---|"]
            for ex, sc in ex_scores.items():
                lines.append(f"| {ex.title()} | **{sc}** | {'█' * int(sc / 10)} |")
            panel.str("ex_board", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))

        # Top issues
        if top_issues:
            lines = ["### Most Common Form Issues", ""]
            for rank, (label, count) in enumerate(top_issues, 1):
                lines.append(f"{rank}. **{label.replace('_', ' ').title()}** — {count} occurrence{'s' if count != 1 else ''}")
            panel.str("top_issues", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))

        # Top strengths
        if top_strengths:
            lines = ["### Most Common Strengths", ""]
            for rank, (label, count) in enumerate(top_strengths, 1):
                lines.append(f"{rank}. {label.capitalize()} *(seen {count}x)*")
            panel.str("top_strengths", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))

        # Best/worst
        if top5:
            lines = ["### Top 5 Best Form", ""]
            for score, fp, exercise, grade in top5:
                lines.append(f"- {_grade_emoji(grade)} **{score:.0f}** — {os.path.basename(fp)} *({exercise.title()})*")
            panel.str("top5", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))
            panel.btn("view_top5", label="Load Top 5 in Grid", on_click=self.on_open_top)

        if bottom5:
            lines = ["### Bottom 5 — Needs Most Work", ""]
            for score, fp, exercise, grade in bottom5:
                lines.append(f"- {_grade_emoji(grade)} **{score:.0f}** — {os.path.basename(fp)} *({exercise.title()})*")
            panel.str("bottom5", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))
            panel.btn("view_bottom5", label="Load Bottom 5 in Grid", on_click=self.on_open_bottom)

    def _render_comparison(self, ctx, panel, cats, compare_a, compare_b):
        panel.str("cmp_label", label="", default="### Side-by-Side Comparison", view=types.MarkdownView(read_only=True))

        a_label = compare_a.title() if compare_a else "*(not set)*"
        b_label = compare_b.title() if compare_b else "*(not set)*"

        panel.str("cmp_status", label="", default=(
            f"**Your Videos:** {a_label}   |   **Model / Reference:** {b_label}\n\n"
            "*Use the **Set as Mine** / **Set as Model** buttons under each category below.*"
        ), view=types.MarkdownView(read_only=True))

        if compare_a and compare_b:
            stats_a = _compute_folder_stats(ctx.dataset, compare_a)
            stats_b = _compute_folder_stats(ctx.dataset, compare_b)

            if stats_a and stats_b:
                diff = round(stats_b["avg_score"] - stats_a["avg_score"], 1)
                diff_str = f"+{diff}" if diff >= 0 else str(diff)
                winner = compare_b.title() if diff >= 0 else compare_a.title()

                lines = [
                    f"#### {compare_a.title()} (Mine) vs {compare_b.title()} (Model)",
                    "",
                    f"| Metric | {compare_a.title()} | {compare_b.title()} |",
                    "|---|---|---|",
                    f"| Videos Analyzed | {stats_a['analyzed']} | {stats_b['analyzed']} |",
                    f"| **Avg Form Score** | **{stats_a['avg_score']}** | **{stats_b['avg_score']}** |",
                    f"| Most Common Grade | {_grade_emoji(stats_a['top_grade'])} {stats_a['top_grade']} | {_grade_emoji(stats_b['top_grade'])} {stats_b['top_grade']} |",
                    f"| Score Difference | | {diff_str} pts |",
                    "",
                    f"**Score bar — Mine:**  {_score_bar(stats_a['avg_score'])} {stats_a['avg_score']}",
                    f"**Score bar — Model:** {_score_bar(stats_b['avg_score'])} {stats_b['avg_score']}",
                    "",
                ]

                # Grade breakdown side by side
                all_grades = sorted(set(list(stats_a["grade_counts"].keys()) + list(stats_b["grade_counts"].keys())))
                if all_grades:
                    lines.append("**Grade breakdown:**")
                    lines.append("")
                    lines.append(f"| Grade | {compare_a.title()} | {compare_b.title()} |")
                    lines.append("|---|---|---|")
                    for g in ["A", "B", "C", "D", "F"]:
                        ca = stats_a["grade_counts"].get(g, 0)
                        cb = stats_b["grade_counts"].get(g, 0)
                        if ca == 0 and cb == 0:
                            continue
                        lines.append(f"| {_grade_emoji(g)} {g} | {ca} | {cb} |")

                lines.append("")

                # Top issues comparison
                if stats_a["top_issues"] or stats_b["top_issues"]:
                    lines.append("**Most common issues:**")
                    lines.append("")
                    a_issues = [label for label, _ in stats_a["top_issues"][:3]]
                    b_issues = [label for label, _ in stats_b["top_issues"][:3]]
                    for i in range(max(len(a_issues), len(b_issues))):
                        ai = a_issues[i].replace("_", " ").title() if i < len(a_issues) else "—"
                        bi = b_issues[i].replace("_", " ").title() if i < len(b_issues) else "—"
                        lines.append(f"- Mine: **{ai}** | Model: **{bi}**")

                lines += ["", f"> **{winner}** scores higher by **{abs(diff):.1f} points**"]

                panel.str("cmp_result", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))
                panel.btn("cmp_load_a", label=f"Load {compare_a.title()} in Grid", on_click=self.on_load_compare_a)
                panel.btn("cmp_load_b", label=f"Load {compare_b.title()} in Grid", on_click=self.on_load_compare_b)

            else:
                msg = "No analyzed videos found in one or both selected folders."
                panel.str("cmp_no_data", label="", default=msg, view=types.MarkdownView(read_only=True))

            panel.btn("cmp_clear", label="Clear Comparison", on_click=self.on_clear_compare)

    def _render_timestamps(self, panel, cat_raw, report):
        lines = [
            f"### Timeline Breakdown — {cat_raw.title()}",
            "",
            "🟢 good form &nbsp;·&nbsp; 🔴 critical &nbsp;·&nbsp; 🟠 moderate &nbsp;·&nbsp; 🟡 minor",
            "",
            "*Click any video in the grid, then scrub to the timestamps shown below.*",
            "",
        ]

        for entry in report:
            name            = entry["name"]
            score           = entry["score"]
            grade           = entry["grade"]
            issues          = entry.get("issues", [])
            highlights      = entry.get("highlights", [])
            strength_texts  = entry.get("strength_texts", [])
            coaching        = entry.get("coaching_summary", "")
            top_fix         = entry.get("top_priority_fix", "")

            lines.append(f"#### {_grade_emoji(grade)} {name}  —  Score: **{score:.0f}** / 100")

            # Merge timestamped events and sort chronologically
            events = []
            for ts_str, label, severity, fix in issues:
                events.append((ts_str, "bad", label, severity, fix))
            for ts_str, label, desc in highlights:
                events.append((ts_str, "good", label, "", desc))
            events.sort(key=lambda x: x[0])

            if events:
                for event in events:
                    ts_str, kind = event[0], event[1]
                    if kind == "bad":
                        _, _, label, severity, fix = event
                        icon    = _severity_icon(severity)
                        fix_str = f" → *{fix}*" if fix else ""
                        lines.append(f"- `{ts_str}` {icon} **{label}** ({severity}){fix_str}")
                    else:
                        _, _, label, _, desc = event
                        lines.append(f"- `{ts_str}` 🟢 **{label}** — *{desc}*")
            else:
                # No timestamped events — show text analysis instead
                if strength_texts:
                    lines.append("**What's working:**")
                    for s in strength_texts[:3]:
                        lines.append(f"- 🟢 {s}")
                if issues == [] and not strength_texts:
                    lines.append("*No timestamped events — showing coaching analysis below.*")

            # Always show coaching summary and top fix
            if coaching:
                lines.append("")
                lines.append(f"**Coach says:** {coaching}")
            if top_fix:
                lines.append("")
                lines.append(f"**Priority fix:** 🔴 {top_fix}")

            lines.append("")
            lines.append("---")
            lines.append("")

        panel.str("ts_report", label="", default="\n".join(lines), view=types.MarkdownView(read_only=True))
        panel.btn("ts_clear", label="Clear Detail View", on_click=self.on_clear_detail)


def register(plugin):
    plugin.register(RepImprovDashboard)
