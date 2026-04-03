"""
FiftyOne panel: repimprov_dashboard

Hackathon demo dashboard for RepImprov.
Shows aggregate coaching intelligence across all analyzed workout videos.
"""

import logging
from collections import Counter, defaultdict

import fiftyone.operators as foo
import fiftyone.operators.types as types

logger = logging.getLogger(__name__)

# ASCII bar chart config
BAR_MAX_WIDTH = 20


def _bar(value: float, max_value: float, width: int = BAR_MAX_WIDTH) -> str:
    """Render a scaled ASCII progress bar."""
    if max_value <= 0:
        filled = 0
    else:
        filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _score_label(score: float) -> str:
    """Return a text quality label for a 0-100 score."""
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 70:
        return "Fair"
    if score >= 60:
        return "Poor"
    return "Critical"


class RepImprovDashboard(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="repimprov_dashboard",
            label="RepImprov Dashboard",
            icon="/assets/icon.svg",
        )

    def render(self, ctx):
        dataset = ctx.dataset
        panel = types.Object()
        lines = []

        # ── Collect metrics ───────────────────────────────────────────────────
        form_scores = []
        posture_scores = []
        rep_counts = []
        grades = []
        error_count = 0

        exercise_scores = defaultdict(list)
        issue_labels = []
        issue_severities = []          # flat list of severity strings
        severity_by_issue = defaultdict(list)  # issue_label -> [severities]
        all_strengths = []
        all_priority_fixes = []
        all_coaching_summaries = []    # (filename, score, grade, summary)

        for sample in dataset.iter_samples():
            grade = sample.get_field("form_grade")
            if grade == "ERROR":
                error_count += 1
                continue

            score = sample.get_field("form_score")
            ps    = sample.get_field("posture_score")
            reps  = sample.get_field("rep_count")

            if score is not None:
                form_scores.append(float(score))
            if ps is not None:
                posture_scores.append(float(ps))
            if reps is not None:
                rep_counts.append(int(reps))
            if grade and grade != "ERROR":
                grades.append(grade)

            exercise = sample.get_field("exercise_detected")
            if exercise and score is not None:
                exercise_scores[exercise].append(float(score))

            form_issues = sample.get_field("form_issues")
            if form_issues and hasattr(form_issues, "detections"):
                for det in form_issues.detections:
                    if det.label:
                        label = det.label
                        sev   = det.get_field("severity") or "minor"
                        issue_labels.append(label)
                        issue_severities.append(sev)
                        severity_by_issue[label].append(sev)

            strengths = sample.get_field("strengths")
            if strengths and isinstance(strengths, list):
                all_strengths.extend(strengths)

            fix = sample.get_field("top_priority_fix")
            if fix:
                all_priority_fixes.append(str(fix))

            summary = sample.get_field("coaching_summary")
            filename = (sample.filepath or "").split("/")[-1].split("\\")[-1]
            if summary and score is not None and grade:
                all_coaching_summaries.append((filename, float(score), str(grade), str(summary)))

        total = len(form_scores)

        # ── Header ────────────────────────────────────────────────────────────
        lines.append("# RepImprov — AI Workout Form Analyzer")
        lines.append(
            "_Powered by TwelveLabs Pegasus 1.2 · "
            "Video Understanding AI Hackathon · Northeastern University_"
        )
        lines.append("")

        if total == 0:
            lines.append("---")
            lines.append(
                "**No analyzed samples yet.**  "
                "Run `RepImprov: Analyze Workout Form` from the Operator Browser "
                "(press `` ` ``) to get started."
            )
            if error_count:
                lines.append(f"> {error_count} sample(s) failed to process.")
            panel.str("content", label="", default="\n".join(lines), view=types.MarkdownView())
            return types.Property(panel)

        # ── Overview stats ────────────────────────────────────────────────────
        avg_form    = sum(form_scores) / total
        avg_posture = sum(posture_scores) / len(posture_scores) if posture_scores else 0.0
        avg_reps    = sum(rep_counts) / len(rep_counts) if rep_counts else 0.0
        critical_count  = issue_severities.count("critical")
        moderate_count  = issue_severities.count("moderate")
        minor_count     = issue_severities.count("minor")
        total_issues    = len(issue_labels)

        lines.append("## Overview")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Videos analyzed | **{total}** |")
        lines.append(f"| Avg form score  | **{avg_form:.1f} / 100** — {_score_label(avg_form)} |")
        lines.append(f"| Avg posture score | **{avg_posture:.1f} / 100** |")
        lines.append(f"| Avg reps per video | **{avg_reps:.1f}** |")
        lines.append(f"| Total issues flagged | **{total_issues}** ({critical_count} critical, {moderate_count} moderate, {minor_count} minor) |")
        if error_count:
            lines.append(f"| Processing errors | {error_count} |")
        lines.append("")

        # ── Form score bar ────────────────────────────────────────────────────
        lines.append("### Avg Form Score")
        lines.append(
            f"`{_bar(avg_form, 100)}`  **{avg_form:.1f}**"
        )
        lines.append("")

        # ── Grade distribution ────────────────────────────────────────────────
        lines.append("## Grade Distribution")
        lines.append("")
        grade_counts = Counter(grades)
        grade_total  = sum(grade_counts.values()) or 1
        for g in ["A", "B", "C", "D", "F"]:
            count = grade_counts.get(g, 0)
            pct   = count / grade_total * 100
            lines.append(
                f"**{g}**  `{_bar(count, grade_total)}`  {count} video(s)  ({pct:.0f}%)"
            )
        lines.append("")

        # ── Per-exercise breakdown ────────────────────────────────────────────
        lines.append("## Per-Exercise Form Score")
        lines.append("")
        if exercise_scores:
            max_avg = max(
                sum(s) / len(s) for s in exercise_scores.values()
            )
            lines.append("| Exercise | Avg Score | Videos | Rating |")
            lines.append("|----------|-----------|--------|--------|")
            for ex, scores in sorted(exercise_scores.items()):
                avg  = sum(scores) / len(scores)
                name = ex.replace("_", " ").title()
                bar  = _bar(avg, 100, width=12)
                lines.append(
                    f"| {name} | `{bar}` {avg:.1f} | {len(scores)} | {_score_label(avg)} |"
                )
        else:
            lines.append("_No exercise data available._")
        lines.append("")

        # ── Issue severity breakdown ──────────────────────────────────────────
        lines.append("## Form Issues by Severity")
        lines.append("")
        if total_issues:
            sev_data = [
                ("Critical", critical_count, "Injury risk — fix immediately"),
                ("Moderate", moderate_count, "Performance loss — address soon"),
                ("Minor",    minor_count,    "Refinement — good to improve"),
            ]
            lines.append("| Severity | Count | Bar | Meaning |")
            lines.append("|----------|-------|-----|---------|")
            for label, count, meaning in sev_data:
                bar = _bar(count, total_issues, width=12)
                lines.append(f"| **{label}** | {count} | `{bar}` | {meaning} |")
            lines.append("")

            # Top 5 issues with their dominant severity
            lines.append("### Top 5 Most Flagged Issues")
            lines.append("")
            issue_counts = Counter(issue_labels)
            lines.append("| Rank | Issue | Count | Dominant Severity |")
            lines.append("|------|-------|-------|-------------------|")
            for rank, (issue, count) in enumerate(issue_counts.most_common(5), 1):
                display  = issue.replace("_", " ").title()
                sevs     = severity_by_issue[issue]
                dominant = Counter(sevs).most_common(1)[0][0].title()
                lines.append(f"| {rank} | {display} | {count} | {dominant} |")
        else:
            lines.append("_No form issues recorded. Great technique across the board!_")
        lines.append("")

        # ── Strengths ─────────────────────────────────────────────────────────
        lines.append("## Most Common Strengths")
        lines.append("")
        if all_strengths:
            strength_counts = Counter(all_strengths)
            max_s = strength_counts.most_common(1)[0][1]
            for strength, count in strength_counts.most_common(5):
                bar = _bar(count, max_s, width=10)
                lines.append(f"- `{bar}` **{strength}** ({count}x)")
        else:
            lines.append("_No strengths data yet._")
        lines.append("")

        # ── Priority fix spotlight ────────────────────────────────────────────
        if all_priority_fixes:
            lines.append("## Most Recommended Fixes")
            lines.append("")
            fix_counts = Counter(all_priority_fixes)
            for fix, count in fix_counts.most_common(3):
                lines.append(f"> **#{list(fix_counts.keys()).index(fix) + 1}** ({count}x)  {fix}")
            lines.append("")

        # ── Needs attention ───────────────────────────────────────────────────
        low_performers = sorted(
            all_coaching_summaries, key=lambda x: x[1]
        )[:3]

        if low_performers:
            lines.append("## Needs Attention")
            lines.append(
                "_Videos with the lowest form scores — prioritize coaching here._"
            )
            lines.append("")
            for filename, score, grade, summary in low_performers:
                lines.append(f"**{filename}** — Grade **{grade}** ({score:.0f}/100)")
                lines.append(f"> {summary}")
                lines.append("")

        # ── Top performers ────────────────────────────────────────────────────
        top_performers = sorted(
            all_coaching_summaries, key=lambda x: x[1], reverse=True
        )[:3]

        if top_performers:
            lines.append("## Top Performers")
            lines.append(
                "_Videos with the highest form scores._"
            )
            lines.append("")
            for filename, score, grade, summary in top_performers:
                lines.append(f"**{filename}** — Grade **{grade}** ({score:.0f}/100)")
                lines.append(f"> {summary}")
                lines.append("")

        # ── Footer ────────────────────────────────────────────────────────────
        lines.append("---")
        lines.append(
            "_RepImprov · TwelveLabs Pegasus 1.2 · FiftyOne Plugin · "
            "Video Understanding AI Hackathon 2025_"
        )

        panel.str(
            "content",
            label="",
            default="\n".join(lines),
            view=types.MarkdownView(),
        )

        return types.Property(panel)


def register(plugin):
    plugin.register(RepImprovDashboard)
