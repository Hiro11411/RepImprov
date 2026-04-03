"""
FiftyOne panel: repimprov_dashboard

Displays aggregate coaching metrics across the analyzed dataset:
  - Average form score
  - Per-exercise form score breakdown
  - Top 5 most frequent form issues
  - Grade distribution (A/B/C/D/F)
  - Most common strengths
"""

import logging
from collections import Counter, defaultdict

import fiftyone.operators as foo
import fiftyone.operators.types as types

logger = logging.getLogger(__name__)


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

        # ── Gather data ───────────────────────────────────────────────────────
        form_scores = []
        posture_scores = []
        grades = []
        exercise_scores = defaultdict(list)
        issue_labels = []
        all_strengths = []

        for sample in dataset.iter_samples():
            # Form score
            score = sample.get_field("form_score")
            if score is not None:
                form_scores.append(float(score))

            # Posture score
            ps = sample.get_field("posture_score")
            if ps is not None:
                posture_scores.append(float(ps))

            # Grade
            grade = sample.get_field("form_grade")
            if grade and grade != "ERROR":
                grades.append(grade)

            # Exercise breakdown
            exercise = sample.get_field("exercise_detected")
            if exercise and score is not None:
                exercise_scores[exercise].append(float(score))

            # Form issues
            form_issues = sample.get_field("form_issues")
            if form_issues and hasattr(form_issues, "detections"):
                for det in form_issues.detections:
                    if det.label:
                        issue_labels.append(det.label)

            # Strengths
            strengths = sample.get_field("strengths")
            if strengths and isinstance(strengths, list):
                all_strengths.extend(strengths)

        total = len(form_scores)

        # ── Build display text ────────────────────────────────────────────────
        lines = []

        lines.append("# RepImprov Dashboard\n")

        if total == 0:
            lines.append(
                "_No analyzed samples found. Run **RepImprov: Analyze Workout Form** first._"
            )
            panel.str("content", label="", default="\n".join(lines))
            return types.Property(panel)

        # Overview
        avg_form = sum(form_scores) / total
        avg_posture = (sum(posture_scores) / len(posture_scores)) if posture_scores else 0.0
        lines.append(f"**Samples analyzed:** {total}")
        lines.append(f"**Avg Form Score:** {avg_form:.1f} / 100")
        lines.append(f"**Avg Posture Score:** {avg_posture:.1f} / 100")
        lines.append("")

        # Grade distribution
        lines.append("## Grade Distribution")
        grade_counts = Counter(grades)
        for g in ["A", "B", "C", "D", "F"]:
            count = grade_counts.get(g, 0)
            bar = "█" * count
            lines.append(f"  {g}: {bar} ({count})")
        lines.append("")

        # Per-exercise breakdown
        lines.append("## Per-Exercise Avg Form Score")
        if exercise_scores:
            for ex, scores in sorted(exercise_scores.items()):
                avg = sum(scores) / len(scores)
                lines.append(f"  {ex.replace('_', ' ').title():<20} {avg:.1f}  (n={len(scores)})")
        else:
            lines.append("  No exercise data yet.")
        lines.append("")

        # Top 5 form issues
        lines.append("## Top 5 Most Frequent Form Issues")
        if issue_labels:
            issue_counts = Counter(issue_labels)
            for issue, count in issue_counts.most_common(5):
                display = issue.replace("_", " ").title()
                lines.append(f"  {display:<30} {count}x")
        else:
            lines.append("  No form issues recorded yet.")
        lines.append("")

        # Most common strengths
        lines.append("## Most Common Strengths")
        if all_strengths:
            strength_counts = Counter(all_strengths)
            for strength, count in strength_counts.most_common(5):
                lines.append(f"  ✓ {strength} ({count}x)")
        else:
            lines.append("  No strengths data yet.")

        panel.str(
            "content",
            label="",
            default="\n".join(lines),
            view=types.MarkdownView(),
        )

        return types.Property(panel)


def register(plugin):
    plugin.register(RepImprovDashboard)
