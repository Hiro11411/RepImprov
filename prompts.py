"""
Prompt templates for TwelveLabs Pegasus form analysis.
"""

# ── Exercise context blocks ───────────────────────────────────────────────────

_AUTO_EXERCISE_CONTEXT = """\
Before scoring anything, observe the video carefully:
- What is the athlete's body orientation? (standing upright, horizontal face-down, lying on back, hinged forward)
- Is equipment involved? (barbell, bodyweight only, dumbbells, bench, pull-up bar)
- What is the primary movement pattern? (descending/ascending with knees bent, hinging at hips, horizontal push, vertical push)

Use those observations to identify the exercise. Do not guess — look at what the athlete is actually doing.

How to distinguish common exercises:
- pushup: athlete face-down, horizontal, hands on floor, pushes body up — NOT standing
- squat: athlete standing upright, feet shoulder-width, descends by bending knees
- deadlift: athlete standing, picks a loaded bar off the floor by hinging at the hips
- bench_press: athlete lying on their back on a bench, pressing a bar upward
- shoulder_press: athlete standing or seated, pressing a bar or dumbbells overhead\
"""


def confirmed_exercise_context(exercise: str) -> str:
    """Return an exercise context block that treats the exercise as already confirmed."""
    return (
        f"Exercise: {exercise} (confirmed — do not re-identify).\n"
        f"Skip identification entirely and evaluate form for {exercise} only.\n"
        f"Set exercise_detected to \"{exercise}\" in your output."
    )


# ── Per-exercise rep counting instructions (used by REP_COUNT_PROMPT) ────────
#
# Each entry gives Pegasus a precise visual anchor for counting, an explicit
# "count when you see X" trigger, and a list of things to exclude.
# Always count at the TOP of the movement — it is the most visually distinct
# position for every exercise here.

_REP_COUNT_INSTRUCTIONS = {
    "squat": """\
POSITIONS
  Start/top: athlete standing upright, hips and knees fully extended, bar across upper back.
  Bottom:    hips descended until the hip crease is at or below the top of the knees.

COUNT ONE REP each time the athlete rises from the bottom and reaches full standing (hips
and knees fully extended at the top). Watch the hips — they are your count trigger.

DO NOT COUNT
  - The initial walkout or setup before the first descent.
  - Any descent where the athlete stops above parallel (hip crease above knees) — that is
    a partial rep and does not count.
  - Re-racking the bar after the set is finished.
  - A rep the athlete clearly fails and does not stand back up from.\
""",

    "deadlift": """\
POSITIONS
  Start: bar on the floor, athlete hinged at hips, back flat, hands gripping the bar.
  Top:   athlete standing tall with hips and knees fully extended, bar at hip level.

COUNT ONE REP each time the athlete reaches full lockout at the top — hips fully forward,
standing upright, shoulders behind the bar. Watch the hips driving through to extension.

DO NOT COUNT
  - Setup adjustments, re-gripping, or hip-hinge practice swings before the first pull.
  - Any rep that does not reach full hip and knee extension at the top.
  - Touch-and-go sets: each touch-and-go cycle (bar touches floor, pulls back up) = one rep.
  - A missed lift where the bar does not reach lockout.\
""",

    "bench_press": """\
POSITIONS
  Start/top: bar at full arm extension above the chest, elbows locked out.
  Bottom:    bar touching or lightly touching the chest (or just above it).

COUNT ONE REP each time the bar is pressed back to full arm extension at the top.
Watch the elbows — count when they lock out.

DO NOT COUNT
  - Unracking the bar from the uprights or handing it off.
  - Racking the bar after the set.
  - Any rep where the elbows do not lock out at the top (failed rep).
  - A rep where a spotter is clearly taking the majority of the weight.\
""",

    "pushup": """\
POSITIONS
  Start/top: arms fully extended, body in a straight line from head to heels, hands on floor.
  Bottom:    chest at or near the floor, elbows bent.

COUNT ONE REP each time the athlete pushes back up to full arm extension at the top.
Watch the elbows straightening — that is your count trigger.

DO NOT COUNT
  - The initial lowering from standing or kneeling into the starting plank position.
  - Any rep where the arms do not fully straighten at the top.
  - Knee pushups count the same way — count at the top when elbows lock out.
  - Rest pauses at the bottom without completing the push back up.\
""",

    "shoulder_press": """\
POSITIONS
  Start: bar or dumbbells at shoulder level, elbows roughly in front of the body.
  Top:   weight pressed to full overhead lockout, arms fully extended.

COUNT ONE REP each time the weight reaches full overhead lockout — arms straight, weight
directly overhead. Watch the elbows straightening fully overhead.

DO NOT COUNT
  - Initial unracking or cleaning the weight to the start position.
  - Racking after the set.
  - Reps where the elbows do not lock out overhead.
  - Push-press reps (where the legs assist) are still valid — count them if lockout is reached.\
""",

    "pull_up": """\
POSITIONS
  Start/bottom: dead hang from the bar, arms fully extended, feet off the floor.
  Top:          chin clearly above the top of the bar.

COUNT ONE REP each time the chin rises above the bar. The chin must visibly clear
the bar — if it only reaches bar height, do not count it.

DO NOT COUNT
  - Jumping up to grab the bar or swinging into position before the first pull.
  - Any pull where the chin does not clearly rise above the bar.
  - Negatives only (athlete lowers but never pulls up) — those are not reps.
  - Kipping is valid as long as the chin clears the bar — count those reps.\
""",

    "crunch": """\
POSITIONS
  Start: lying on back, knees bent, upper back and shoulder blades flat on the floor.
  Top:   shoulder blades fully lifted off the floor, peak core contraction reached.

COUNT ONE REP each time the shoulder blades fully lift off the floor and the athlete
reaches peak contraction. Watch the shoulder blades — they are your count trigger.

DO NOT COUNT
  - Head nods or neck lifts where the shoulder blades stay on the floor.
  - Any movement that does not result in the shoulder blades visibly clearing the floor.
  - Sit-ups where the athlete comes fully upright — that is a different exercise; count
    only if it matches the crunch pattern (partial curl, not a full sit-up).\
""",
}

REP_COUNT_PROMPT = """You are an expert at counting exercise repetitions in workout videos.

Exercise: {exercise} (confirmed)

--- HOW TO COUNT REPS FOR THIS EXERCISE ---
{instructions}

--- GENERAL RULES ---
- Watch the full video through once before you start counting. Do not count as you go on
  the first pass.
- When you are unsure whether a borderline movement qualifies as a rep, do NOT count it.
  Undercounting by one is far better than overcounting.
- If the video is very short (under 3 seconds), the answer may legitimately be 0.
- Rest pauses between reps do not affect the count — only complete cycles count.

Return ONLY valid JSON with no markdown backticks:
{{"rep_count": <integer>, "confidence": <0-100 integer>}}
"""


# ── Main analysis prompts ─────────────────────────────────────────────────────

FORM_ASSESSMENT_PROMPT = """You are an expert strength and conditioning coach analyzing a workout video.

{exercise_context}

Sensitivity level: {sensitivity}
- strict: flag even minor deviations from textbook form
- moderate: flag only meaningful deviations that affect performance or safety
- lenient: flag only significant safety risks

Exercise-specific form checks to evaluate:
- squat: knee cave (valgus collapse), excessive forward lean, depth (below parallel), heel rise, back rounding
- deadlift: lower back rounding, bar drift away from body, hip hinge initiation, lockout at top
- bench_press: elbow flare past 90°, bar path consistency, wrist hyperextension, leg drive
- pushup: hip sag, elbow angle (should be ~45° from torso), depth at bottom, head position, core stability
- shoulder_press: lower back arch, elbow position, bar path deviation
- pull_up: incomplete lockout at bottom, chin not clearing bar, excessive kipping, shoulder shrug at top
- crunch: pulling on neck, lower back lifting off floor, incomplete range of motion, fast uncontrolled tempo

Write one sentence describing what you see, then output ONLY valid JSON with no markdown backticks:
{{"form_score": 78, "form_grade": "C", "exercise_detected": "<exercise name>", "verdict": "<short honest phrase>", "confidence": 85}}
"""

POSTURE_ANALYSIS_PROMPT = """You are an expert strength and conditioning coach analyzing a {exercise_type} video.

Sensitivity level: {sensitivity}
- strict: flag even minor deviations from textbook form
- moderate: flag only meaningful deviations that affect performance or safety
- lenient: flag only significant safety risks

Watch the full video, then go back and identify specific moments where form breaks down.
Only report issues you can pinpoint to a specific timestamp — do not report general impressions without a time reference.

Exercise-specific issues to look for:
- squat: knee cave (valgus collapse), excessive forward lean, insufficient depth, heel rise, rounded upper back
- deadlift: lower back rounding, bar drift away from body, early hip rise, incomplete lockout, neck hyperextension
- bench_press: elbow flare past 90°, inconsistent bar path, wrist hyperextension, leg drive loss, butt lift
- pushup: hip sag, flared elbows (past 45°), incomplete depth, forward head position, collapsing core
- shoulder_press: lower back arch, elbow flare, bar path deviation, incomplete lockout
- pull_up: incomplete lockout at bottom, chin not clearing bar, excessive kipping, shoulder shrug at top
- crunch: pulling on neck with hands, lower back lifting off floor, incomplete range of motion, fast uncontrolled descent

For each issue found, provide:
- timestamp_seconds: the exact moment it occurs (float) — only include if you can point to a specific frame
- problem: short snake_case name of the issue (e.g. "hip_sag", "elbow_flare", "lower_back_rounding")
- severity: "critical" (injury risk), "moderate" (performance loss), or "minor" (refinement needed)
- fix: one specific, actionable coaching cue to correct it

Return a JSON object with an "issues" array. If no issues are visible, return an empty array.
Do not fabricate issues — it is fine and honest to return fewer issues or an empty array.

Output ONLY valid JSON with no markdown backticks:
{{"issues": [{{"timestamp_seconds": 2.1, "problem": "<issue_name>", "severity": "<critical|moderate|minor>", "fix": "<one actionable cue>"}}]}}
"""

STRENGTHS_PROMPT = """You are an expert strength and conditioning coach analyzing a {exercise_type} video.

Sensitivity level: {sensitivity}

Watch the full video and identify specific things the athlete does well.
Only report strengths you actually observed — be specific, not generic.

Exercise-specific positives to look for:
- squat: consistent depth, neutral spine, controlled descent, knees tracking toes, braced core, even tempo
- deadlift: tight lats, proper hip hinge setup, bar staying close to body, strong lockout, neutral neck
- bench_press: consistent bar path, stable arch, tight grip, controlled descent, active leg drive
- pushup: straight body line from head to heel, full depth, controlled tempo, stable shoulders, proper elbow angle
- shoulder_press: strong core brace, full lockout, controlled descent, stable base
- pull_up: full lockout at bottom, chin clearly over bar, controlled descent, engaged scapulae
- crunch: controlled tempo, hands not pulling neck, lower back staying grounded, full contraction at top

Provide:
1. strengths: 2-5 specific things the athlete does well — reference what you saw, not generic praise
2. top_priority_fix: the single most important correction that would have the biggest impact (one sentence)
3. coaching_summary: 2-3 sentences of encouraging but honest coaching — acknowledge what is working and give clear direction

Output ONLY valid JSON with no markdown backticks:
{{"strengths": ["<specific observed strength>", "<specific observed strength>"], "top_priority_fix": "<most impactful correction>", "coaching_summary": "<2-3 sentences of specific coaching feedback>"}}
"""
