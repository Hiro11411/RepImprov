"""
RepImprov — AI-powered workout form analyzer
FiftyOne plugin using TwelveLabs Pegasus 1.2
"""

import fiftyone.operators as foo

from .workout_operator import AnalyzeWorkoutForm
from .panel import RepImprovDashboard


def register(p):
    plugin.register(AnalyzeWorkoutForm)
    plugin.register(RepImprovDashboard)
