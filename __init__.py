"""
RepImprov — AI-powered workout form analyzer
FiftyOne plugin using TwelveLabs Pegasus 1.2
"""

import logging

logger = logging.getLogger(__name__)


def register(plugin):
    try:
        from .workout_operator import AnalyzeWorkoutForm
        from .panel import RepImprovDashboard

        plugin.register(AnalyzeWorkoutForm)
        logger.info("Registered operator: AnalyzeWorkoutForm")

        plugin.register(RepImprovDashboard)
        logger.info("Registered panel: RepImprovDashboard")

        logger.info("RepImprov plugin registered successfully")

    except Exception as exc:
        logger.exception("RepImprov plugin registration failed: %s", exc)
        raise
