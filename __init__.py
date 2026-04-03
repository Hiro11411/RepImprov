import logging
logger = logging.getLogger(__name__)

def register(plugin):
    logger.info("RepImprov: starting registration")
    
    from .workout_operator import AnalyzeWorkoutForm
    plugin.register(AnalyzeWorkoutForm)
    logger.info("RepImprov: operator registered")
    
    try:
        from .panel import RepImprovDashboard
        plugin.register(RepImprovDashboard)
        logger.info("RepImprov: panel registered")
    except Exception as e:
        logger.error(f"RepImprov: panel failed to register: {e}")
        # operator still works without panel