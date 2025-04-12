import logging
import traceback

logger = logging.getLogger(__name__)


def trace_error():
    logger.error(traceback.format_exc())
