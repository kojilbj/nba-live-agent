"""Structured logging setup for nba_live_agent.

Library modules (nba_client, x_client, agent) only call
logging.getLogger(__name__) — they never attach handlers themselves.
configure_logging() is the one place handlers get wired up, called by the
CLI entry point so importing the package elsewhere doesn't have side
effects.
"""

import copy
import logging

DEFAULT_LOG_FILE = "nba_live_agent.log"
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class _NoTracebackFormatter(logging.Formatter):
    """Strips exc_info/stack_info before formatting so logger.exception()
    calls don't dump a full traceback to the console — the traceback still
    reaches the log file via the regular formatter.

    Formats a copy of the record rather than mutating it in place: the same
    LogRecord instance is passed to every handler on the logger, so clearing
    exc_info directly would also erase the traceback before the file
    handler's formatter gets to see it.
    """

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return super().format(record)


def configure_logging(verbose: bool = False, log_file: str | None = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configures the "nba_live_agent" logger. Console output stays at
    WARNING by default so it doesn't compete with the CLI's own print()
    output; --verbose raises it to DEBUG. The log file (if given) always
    captures DEBUG and up, independent of verbosity, so full detail is
    available for troubleshooting either way.

    Safe to call more than once — clears and rebuilds handlers each time
    rather than tracking whether it's "already configured".
    """
    logger = logging.getLogger("nba_live_agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_handler.setFormatter(_NoTracebackFormatter(_LOG_FORMAT))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
