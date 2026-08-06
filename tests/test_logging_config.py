import logging

from nba_live_agent.logging_config import configure_logging


def test_configure_logging_default_console_level_is_warning():
    logger = configure_logging(verbose=False, log_file=None)
    assert logger.name == "nba_live_agent"
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.WARNING


def test_configure_logging_verbose_sets_console_level_to_debug():
    logger = configure_logging(verbose=True, log_file=None)
    assert logger.handlers[0].level == logging.DEBUG


def test_configure_logging_adds_file_handler_at_debug(tmp_path):
    log_file = tmp_path / "test.log"
    logger = configure_logging(verbose=False, log_file=str(log_file))

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].level == logging.DEBUG
    assert log_file.exists()


def test_configure_logging_without_log_file_has_no_file_handler():
    logger = configure_logging(verbose=False, log_file=None)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_configure_logging_is_idempotent():
    configure_logging(verbose=False, log_file=None)
    configure_logging(verbose=True, log_file=None)
    logger = logging.getLogger("nba_live_agent")
    assert len(logger.handlers) == 1
