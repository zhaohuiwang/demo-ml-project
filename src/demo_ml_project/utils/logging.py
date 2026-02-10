

# # project-root(demo-ml-project)/src/demo_ml_project/utils/logging.py
# src/demo_ml_project/utils/logging.py

import logging
import logging.config
from pathlib import Path
from typing import Literal, Optional

LOG_LEVEL = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def configure_logging(
    log_dir: Optional[Path | str] = None,
    log_file: str = "training.log",
    default_level: LOG_LEVEL = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    json_format: bool = False,
) -> None:
    """
    Configure application logging with console + optional rotating file handler.

    Should be called once early in the main script (before any logging happens).

    Parameters
    ----------
    log_dir : Path or str, optional
        Directory for log files. Created if it doesn't exist.
    log_file : str
        Name of the main log file
    default_level : LOG_LEVEL
        Root logger level
    max_bytes : int
        Max size per log file before rotation
    backup_count : int
        Number of backup log files to keep
    json_format : bool
        Whether to use JSON formatting for file logs
    """
    if log_dir is None:
        log_dir = Path("logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
    }

    if log_file:
        file_handler = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json" if json_format else "standard",
            "filename": str(log_path),
            "maxBytes": max_bytes,
            "backupCount": backup_count,
            "encoding": "utf-8",
        }
        handlers["file"] = file_handler

    formatters = {
        "standard": {
            "()": "logging.Formatter",
            "format": "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "console": {
            "()": "logging.Formatter",
            "format": "%(levelname)-8s %(name)-18s %(message)s",
            "datefmt": "%H:%M:%S",
        },
    }

    if json_format:
        formatters["json"] = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d %(funcName)s",
        }

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "root": {
            "level": default_level,
            "handlers": ["console"] + (["file"] if log_file else []),
        },
    })


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance after global configuration has been applied.
    """
    if name is None:
        name = __name__
    return logging.getLogger(name)

# import logging
# import logging.config
# from pathlib import Path
# from typing import Optional, Literal

# LOG_LEVELS = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# def configure_logging(
#     log_dir: Path | str | None = None,
#     default_level: LOG_LEVELS = "INFO",
#     log_file: str = "app.log",
#     max_bytes: int = 10 * 1024 * 1024,        # 10 MB
#     backup_count: int = 5,
#     json_format: bool = False,
# ) -> None:
#     """
#     One-time global logging configuration.
#     Call this early (preferably in main script).
#     """
#     if log_dir is None:
#         log_dir = Path("logs")
#     log_dir = Path(log_dir)
#     log_dir.mkdir(parents=True, exist_ok=True)

#     log_path = log_dir / log_file

#     handlers = {
#         "console": {
#             "class": "logging.StreamHandler",
#             "level": "DEBUG",
#             "formatter": "console",
#             "stream": "ext://sys.stdout",
#         },
#     }

#     if log_file:
#         handlers["file"] = {
#             "class": "logging.handlers.RotatingFileHandler",
#             "level": "DEBUG",
#             "formatter": "json" if json_format else "standard",
#             "filename": str(log_path),
#             "maxBytes": max_bytes,
#             "backupCount": backup_count,
#             "encoding": "utf-8",
#         }

#     formatters = {
#         "standard": {
#             "()": "logging.Formatter",
#             "format": "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
#             "datefmt": "%Y-%m-%d %H:%M:%S",
#         },
#         "console": {
#             "()": "logging.Formatter",
#             "format": "%(levelname)-8s %(name)-18s %(message)s",
#             "datefmt": "%H:%M:%S",
#         },
#     }

#     if json_format:
#         formatters["json"] = {
#             "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
#             "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d %(funcName)s",
#         }

#     logging.config.dictConfig({
#         "version": 1,
#         "disable_existing_loggers": False,
#         "formatters": formatters,
#         "handlers": handlers,
#         "root": {
#             "level": default_level,
#             "handlers": ["console"] + (["file"] if log_file else []),
#         },
#     })


# def get_logger(name: str | None = None) -> logging.Logger:
#     """Get logger after configuration has been set"""
#     if name is None:
#         name = __name__
#     return logging.getLogger(name)