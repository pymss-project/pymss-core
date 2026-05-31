import gzip
import logging
import os
import shutil
import sys
from datetime import datetime

MAX_LOG = 100
LOG_DIR = ".logs"
LOG_ENV_NAME = "PYMSS_LOG_FILE"


def _safe_relpath(pathname):
    if not pathname:
        return pathname

    normalized = os.path.normpath(pathname)
    if os.name == "nt" and normalized.startswith("\\\\?\\"):
        normalized = normalized[4:]

    try:
        return os.path.relpath(normalized)
    except ValueError:
        return normalized


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DBG": "\033[1;36m",
        "INF": "\033[1;32m",
        "WAR": "\033[1;33m",
        "ERR": "\033[1;31m",
        "CRI": "\033[1;35m",
    }
    MESSAGE_COLORS = {
        "DBG": "\033[36m",
        "INF": "\033[32m",
        "WAR": "\033[33m",
        "ERR": "\033[31m",
        "CRI": "\033[35m",
    }
    RESET = "\033[0m"
    LEVEL_MAP = {
        "DEBUG": "DBG",
        "INFO": "INF",
        "WARNING": "WAR",
        "ERROR": "ERR",
        "CRITICAL": "CRI",
    }

    def __init__(self, enable_color=True):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(pathname)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S",
        )
        self.enable_color = enable_color

    def format(self, record):
        record.pathname = _safe_relpath(record.pathname)
        original_levelname = record.levelname
        original_msg = record.msg

        short_level = self.LEVEL_MAP.get(record.levelname, record.levelname[:3])
        if self.enable_color:
            level_color = self.COLORS.get(short_level, "")
            message_color = self.MESSAGE_COLORS.get(short_level, "")
            record.levelname = f"{level_color}{short_level}{self.RESET}"
            record.msg = f"{message_color}{record.getMessage()}{self.RESET}"
            record.args = ()
        else:
            record.levelname = short_level

        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname
            record.msg = original_msg


class FileFormatter(logging.Formatter):
    LEVEL_MAP = ColorFormatter.LEVEL_MAP

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(pathname)s:%(lineno)d | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record):
        record.pathname = _safe_relpath(record.pathname)
        original_levelname = record.levelname
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname[:3])
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _supports_color(stream):
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("PYMSS_FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _compress_log_file(path):
    gz_path = f"{path}.gz"
    if os.path.exists(gz_path):
        return
    try:
        with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(path)
    except OSError:
        pass


def _parse_log_time(filename):
    stem = filename
    if stem.endswith(".gz"):
        stem = stem[:-3]
    if stem.endswith(".log"):
        stem = stem[:-4]
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(stem, fmt)
        except ValueError:
            continue
    return datetime.min


def manage_log_files(log_dir, max_log):
    try:
        log_files = [
            filename
            for filename in os.listdir(log_dir)
            if filename.endswith(".log") or filename.endswith(".log.gz")
        ]
    except OSError:
        return

    current_log = os.environ.get(LOG_ENV_NAME)
    for filename in log_files:
        path = os.path.join(log_dir, filename)
        if filename.endswith(".log") and path != current_log:
            _compress_log_file(path)

    try:
        log_files = [
            filename
            for filename in os.listdir(log_dir)
            if filename.endswith(".log") or filename.endswith(".log.gz")
        ]
    except OSError:
        return

    log_files = sorted(log_files, key=_parse_log_time)
    while len(log_files) > max_log:
        oldest_file = log_files.pop(0)
        path = os.path.join(log_dir, oldest_file)
        if path == current_log:
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def _get_log_path(log_dir):
    log_path = os.environ.get(LOG_ENV_NAME)
    if log_path:
        return log_path

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))
    os.environ[LOG_ENV_NAME] = log_path
    return log_path


def set_log_level(logger, level):
    if hasattr(logger, "console_handler"):
        logger.console_handler.setLevel(level)


def get_separation_logger(
    console_level=logging.INFO,
    enable_file_log=False,
    max_log=MAX_LOG,
    log_dir=LOG_DIR,
    enable_color=None,
):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if hasattr(logger, "console_handler"):
        logger.console_handler.setLevel(console_level)
        if enable_file_log and not hasattr(logger, "file_handler"):
            log_path = _get_log_path(log_dir)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(FileFormatter())
            logger.addHandler(file_handler)
            logger.file_handler = file_handler
            manage_log_files(log_dir, max_log)
        return logger

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    if enable_color is None:
        enable_color = _supports_color(console_handler.stream)
    console_handler.setFormatter(ColorFormatter(enable_color=enable_color))
    logger.addHandler(console_handler)
    logger.console_handler = console_handler

    if enable_file_log:
        log_path = _get_log_path(log_dir)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)
        logger.file_handler = file_handler
        manage_log_files(log_dir, max_log)

    return logger
