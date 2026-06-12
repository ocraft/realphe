from importlib import resources
import logging
import logging.config


_LOGGING_INITIALIZED = False


def _init_logging():
    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED:
        return

    log_ini_path = resources.files("realphe.env").joinpath("log.ini")

    logging.config.fileConfig(log_ini_path, disable_existing_loggers=False)

    _LOGGING_INITIALIZED = True


def get_log(name: str) -> logging.Logger:
    _init_logging()
    return logging.getLogger(name)


def format_duration(seconds: float) -> str:

    units = [
        (3600, 'h'),
        (60, 'min'),
        (1, 's'),
        (1e-3, 'ms'),
        (1e-6, 'μs'),
        (1e-9, 'ns'),
    ]

    for scale, unit in units:

        if seconds >= scale:
            value = int(seconds / scale)
            return f'{value}{unit}'

    return '0ns'
