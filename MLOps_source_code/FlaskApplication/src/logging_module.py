
import os
import logging
from logging.handlers import RotatingFileHandler
from flask.logging import default_handler
from flask import has_request_context, request
# from newrelic.agent import NewRelicContextFormatter

# log_format = logging.Formatter(
#     '[%(asctime)s] {%(levelname)s %(name)s %(threadName)s} : %(message)s')

if os.environ.get("logger_name") is None:
    os.environ["logger_name"] = "flask_app_logger"

logger_name = os.environ.get("logger_name")
debug_info_log_file_name = 'logs/debug.log'
warn_error_critical_log_file_name = 'logs/error.log'


MAX_BYTES = 1000000
BACKUP_COUNT = 10
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)


class RequestFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
        else:
            record.url = None
            record.remote_addr = None

        return super().format(record)


formatter = RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s %(levelname)s '
    '{%(name)s %(threadName)s} : %(message)s'
)


class DEBUG_FILTER:
    def __init__(self, level=logging.INFO):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


class ERROR_FILTER:
    def __init__(self, level=logging.INFO):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno > self.__level


def set_stream_handler(log_level):
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)


# def set_new_relic_stream_handler(log_level):
#     stream_handler = logging.StreamHandler()
#     formatter = NewRelicContextFormatter()
#     stream_handler.setFormatter(formatter)
#     stream_handler.setLevel(log_level)
#     logger.addHandler(stream_handler)


def set_file_handler(log_file_name, log_level, filter_=DEBUG_FILTER()):
    file_handler = RotatingFileHandler(
        log_file_name, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    file_handler.addFilter(filter_)
    logger.addHandler(file_handler)


set_stream_handler(logging.DEBUG)
# set_new_relic_stream_handler(logging.DEBUG)
set_file_handler(warn_error_critical_log_file_name,
                 logging.WARN, ERROR_FILTER())
set_file_handler(debug_info_log_file_name, logging.DEBUG, DEBUG_FILTER())


