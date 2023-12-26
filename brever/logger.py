import logging
import sys


def set_logger(log_file=None, distributed=False, rank=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    prefix = '%(asctime)s [%(levelname)s:%(module)s]'
    if distributed:
        if rank is None:
            raise ValueError('must provide rank when distributed=True')
        f = ContextFilter(rank)
        logger.addFilter(f)
        formatter = logging.Formatter(prefix + ' [rank %(rank)s] %(message)s')
    else:
        formatter = logging.Formatter(prefix + ' %(message)s')

    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


class ContextFilter(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True
