import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.NOTSET)
if __name__ == '__main__':
    logging.error("错误")
    logging.debug("dejjjjjjjjjbug")
    logging.warning("warning")
    logging.info("info")
    logging.critical("critical")

    c = defaultdict(Counter)

    b = {}
    print(c)
