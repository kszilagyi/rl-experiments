import logging
from pathlib import Path

FORMAT = '%(asctime)-15s %(message)s'
OUTPUT_DIR = Path('./job_output').resolve()
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(format=FORMAT, filename=str(OUTPUT_DIR / 'logs.txt'))
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter(FORMAT)
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def logg(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
