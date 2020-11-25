import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, filename='./logs.txt')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter(FORMAT)
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def logg(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
