import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

def logg(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
