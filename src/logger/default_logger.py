import logging
import os

# disable matplotlib and PIL logging
logger_blocklist = [
    "matplotlib",
    "PIL",
    "h5py",
    "wandb",
    "git",
]

for module in logger_blocklist:
    logging.getLogger(module).setLevel(logging.WARNING)

def setup_logging(filename=None, level=logging.WARNING):
    # Central logging configuration
    logging.basicConfig(
        level=level,  # Set the base level
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )

    if filename is not None:
        add_file_handler(logging.getLogger(), filename)

def create_file_handler(filename):
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
    return file_handler

def add_file_handler(logger, filename):
    file_handler = create_file_handler(filename)
    logger.addHandler(file_handler)

def update_file_handler(logger, filename):
    # Remove existing FileHandler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()  # Close the old file handler

    new_file_handler = create_file_handler(filename)
    logger.addHandler(new_file_handler)

def update_file_handler_root(filename):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = create_file_handler(filename)
    root_logger.addHandler(file_handler)