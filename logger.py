#Create a reusable logger for the entire project

import logging
import os

def get_logger(name: str):
    # Create logs/ folder if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    # Save logs to a file named after the module
    handler = logging.FileHandler(f"logs/{name}.log", mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log