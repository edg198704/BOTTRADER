# bot_core/__init__.py
import logging
import os

# Configure basic logging for the package
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
