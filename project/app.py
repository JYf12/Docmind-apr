import sys
import os
import logging
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Suppress OTel "Failed to detach context" warning caused by generator/context interaction.
# Tracing is unaffected.
# Known bug: https://github.com/open-telemetry/opentelemetry-python/issues/2606

# Configure logging to output to console
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class _SuppressOtelDetachWarning(logging.Filter):
    def filter(self, record):
        return "Failed to detach context" not in record.getMessage()

logging.getLogger("opentelemetry.context").addFilter(_SuppressOtelDetachWarning())

from ui.css import custom_css
from ui.gradio_app import create_gradio_ui

if __name__ == "__main__":
    print("\n🔨 Creating RAG Assistant...")
    demo = create_gradio_ui()
    print("\n🚀 Launching RAG Assistant...")
    demo.launch(css=custom_css)