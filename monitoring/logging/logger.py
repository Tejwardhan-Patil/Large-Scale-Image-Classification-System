import logging
import logging.config
import os
import json
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from contextlib import contextmanager

# Define logger configuration as a dict for structured logging
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s"
        },
        "json": {
            "format": json.dumps({
                "time": "%(asctime)s",
                "logger": "%(name)s",
                "level": "%(levelname)s",
                "thread": "%(threadName)s",
                "message": "%(message)s",
                "file": "%(pathname)s",
                "line": "%(lineno)d"
            })
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed",
            "level": "DEBUG"
        },
        "error_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "logs/error.log",
            "when": "midnight",
            "backupCount": 7,
            "formatter": "json",
            "level": "ERROR"
        }
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"]
        }
    }
}

# Ensure the log directory exists
os.makedirs("logs", exist_ok=True)

# Apply the logger configuration
logging.config.dictConfig(LOG_CONFIG)

logger = logging.getLogger(__name__)

# Custom filter for logging additional context
class RequestIdFilter(logging.Filter):
    """Filter to inject request ID into log records."""
    def __init__(self, request_id):
        self.request_id = request_id

    def filter(self, record):
        record.request_id = self.request_id
        return True

# Thread-safe context manager for logging with context (e.g., request ID)
@contextmanager
def log_context(request_id):
    """Context manager to add request ID into logs."""
    filter = RequestIdFilter(request_id)
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)

# A function to simulate predictions with context-based logging
def log_prediction(image_id, predicted_label, true_label, confidence, request_id=None):
    """Log model prediction details, optionally within a request context."""
    with log_context(request_id) if request_id else contextmanager(lambda: (yield)):
        logger.info(f"Image ID: {image_id}, Predicted: {predicted_label}, True: {true_label}, Confidence: {confidence}")

# Error handling and logging with external services (Sentry, etc.)
def log_error(exception, message, send_to_sentry=False):
    """Log error details and optionally send to external monitoring services."""
    logger.error(f"{message} - Exception: {exception}", exc_info=True)
    if send_to_sentry:
        # Integration with Sentry or another error tracking service
        try:
            import sentry_sdk
            sentry_sdk.capture_exception(exception)
        except ImportError:
            logger.warning("Sentry SDK not installed; error not sent.")

# Function to demonstrate logging multiple threads
def threaded_logging_simulation(thread_name, request_id):
    """Simulate logging in a threaded environment with context."""
    threading.current_thread().name = thread_name
    try:
        log_prediction("img_001.jpg", "cat", "dog", 0.87, request_id=request_id)
        raise ValueError("Threaded model pipeline error.")
    except Exception as e:
        log_error(e, "Threaded logging test failure", send_to_sentry=True)

# Adding extra context dynamically in a log entry
def log_with_custom_context(image_id, additional_context):
    """Log prediction with extra custom context."""
    extra = {"image_id": image_id, **additional_context}
    logger.info("Logging with additional context", extra=extra)

# Function to simulate logging during inference
def inference_logging(image_path, model_version, confidence_threshold=0.9):
    """Log the inference process and potential errors."""
    try:
        logger.info(f"Starting inference on image: {image_path} using model version: {model_version}")
        # Simulate inference logic
        predicted_label = "dog"
        confidence = 0.85
        if confidence < confidence_threshold:
            logger.warning(f"Low confidence ({confidence}) for prediction: {predicted_label}")
        log_prediction(image_path, predicted_label, "dog", confidence)
    except Exception as e:
        log_error(e, "Inference failed", send_to_sentry=True)

# Advanced log filtering with environment-based logic
class EnvironmentFilter(logging.Filter):
    """Custom filter to add environment details to log records."""
    def filter(self, record):
        record.environment = os.getenv("ENVIRONMENT", "development")
        return True

# Adding the environment filter to the logger
env_filter = EnvironmentFilter()
logger.addFilter(env_filter)

# Simulate different logging levels in action
def simulate_logging_levels():
    """Demonstrates different logging levels."""
    logger.debug("Debug message - useful for tracing.")
    logger.info("Info message - general operational messages.")
    logger.warning("Warning message - something unexpected happened.")
    logger.error("Error message - an operation failed.")
    logger.critical("Critical message - the system is in a bad state!")

# Utility function to initialize Sentry for error tracking
def initialize_sentry(dsn):
    """Initialize Sentry for external error tracking."""
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=dsn)
        logger.info("Sentry successfully initialized.")
    except ImportError:
        logger.warning("Sentry SDK is not installed. Unable to initialize.")

# Logging with external services, multi-threading, and different log levels
if __name__ == "__main__":
    # Initialize Sentry
    initialize_sentry(dsn="https://PublicKey@o0.ingest.sentry.io/0")

    # Simulate different logging levels
    simulate_logging_levels()

    # Simulate logging during inference
    inference_logging("img_002.jpg", model_version="v2.1")

    # Simulate multi-threaded logging
    thread1 = threading.Thread(target=threaded_logging_simulation, args=("Thread-1", "req-123"))
    thread2 = threading.Thread(target=threaded_logging_simulation, args=("Thread-2", "req-456"))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # Log with custom context
    log_with_custom_context("img_003.jpg", {"processing_time": "500ms", "gpu_utilization": "85%"})