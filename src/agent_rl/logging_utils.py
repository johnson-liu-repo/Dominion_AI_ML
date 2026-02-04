import logging

_TRAIN_LOGGER_NAME = "train"


def configure_training_logging():
    """
    Configure logging so only training-relevant messages are emitted.
    Suppresses noisy pyminion INFO logs while enabling a dedicated
    training logger with a simple message-only formatter.
    """
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    train_logger = logging.getLogger(_TRAIN_LOGGER_NAME)
    train_logger.setLevel(logging.INFO)
    train_logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in train_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        train_logger.addHandler(handler)

    return train_logger


def get_train_logger():
    return logging.getLogger(_TRAIN_LOGGER_NAME)
