import signal
import sys


def signal_handler(sig, frame):
    """
    simple signal handler to avoid the KeyboardInterrupt message
    """
    print("Ctrl+C pressed")
    sys.exit(0)


def set_signal_handler():
    signal.signal(signal.SIGINT, signal_handler)
