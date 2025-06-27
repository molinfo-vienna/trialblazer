"""Documentation about trialblazer."""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Huanni Zhang"
__email__ = "huanni.zhang@univie.ac.at"
__version__ = "0.1.0"

from .trialblazer import Trialblazer
from .trialtrainer import TrialTrainer
