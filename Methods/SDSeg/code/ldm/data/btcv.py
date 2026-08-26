"""BTCV dataset entry points.

The original SDSeg code called this dataset Synapse.  Keep the implementation
in ``synapse.py`` for compatibility, but expose BTCV-named classes for current
configs and scripts.
"""

from .synapse import (
    ALL_EVAL_VOLUME_IDS,
    COLOR_MAP,
    TEST_VOLUME_IDS,
    TRAIN_VOLUME_IDS,
    VALIDATION_VOLUME_IDS,
    SynapseBase,
    SynapseTrain,
    SynapseValidation,
    SynapseValidationEval,
    SynapseValidationVolume,
    SynapseValidationVolume4test,
    colorize,
)


class BTCVBase(SynapseBase):
    pass


class BTCVTrain(SynapseTrain):
    pass


class BTCVValidation(SynapseValidation):
    pass


class BTCVValidationEval(SynapseValidationEval):
    pass


class BTCVValidationVolume(SynapseValidationVolume):
    pass


class BTCVValidationVolume4test(SynapseValidationVolume4test):
    pass


__all__ = [
    "ALL_EVAL_VOLUME_IDS",
    "COLOR_MAP",
    "TEST_VOLUME_IDS",
    "TRAIN_VOLUME_IDS",
    "VALIDATION_VOLUME_IDS",
    "colorize",
    "BTCVBase",
    "BTCVTrain",
    "BTCVValidation",
    "BTCVValidationEval",
    "BTCVValidationVolume",
    "BTCVValidationVolume4test",
]
