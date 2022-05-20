from mmu.methods.prpoint import PrecisionRecallEllipticalUncertainty
from mmu.methods.prpoint import PrecisionRecallMultinomialUncertainty
from mmu.methods.prcurve import PrecisionRecallCurveMultinomialUncertainty
from mmu.methods.prcurve import PrecisionRecallCurveEllipticalUncertainty

PREU = PrecisionRecallEllipticalUncertainty
PRMU = PrecisionRecallMultinomialUncertainty
PRCMU = PrecisionRecallCurveMultinomialUncertainty
PRCEU = PrecisionRecallCurveEllipticalUncertainty

__all__ = [
    'PREU',
    'PRMU',
    'PRCMU',
    'PRCEU',
    'PrecisionRecallEllipticalUncertainty',
    'PrecisionRecallMultinomialUncertainty',
    'PrecisionRecallCurveMultinomialUncertainty',
]
