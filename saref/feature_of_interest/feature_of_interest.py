from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
if TYPE_CHECKING:
    from ..measurement.measurement import Measurement
    from ..property_.property_ import Property
class FeatureOfInterest:
    def __init__(self,
                hasMeasurement: Union[None, Measurement] = None,
                hasProperty: Union[None, Property] = None):
        self.hasMeasurement = hasMeasurement
        self.hasProperty = hasProperty
