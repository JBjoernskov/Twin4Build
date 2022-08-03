from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union
if TYPE_CHECKING:
    from ..date_time.date_time.DateTime import DateTime
    from ..unit_of_measure.unit_of_measure import UnitOfMeasure
    from ..feature_of_interest.feature_of_interest import FeatureOfInterest
    from ..property_.property_ import Property

class Measurement:
    def __init__(self,
                hasTimeStamp: Union[None, DateTime] = None,
                hasValue: Union[None, float] = None,
                isMeasuredIn: Union[None, UnitOfMeasure] = None,
                isMeasurementOf: Union[None, FeatureOfInterest] = None,
                relatesToProperty: Union[None, Property] = None):
        # hasTimestamp: Union(None, DateTime)
        # hasValue: Union(None, float)
        # isMeasuredIn: Union(None, UnitOfMeasure)
        # isMeasurementOf: Union(None, FeatureOfInterest)
        # relatesToProperty: Union(None, Property)
        self.hasTimeStamp = hasTimeStamp
        self.hasValue = hasValue
        self.isMeasuredIn = isMeasuredIn
        self.isMeasurementOf = isMeasurementOf
        self.relatesToProperty = relatesToProperty