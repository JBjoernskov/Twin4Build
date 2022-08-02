import itertools
from report import Report

class System(Report):
    id_iter = itertools.count()
    def __init__(self,
                connectedTo = None,
                hasSubSystem = None,
                subSystemOf = None,
                connectsAt = None,
                connectedThrough = None, 
                connectionVisits = None,
                input = None,
                output = None,
                **kwargs):
        super().__init__(**kwargs)
        self.connectedTo = connectedTo
        self.hasSubSystem = hasSubSystem
        self.subSystemOf = subSystemOf
        self.connectsAt = connectsAt
        self.connectedThrough = connectedThrough
        self.connectionVisits = connectionVisits ###
        self.input = input ###
        self.output = output ###
        self.systemId = next(self.id_iter) ###