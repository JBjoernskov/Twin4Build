import itertools
import Report

class System(Report.Report):
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
        # super().__init__(**kwargs)
        self.connectedTo = connectedTo
        self.hasSubSystem = hasSubSystem
        self.subSystemOf = subSystemOf
        self.connectsAt = connectsAt
        self.connectedThrough = connectedThrough
        self.connectionVisits = connectionVisits ###
        self.input = input ###
        self.output = output ###
        self.systemId = next(self.id_iter) ###
        super().__init__(**kwargs)

class Connection:
    def __init__(self,
                connectsSystem = None,
                connectsSystemAt = None,
                connectionType = None, 
                **kwargs):
        self.connectsSystem = connectsSystem
        self.connectsSystemAt = connectsSystemAt
        self.connectionType = connectionType ###
        super().__init__(**kwargs)


class ConnectionPoint:
    def __init__(self,
                connectionPointOf = None,
                connectsSystemThrough = None, 
                **kwargs):
        self.connectionPointOf = connectionPointOf
        self.connectsSystemThrough = connectsSystemThrough
        super().__init__(**kwargs)

    