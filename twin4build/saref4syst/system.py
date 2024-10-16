from __future__ import annotations
from typing import Union
from twin4build.utils.plot.simulation_result import SimulationResult
from prettytable import PrettyTable

class System(SimulationResult):
    """
    A class representing a system.
    """
    def __str__(self):
        t = PrettyTable(field_names=["input", "output"], divider=True)
        title = f"Component overview    id: {self.id}"
        t.title = title
        input_list = list(self.input.keys())
        output_list = list(self.output.keys())
        n = max(len(input_list), len(output_list))
        for j in range(n):
            i = input_list[j] if j<len(input_list) else ""
            o = output_list[j] if j<len(output_list) else ""
            t.add_row([i, o], divider=True if j==len(input_list)-1 else False)
            
        return t.get_string()
    
    def __init__(self,
                connectedTo: Union[list, None]=None,
                feedsFluidTo: Union[list, None]=None, #Assymetric subproperty of connectedTo - from Flow Systems Ontology
                hasFluidFedBy: Union[list, None]=None, #Assymetric subproperty of connectedTo - from Flow Systems Ontology
                suppliesFluidTo: Union[list, None]=None, #Assymetric subproperty of connectedTo - from Flow Systems Ontology
                hasFluidSuppliedBy: Union[list, None]=None, #Assymetric subproperty of connectedTo - from Flow Systems Ontology
                returnsFluidTo: Union[list, None]=None, #Assymetric subproperty of connectedTo - from Flow Systems Ontology
                hasFluidReturnedBy: Union[list, None]=None, #Assymetric subproperty of connectedTo - from Flow Systems Ontology
                hasSubSystem: Union[list, None]=None,
                subSystemOf: Union[list, None]=None,
                connectsAt: Union[list, None]=None,
                connectedThrough: Union[list, None]=None, 
                input: Union[dict, None]=None,
                output: Union[dict, None]=None,
                outputGradient: Union[dict, None]=None,
                parameterGradient: Union[dict, None]=None,
                id: Union[str, None]=None,
                **kwargs):
        """
        Initialize a System object.

        Args:
            connectedTo (list, optional): A list of systems that the system is connected to. Defaults to None.
            feedsFluidTo (list, optional): A list of systems that the system feeds fluid to. Defaults to None.
            hasFluidFedBy (list, optional): A list of systems that the system has fluid fed by. Defaults to None.
            suppliesFluidTo (list, optional): A list of systems that the system supplies fluid to. Defaults to None.
            hasFluidSuppliedBy (list, optional): A list of systems that the system has fluid supplied by. Defaults to None.
            returnsFluidTo (list, optional): A list of systems that the system returns fluid to. Defaults to None.
            hasFluidReturnedBy (list, optional): A list of systems that the system has fluid returned by. Defaults to None.
            hasSubSystem (list, optional): A list of systems that the system has as a sub system. Defaults to None.
            subSystemOf (list, optional): A list of systems that the system is a sub system of. Defaults to None.
            connectsAt (list, optional): A list of connection points that the system connects to. Defaults to None.
            connectedThrough (list, optional): A list of systems that the system connects through. Defaults to None.
            input (dict, optional): A dictionary of inputs to the system. Defaults to None.
            output (dict, optional): A dictionary of outputs from the system. Defaults to None.
            outputGradient (dict, optional): A dictionary of output gradients to the system. Defaults to None.
            parameterGradient (dict, optional): A dictionary of parameter gradients to the system. Defaults to None.
            id (str, optional): The id of the system. Defaults to None.
        """
        super().__init__(**kwargs)
        assert isinstance(connectedTo, list) or connectedTo is None, "Attribute \"connectedTo\" is of type \"" + str(type(connectedTo)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(feedsFluidTo, list) or feedsFluidTo is None, "Attribute \"feedsFluidTo\" is of type \"" + str(type(feedsFluidTo)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasFluidFedBy, list) or hasFluidFedBy is None, "Attribute \"hasFluidFedBy\" is of type \"" + str(type(hasFluidFedBy)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(suppliesFluidTo, list) or suppliesFluidTo is None, "Attribute \"suppliesFluidTo\" is of type \"" + str(type(suppliesFluidTo)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasFluidSuppliedBy, list) or hasFluidSuppliedBy is None, "Attribute \"hasFluidSuppliedBy\" is of type \"" + str(type(hasFluidSuppliedBy)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(returnsFluidTo, list) or returnsFluidTo is None, "Attribute \"returnsFluidTo\" is of type \"" + str(type(returnsFluidTo)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasFluidReturnedBy, list) or hasFluidReturnedBy is None, "Attribute \"hasFluidReturnedBy\" is of type \"" + str(type(hasFluidReturnedBy)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(hasSubSystem, list) or hasSubSystem is None, "Attribute \"hasSubSystem\" is of type \"" + str(type(hasSubSystem)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(subSystemOf, list) or subSystemOf is None, "Attribute \"subSystemOf\" is of type \"" + str(type(subSystemOf)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectsAt, list) or connectsAt is None, "Attribute \"connectsAt\" is of type \"" + str(type(connectsAt)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(connectedThrough, list) or connectedThrough is None, "Attribute \"connectedThrough\" is of type \"" + str(type(connectedThrough)) + "\" but must be of type \"" + str(list) + "\""
        assert isinstance(input, dict) or input is None, "Attribute \"input\" is of type \"" + str(type(input)) + "\" but must be of type \"" + str(dict) + "\""
        assert isinstance(output, dict) or output is None, "Attribute \"output\" is of type \"" + str(type(output)) + "\" but must be of type \"" + str(dict) + "\""
        assert isinstance(id, str), "Attribute \"id\" is of type \"" + str(type(id)) + "\" but must be of type \"" + str(str) + "\""
        if connectedTo is None:
            connectedTo = []
        if feedsFluidTo is None:
            feedsFluidTo = []
        if hasFluidFedBy is None:
            hasFluidFedBy = []
        if suppliesFluidTo is None:
            suppliesFluidTo = []
        if hasFluidSuppliedBy is None:
            hasFluidSuppliedBy = []
        if returnsFluidTo is None:
            returnsFluidTo = []
        if hasFluidReturnedBy is None:
            hasFluidReturnedBy = []
        if hasSubSystem is None:
            hasSubSystem = []
        if subSystemOf is None:
            subSystemOf = []
        if connectsAt is None:
            connectsAt = []
        if connectedThrough is None:
            connectedThrough = []
        if input is None:
            input = {}
        if output is None:
            output = {}
        if outputGradient is None:
            outputGradient = {}
        if parameterGradient is None:
            parameterGradient = {}
        self.connectedTo = connectedTo
        self.feedsFluidTo = feedsFluidTo
        self.hasFluidFedBy = hasFluidFedBy
        self.suppliesFluidTo = suppliesFluidTo
        self.hasFluidSuppliedBy = hasFluidSuppliedBy
        self.returnsFluidTo = returnsFluidTo
        self.hasFluidReturnedBy = hasFluidReturnedBy
        self.hasSubSystem = hasSubSystem
        self.subSystemOf = subSystemOf
        self.connectsAt = connectsAt
        self.connectedThrough = connectedThrough
        self.input = input 
        self.output = output
        self.outputGradient = outputGradient
        self.parameterGradient = parameterGradient
        self.id = id
