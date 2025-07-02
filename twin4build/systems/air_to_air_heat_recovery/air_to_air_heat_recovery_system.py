from twin4build.utils.constants import Constants
import twin4build.core as core
from twin4build.translator.translator import SignaturePattern, Node, Exact, MultiPath, Optional_, SinglePath
import twin4build.utils.types as tps
import datetime
from typing import Optional

def get_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery)
    node1 = Node(cls=core.namespace.S4BLDG.OutdoorEnvironment)
    node2 = Node(cls=core.namespace.S4BLDG.FlowJunction)
    node3 = Node(cls=core.namespace.S4BLDG.FlowJunction)
    node4 = Node(cls=core.namespace.S4BLDG.PrimaryAirFlowRateMax)
    node5 = Node(cls=core.namespace.SAREF.PropertyValue)
    node6 = Node(cls=core.namespace.XSD.float)
    node7 = Node(cls=core.namespace.S4BLDG.SecondaryAirFlowRateMax)
    node8 = Node(cls=core.namespace.SAREF.PropertyValue)
    node9 = Node(cls=core.namespace.XSD.float)
    node10 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery) #primary
    node11 = Node(cls=core.namespace.S4BLDG.AirToAirHeatRecovery) #secondary
    node12 = Node(cls=core.namespace.S4BLDG.Controller)
    node13 = Node(cls=core.namespace.SAREF.Motion)
    node14 = Node(cls=core.namespace.S4BLDG.Schedule)
    sp = SignaturePattern(semantic_model_=core.ontologies, ownedBy="AirToAirHeatRecoverySystem")

    # buildingTemperature (SecondaryTemperatureIn)
    sp.add_triple(SinglePath(subject=node10, object=node1, predicate=core.namespace.FSO.hasFluidSuppliedBy))
    sp.add_triple(SinglePath(subject=node10, object=node2, predicate=core.namespace.FSO.suppliesFluidTo))
    sp.add_triple(SinglePath(subject=node11, object=node3, predicate=core.namespace.FSO.hasFluidReturnedBy))

    sp.add_triple(Optional_(subject=node5, object=node6, predicate=core.namespace.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node5, object=node4, predicate=core.namespace.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node5, predicate=core.namespace.SAREF.hasPropertyValue))

    # airFlowRateMax
    sp.add_triple(Optional_(subject=node8, object=node9, predicate=core.namespace.SAREF.hasValue))
    sp.add_triple(Optional_(subject=node8, object=node7, predicate=core.namespace.SAREF.isValueOfProperty))
    sp.add_triple(Optional_(subject=node0, object=node8, predicate=core.namespace.SAREF.hasPropertyValue))

    sp.add_triple(Exact(subject=node10, object=node0, predicate=core.namespace.S4SYST.subSystemOf))
    sp.add_triple(Exact(subject=node11, object=node0, predicate=core.namespace.S4SYST.subSystemOf))

    sp.add_triple(Exact(subject=node12, object=node13, predicate=core.namespace.SAREF.controls))
    sp.add_triple(Exact(subject=node13, object=node0, predicate=core.namespace.SAREF.isPropertyOf))
    sp.add_triple(Exact(subject=node12, object=node14, predicate=core.namespace.SAREF.hasProfile))

    sp.add_parameter("primaryAirFlowRateMax", node6)
    sp.add_parameter("secondaryAirFlowRateMax", node9)

    sp.add_input("primaryTemperatureIn", node1, "outdoorTemperature")
    sp.add_input("secondaryTemperatureIn", node3, "airTemperatureOut")
    sp.add_input("primaryAirFlowRate", node2, "airFlowRateIn")
    sp.add_input("secondaryAirFlowRate", node3, "airFlowRateOut")
    sp.add_input("primaryTemperatureOutSetpoint", node14, "scheduleValue")

    sp.add_modeled_node(node0)
    sp.add_modeled_node(node10)
    sp.add_modeled_node(node11)

    return sp

class AirToAirHeatRecoverySystem(core.System):
    r"""
    An air-to-air heat recovery system model.
    
    This model represents a heat exchanger that recovers heat between supply and
    exhaust air streams. The effectiveness varies based on flow rates and operation
    mode (heating or cooling). The model includes temperature setpoint control and
    handles cases where heat recovery is not feasible.
    
    The model is implemented with the following features:
       - Flow-dependent effectiveness interpolation
       - Separate effectiveness values for heating and cooling modes
       - Temperature setpoint control
       - Energy conservation between air streams
    
    Mathematical Formulation:

       The effectiveness :math:`\varepsilon` is interpolated based on flow rate :math:`f`:

       .. math::

          \varepsilon(f) = \varepsilon_{75} + (\varepsilon_{100} - \varepsilon_{75}) \cdot \frac{f - 0.75}{1 - 0.75}

       where:
          :math:`\varepsilon_{75}` is the effectiveness at 75% flow
          :math:`\varepsilon_{100}` is the effectiveness at 100% flow
          :math:`f` is the normalized flow rate (between 0.75 and 1.0)

       The outlet temperature of the primary air stream is:

       .. math::

          T_\text{out,primary} = T_\text{in,primary} + \varepsilon(f) \cdot (T_\text{in,secondary} - T_\text{in,primary}) \cdot \frac{C_\min}{C_\text{sup}}

       where:
          :math:`T_\text{in,primary}` is the inlet temperature of the primary air
          :math:`T_\text{in,secondary}` is the inlet temperature of the secondary air
          :math:`C_\min = \min(C_\text{sup}, C_\text{exh})` is the minimum heat capacity rate
          :math:`C_\text{sup}` is the heat capacity rate of the supply (primary) air
          :math:`C_\text{exh}` is the heat capacity rate of the exhaust (secondary) air

       The outlet temperature of the secondary air stream is:

       .. math::

          T_\text{out,secondary} = T_\text{in,secondary} - \Delta T \cdot \frac{C_\text{sup}}{C_\text{exh}}

       where:
          :math:`\Delta T = T_\text{out,primary} - T_\text{in,primary}`

    Args:
       eps_75_h (float): Effectiveness at 75% flow in heating mode
       eps_100_h (float): Effectiveness at 100% flow in heating mode
       eps_75_c (float): Effectiveness at 75% flow in cooling mode
       eps_100_c (float): Effectiveness at 100% flow in cooling mode
       primaryAirFlowRateMax (float): Maximum primary (supply) air flow rate [kg/s]
       secondaryAirFlowRateMax (float): Maximum secondary (exhaust) air flow rate [kg/s]
    """
    sp = [get_signature_pattern()]
    def __init__(self,
                eps_75_h=None,
                eps_100_h=None,
                eps_75_c=None,
                eps_100_c=None,
                primaryAirFlowRateMax=None,
                secondaryAirFlowRateMax=None,
                **kwargs):
        super().__init__(**kwargs)
        self.eps_75_h = eps_75_h
        self.eps_100_h = eps_100_h
        self.eps_75_c = eps_75_c
        self.eps_100_c = eps_100_c
        self.primaryAirFlowRateMax = primaryAirFlowRateMax
        self.secondaryAirFlowRateMax = secondaryAirFlowRateMax

        self.input = {"primaryAirFlowRate": tps.Scalar(),
                      "secondaryAirFlowRate": tps.Scalar(),
                      "primaryTemperatureIn": tps.Scalar(),
                      "secondaryTemperatureIn": tps.Scalar(),
                      "primaryTemperatureOutSetpoint": tps.Scalar()}
        self.output = {"primaryTemperatureOut": tps.Scalar(),
                       "secondaryTemperatureOut": tps.Scalar()}
        self._config = {"parameters": ["eps_75_h",
                                       "eps_100_h",
                                       "eps_75_c",
                                       "eps_100_c",
                                       "primaryAirFlowRateMax",
                                       "secondaryAirFlowRateMax"]}

    @property
    def config(self):
        """Get the configuration parameters.

        Returns:
            dict: Dictionary containing configuration parameters.
        """
        return self._config

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None,
                    model=None):
        """Initialize the system for simulation.
        
        This method is currently not implemented as the system does not require initialization.
        
        Args:
            startTime (datetime, optional): Start time of the simulation period.
            endTime (datetime, optional): End time of the simulation period.
            stepSize (float, optional): Time step size in seconds.
            model (object, optional): Simulation model object.
        """
        pass

    def do_step(self, 
                secondTime: Optional[float] = None, 
                dateTime: Optional[datetime.datetime] = None, 
                stepSize: Optional[float] = None, 
                stepIndex: Optional[int] = None) -> None:
        """Perform one simulation step.
        
        This method calculates the heat recovery between supply and exhaust air streams
        based on the current flow rates and temperatures. The effectiveness is interpolated
        based on the flow rates, and the operation mode (heating/cooling) is determined
        by comparing inlet temperatures and setpoints.
        
        The method handles the following cases:
        1. No flow in either stream: Pass-through temperatures
        2. Heat recovery feasible: Calculate effectiveness and heat transfer
        3. Heat recovery not feasible: Pass-through temperatures
        
        Args:
            secondTime (float, optional): Current simulation time in seconds.
            dateTime (datetime, optional): Current simulation date and time.
            stepSize (float, optional): Time step size in seconds.
            stepIndex (int, optional): Current simulation step index.
        """
        self.output.update(self.input)
        tol = 1e-5
        if self.input["primaryAirFlowRate"]>tol and self.input["secondaryAirFlowRate"]>tol:
            m_a_max = max(self.primaryAirFlowRateMax, self.secondaryAirFlowRateMax)
            if self.input["primaryTemperatureIn"] < self.input["secondaryTemperatureIn"]:
                eps_75 = self.eps_75_h
                eps_100 = self.eps_100_h
                feasibleMode = "Heating"
            else:
                eps_75 = self.eps_75_c
                eps_100 = self.eps_100_c
                feasibleMode = "Cooling"

            operationMode = "Heating" if self.input["primaryTemperatureIn"]<self.input["primaryTemperatureOutSetpoint"] else "Cooling"

            if feasibleMode==operationMode:
                f_flow = 0.5*(self.input["primaryAirFlowRate"] + self.input["secondaryAirFlowRate"])/m_a_max
                eps_op = eps_75 + (eps_100-eps_75)*(f_flow-0.75)/(1-0.75)
                C_sup = self.input["primaryAirFlowRate"]*Constants.specificHeatCapacity["air"]
                C_exh = self.input["secondaryAirFlowRate"]*Constants.specificHeatCapacity["air"]
                C_min = min(C_sup, C_exh)
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"] + eps_op*(self.input["secondaryTemperatureIn"] - self.input["primaryTemperatureIn"])*(C_min/C_sup), stepIndex)

                if operationMode=="Heating" and self.output["primaryTemperatureOut"]>self.input["primaryTemperatureOutSetpoint"]:
                    self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureOutSetpoint"], stepIndex)
                elif operationMode=="Cooling" and self.output["primaryTemperatureOut"]<self.input["primaryTemperatureOutSetpoint"]:
                    self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureOutSetpoint"], stepIndex)
                
                 # Calculate secondaryTemperatureOut using energy conservation
                primary_delta_T = self.output["primaryTemperatureOut"].get() - self.input["primaryTemperatureIn"].get()
                secondary_delta_T = primary_delta_T * (C_sup/C_exh)
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"].get() - secondary_delta_T, stepIndex)
                    
            else:
                self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"], stepIndex)
                self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"], stepIndex)
        else:
            self.output["primaryTemperatureOut"].set(self.input["primaryTemperatureIn"], stepIndex)
            self.output["secondaryTemperatureOut"].set(self.input["secondaryTemperatureIn"], stepIndex)

