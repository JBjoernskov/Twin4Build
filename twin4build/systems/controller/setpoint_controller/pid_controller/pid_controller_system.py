# Standard library imports
import datetime
from typing import Optional

# Third party imports
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares

# Local application imports
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.translator.translator import (
    Exact,
    MultiPath,
    Node,
    Optional_,
    SignaturePattern,
    SinglePath,
)


def get_signature_pattern():
    node0 = Node(cls=core.namespace.S4BLDG.SetpointController)
    node1 = Node(cls=core.namespace.SAREF.Sensor)
    node2 = Node(cls=core.namespace.SAREF.Property)
    node3 = Node(cls=core.namespace.S4BLDG.Schedule)
    node4 = Node(cls=core.namespace.XSD.boolean)
    sp = SignaturePattern(
        semantic_model_=core.ontologies, ownedBy="PIControllerFMUSystem"
    )
    sp.add_triple(
        Exact(subject=node0, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node1, object=node2, predicate=core.namespace.SAREF.observes)
    )
    sp.add_triple(
        Exact(subject=node0, object=node3, predicate=core.namespace.SAREF.hasProfile)
    )
    sp.add_triple(
        Exact(subject=node0, object=node4, predicate=core.namespace.S4BLDG.isReverse)
    )

    sp.add_input("actualValue", node1, "measuredValue")
    sp.add_input("setpointValue", node3, "scheduleValue")
    sp.add_parameter("isReverse", node4)
    sp.add_modeled_node(node0)
    return sp


class PIDControllerSystem(core.System, nn.Module):
    sp = [get_signature_pattern()]

    def __init__(
        self,
        # isTemperatureController=None,
        # isCo2Controller=None,
        kp=0.001,
        Ti=10,
        Td=0.0,
        isReverse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nn.Module.__init__(self)
        self.isReverse = isReverse

        kp = abs(kp)

        if isReverse == False:
            kp = -kp
        Ti = abs(Ti)
        Td = abs(Td)

        self.kp = tps.Parameter(
            torch.tensor(kp, dtype=torch.float64), requires_grad=False
        )
        self.Ti = tps.Parameter(
            torch.tensor(Ti, dtype=torch.float64), requires_grad=False
        )
        self.Td = tps.Parameter(
            torch.tensor(Td, dtype=torch.float64), requires_grad=False
        )

        self.input = {"actualValue": tps.Scalar(), "setpointValue": tps.Scalar()}
        self.output = {"inputSignal": tps.Scalar(0)}
        self._config = {"parameters": ["kp", "Ti", "Td", "isReverse"]}

    @property
    def config(self):
        return self._config

    def initialize(
        self,
        startTime: datetime.datetime,
        endTime: datetime.datetime,
        stepSize: int,
        simulator: core.Simulator,
    ) -> None:
        self.input["actualValue"].initialize(
            startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator
        )
        self.input["setpointValue"].initialize(
            startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator
        )
        self.output["inputSignal"].initialize(
            startTime=startTime, endTime=endTime, stepSize=stepSize, simulator=simulator
        )
        # self.acc_err = torch.tensor([0], dtype=torch.float64, requires_grad=False)
        self.err_prev = torch.tensor([0], dtype=torch.float64, requires_grad=False)
        self.err_prev_m1 = torch.tensor([0], dtype=torch.float64, requires_grad=False)
        self.u_prev = torch.tensor([0], dtype=torch.float64, requires_grad=False)

    def asymptotic_smooth_saturation(
        self, u, lower=0.0, upper=1.0, eps=0, curve_start=0.01, steepness=1
    ):
        effective_min = lower + eps
        effective_max = upper - eps

        lower_curve_point = effective_min + curve_start
        upper_curve_point = effective_max - curve_start

        # Three explicit regions
        result = torch.where(
            u < lower_curve_point,
            # Lower region: curve toward effective_min
            effective_min
            + (lower_curve_point - effective_min)
            * torch.exp(-steepness * (lower_curve_point - u) / curve_start),
            torch.where(
                u > upper_curve_point,
                # Upper region: curve toward effective_max
                effective_max
                - (effective_max - upper_curve_point)
                * torch.exp(-steepness * (u - upper_curve_point) / curve_start),
                # Linear region: perfect passthrough
                u,
            ),
        )

        return result

    def do_step(
        self,
        secondTime: float,
        dateTime: datetime.datetime,
        stepSize: int,
        stepIndex: int,
    ) -> None:
        err = self.input["setpointValue"].get() - self.input["actualValue"].get()
        du = self.kp.get() * (
            (1 + stepSize / self.Ti.get() + self.Td.get() / stepSize) * err
            + (-1 - 2 * self.Td.get() / stepSize) * self.err_prev
            + self.Td.get() / stepSize * self.err_prev_m1
        )

        u = self.u_prev + du
        u = self.asymptotic_smooth_saturation(
            u, lower=0.0, upper=1.0, curve_start=0.05, steepness=1
        )
        self.u_prev = u
        self.err_prev_m1 = self.err_prev
        self.err_prev = err

        self.output["inputSignal"].set(u, stepIndex)
