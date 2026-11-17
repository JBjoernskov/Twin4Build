# Standard library imports
import cProfile
import io
import os
import pstats

# os.environ['DISABLE_AUTORESET_PRINT'] = '1'
# os.environ['LINE_PROFILE'] = '1'
import sys
from datetime import datetime, timezone
from pstats import SortKey

# Third party imports
# import juliacall
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

# Add Equivalent class between brick:ec2ab3b8_518f_41d7_81c4_3b396f0d9d23 and brick:Weather_Station
from rdflib import URIRef

# Local application imports
import twin4build as tb
import twin4build.core as core
from twin4build.utils.data_loaders.load import load_from_database
from twin4build.utils.logger import LOGGER

if __name__ == "__main__":
    # Load the semantic model
    file_path = r"C:\Users\jabj\Documents\python\Datasets\mortar\mortargraphs\bldg13.ttl"

    # LOGGER.logfile = "mortar_bldg1.log"
    # LOGGER.show_caller("_solve_milp", include_stack=True)
    # LOGGER.show_caller("_connect_components", include_stack=True)

    sm = tb.SemanticModel(rdf_file=file_path, id="bldg13", verbose=1500)
    sm.visualize()
    translator = tb.Translator()
    sim_model = translator.translate(
        sm,
        systems_=[
            tb.BuildingSpaceSystem,
            tb.AirHandlingUnitSystem,
            tb.OutdoorEnvironmentSystem,
            tb.FanCoilUnitSystem
        ],
        verbose=1000,
    )
    sim_model.serialize()
    sim_model.visualize(forward_only=True)
