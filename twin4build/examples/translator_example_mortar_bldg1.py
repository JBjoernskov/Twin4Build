# Standard library imports
import os
# os.environ['DISABLE_AUTORESET_PRINT'] = '1'
# os.environ['LINE_PROFILE'] = '1'
import sys
from datetime import datetime, timezone

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
from twin4build.utils.print_progress import LOGGER
import cProfile, pstats, io
from pstats import SortKey

if __name__ == "__main__":
    # Load the semantic model
    file_path = r"C:\Users\jabj\Documents\python\Datasets\mortar\mortargraphs\bldg1.ttl"

    # LOGGER.set_caller_filter_mode("whitelist")
    LOGGER.hide_debug()
    # LOGGER.show_caller("_solve_milp", include_stack=True)
    # LOGGER.show_caller("_connect_components", include_stack=True)


    sm = tb.SemanticModel(rdf_file=file_path, id="bldg1", verbose=1500)
    sm.visualize()
    translator = tb.Translator()
    sim_model = translator.translate(sm, systems_=[tb.BuildingSpaceTorchSystem, tb.AirHandlingUnitTorchSystem, tb.OutdoorEnvironmentSystem], verbose=1000)
    sim_model.serialize()
    sim_model.visualize(forward_only=True)





