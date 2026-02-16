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
    # file_path = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\semantic_model\semantic_model.ttl"
    # file_path = r"C:\Users\jabj\Documents\python\Datasets\mortar\mortargraphs\bldg2.ttl"

    file_path = r"C:\Users\jabj\Documents\python\Datasets\Building Timeseries dataset\Site_A.ttl"
    # file_path = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\site_a\instance_graph.ttl"

    # file_path = r"C:\Users\jabj\Documents\python\Datasets\Building Timeseries dataset\Site_B.ttl"
    # file_path = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\site_b\semantic_model.ttl"

    # file_path = r"C:\Users\jabj\Documents\python\Datasets\HTR\HTR full graph.ttl"

    # sm = tb.SemanticModel(
    #     rdf_file=file_path, id="site_a", verbose=10
    # )
    # sm = tb.SemanticModel(
    #     rdf_file=file_path, id="site_b", verbose=10
    # )

    

    
    # PRINTPROGRESS._use_curses = False
    # PRINTPROGRESS.disable()
    # PRINTPROGRESS.set_caller_filter_mode("blacklist")
    # LOGGER.hide_caller("_match_patterns", include_stack=True)
    # LOGGER.hide_debug()
    LOGGER.set_caller_filter_mode("whitelist")
    LOGGER.show_caller("_solve_milp", include_stack=True)
    LOGGER.show_caller("_connect_components", include_stack=True)
    pr = cProfile.Profile()


    sm = tb.SemanticModel(rdf_file=file_path, id="site_a", verbose=1500)

    # Site A
    weather_station_class = core.namespace.BRICK.Weather_Station
    custom_class = URIRef("https://brickschema.org/schema/Brick#ec2ab3b8_518f_41d7_81c4_3b396f0d9d23")
    # Add the equivalentClass relationship to the ontology graph
    sm.ontology_graph.add((custom_class, core.namespace.OWL.equivalentClass, weather_station_class))
    # sm.reason()

    # HTR
    # Map fso:feedsFluidTo to brick:feeds to enable reasoning across ontologies
    # fso_feeds_fluid_to = core.namespace.FSO.feedsFluidTo
    # brick_feeds = core.namespace.BRICK.feeds
    # # Add the equivalentProperty relationship to the ontology graph
    # sm.ontology_graph.add((fso_feeds_fluid_to, core.namespace.OWL.equivalentProperty, brick_feeds))
    # print(f"Added equivalence: fso:feedsFluidTo ≡ brick:feeds")






    # pr.enable()


    translator = tb.Translator()
    sim_model = translator.translate(sm, systems_=[tb.BuildingSpaceTorchSystem, tb.AirHandlingUnitTorchSystem, tb.OutdoorEnvironmentSystem], verbose=1000)
    # sim_model = translator.translate(sm, systems_=[tb.BuildingSpaceTorchSystem, tb.AirHandlingUnitTorchSystem], verbose=9)

    sim_model.serialize()
    sim_model.visualize(forward_only=True)


    # pr.disable()
    # LOGGER.reset()

    # s = io.StringIO()
    # sortby = SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())







