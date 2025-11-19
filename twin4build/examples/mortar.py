# Standard library imports
import os
import sys
from datetime import datetime, timezone

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import juliacall

# Local application imports
import twin4build.core as core
import twin4build as tb
from twin4build.utils.data_loaders.load import load_from_database

# Add Equivalent class between brick:ec2ab3b8_518f_41d7_81c4_3b396f0d9d23 and brick:Weather_Station
from rdflib import URIRef


def plot_timeseries_data(data, title="Timeseries Data", max_plots_per_figure=5):
    """
    General function to plot any timeseries data
    Args:
        data: List of items, where each item is either:
              - A tuple (DataFrame, point_label) for a single plot
              - A list of tuples [(DataFrame, point_label), ...] for multiple series on the same subplot
        title: Title for the plot(s)
        max_plots_per_figure: Maximum number of subplots per figure (default: 5)
    """
    if not data:
        print("No data available for plotting")
        return

    num_points = len(data)

    # Split data into chunks if there are too many points
    chunks = []
    for i in range(0, num_points, max_plots_per_figure):
        chunks.append(data[i:i + max_plots_per_figure])
    
    num_figures = len(chunks)
    
    # Create a figure for each chunk
    for fig_idx, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        
        # Create subplots for this chunk
        fig, axes = plt.subplots(chunk_size, 1, figsize=(12, 3 * chunk_size), sharex=True)
        if chunk_size == 1:
            axes = [axes]
        
        # Add figure number to title if multiple figures
        if num_figures > 1:
            fig.suptitle(f"{title} (Part {fig_idx + 1}/{num_figures})", fontsize=16)
        else:
            fig.suptitle(title, fontsize=16)

        for i, item in enumerate(chunk):
            ax = axes[i]
            
            # Check if item is a list (multiple series on same subplot) or a single tuple
            if isinstance(item, list):
                # Multiple series on the same subplot
                has_data = False
                subplot_title_parts = []
                
                for df, point_label in item:
                    if df is not None and not df.empty:
                        ax.plot(df.index, df.iloc[:, 0], label=point_label, linewidth=1, alpha=0.8)
                        has_data = True
                        subplot_title_parts.append(point_label.split(" | ")[0])  # Use short name for title
                        
                        print(f"  - {point_label}: {len(df)} data points")
                        print(f"    Value range: {df.iloc[:, 0].min():.2f} to {df.iloc[:, 0].max():.2f}")
                    else:
                        print(f"  - {point_label}: No data available")
                
                if has_data:
                    ax.set_title(" vs ".join(subplot_title_parts))
                    ax.set_ylabel("Value")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')
                else:
                    ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title("Multiple Series (No Data)")
            else:
                # Single series
                df, point_label = item
                if df is not None and not df.empty:
                    ax.plot(df.index, df.iloc[:, 0], label=point_label, linewidth=1)
                    ax.set_title(f"{point_label}")
                    ax.set_ylabel("Value")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    print(f"  - {point_label}: {len(df)} data points")
                    print(f"    Value range: {df.iloc[:, 0].min():.2f} to {df.iloc[:, 0].max():.2f}")
                else:
                    print(f"  - {point_label}: No data available")
                    ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{point_label} (No Data)")

            # Only show x-axis labels on the bottom subplot
            if i == chunk_size - 1:
                ax.set_xlabel("Time")
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.set_xlabel("")

        plt.tight_layout()
    

def plot_scatter(data_x, data_y, label_x, label_y, title="Scatter Plot", max_points=10000):
    """
    Create a scatter plot comparing two sensors
    Args:
        data_x: DataFrame for x-axis
        data_y: DataFrame for y-axis
        label_x: Label for x-axis sensor
        label_y: Label for y-axis sensor
        title: Title for the plot
        max_points: Maximum number of points to display (for performance)
    """
    if data_x is None or data_x.empty or data_y is None or data_y.empty:
        print("Cannot create scatter plot: one or both datasets are empty")
        return
    
    # Align the data by timestamp (inner join)
    combined = pd.merge(data_x, data_y, left_index=True, right_index=True, how='inner', suffixes=('_x', '_y'))
    
    if combined.empty:
        print("Cannot create scatter plot: no overlapping timestamps")
        return
    
    total_points = len(combined)
    print(f"\nScatter plot data:")
    print(f"  Total number of points: {total_points}")
    print(f"  X-axis ({label_x}): {combined.iloc[:, 0].min():.2f} to {combined.iloc[:, 0].max():.2f}")
    print(f"  Y-axis ({label_y}): {combined.iloc[:, 1].min():.2f} to {combined.iloc[:, 1].max():.2f}")
    
    # Calculate correlation on full dataset
    correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])
    print(f"  Correlation coefficient: {correlation:.3f}")
    
    # Subsample if there are too many points (for display and trend line calculation)
    if total_points > max_points:
        print(f"  Subsampling to {max_points} points for display")
        # Use uniform sampling to preserve distribution
        combined_display = combined.sample(n=max_points, random_state=42)
    else:
        combined_display = combined
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(combined_display.iloc[:, 0], combined_display.iloc[:, 1], alpha=0.5, s=10)
    ax.set_xlabel(label_x, fontsize=12)
    ax.set_ylabel(label_y, fontsize=12)
    ax.set_title(f"{title}\nCorrelation: {correlation:.3f} (n={total_points:,})", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add a trend line - use subsampled data and add error handling
    try:
        # Use even smaller sample for trend line if needed
        sample_size = min(5000, len(combined_display))
        if len(combined_display) > sample_size:
            trend_data = combined_display.sample(n=sample_size, random_state=42)
        else:
            trend_data = combined_display
            
        z = np.polyfit(trend_data.iloc[:, 0], trend_data.iloc[:, 1], 1)
        p = np.poly1d(z)
        
        # Sort x values for smooth line
        x_sorted = np.sort(combined_display.iloc[:, 0].values)
        ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2, 
                label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        ax.legend()
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"  Warning: Could not compute trend line ({e})")
        # Continue without trend line
    
    plt.tight_layout()

def get_sensor_id(sm, uuid):
    obj = sm.get_instance(uuid)
    sensor_id = obj.get_predicate_object_pairs()[sm.SENAPS.stream_id][0].uri
    sensor_type = obj.type
    z_ = {e for e in sensor_type if e.has_subclasses() == False}
    z = {e.uri.n3(sm.graph.namespace_manager) for e in z_}
    if len(z) == 0:
        z = {"Unknown class"}
    sensor_type = " | ".join(z)  # data

    return sensor_id, sensor_type

def get_vav_points_with_timeseries(semantic_model):
    """
    Get VAVs and their points with timeseries IDs from the semantic model
    Returns a dictionary with VAV names as keys and lists of points as values
    """
    # SPARQL query to find VAVs and their points with timeseries IDs
    query = """
    PREFIX brick: <https://brickschema.org/schema/Brick#>
    PREFIX ref: <https://brickschema.org/schema/Brick/ref#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ns2: <http://buildsys.org/ontologies/bldg1#>
    
    SELECT ?vav ?point ?point_label ?timeseries_id
    WHERE {
        ?vav a brick:VAV .
        ?vav brick:hasPoint ?point .
        ?point rdfs:label ?point_label .
        ?point ref:hasExternalReference ?ref .
        ?ref ref:hasTimeseriesId ?timeseries_id .
    }
    ORDER BY ?vav ?point_label
    """

    results = semantic_model.graph.query(query)

    # Organize results by VAV (using local part of URI as name)
    vav_data = {}

    def get_local_name(uri):
        return str(uri).split("#")[-1]

    for row in results:
        vav_uri = str(row[0])
        vav_name = get_local_name(vav_uri)
        point_uri = str(row[1])
        point_label = str(row[2])
        timeseries_id = str(row[3])

        if vav_name not in vav_data:
            vav_data[vav_name] = []

        vav_data[vav_name].append(
            {"uri": point_uri, "label": point_label, "timeseries_id": timeseries_id}
        )

    return vav_data


if __name__ == "__main__":
    # Load the semantic model
    # file_path = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\semantic_model\semantic_model.ttl"
    # file_path = r"C:\Users\jabj\Documents\python\Datasets\mortar\mortargraphs\bldg2.ttl"

    # file_path = r"C:\Users\jabj\Documents\python\Datasets\Building Timeseries dataset\Site_A.ttl"
    # file_path = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\site_a\instance_graph.ttl"    

    file_path = r"C:\Users\jabj\Documents\python\Datasets\Building Timeseries dataset\Site_B.ttl"
    # file_path = r"C:\Users\jabj\Documents\python\Twin4Build\twin4build\examples\generated_files\models\site_b\semantic_model.ttl"

    # file_path = r"C:\Users\jabj\Documents\python\Datasets\HTR\HTR full graph.ttl"


    # sm = tb.SemanticModel(
    #     rdf_file=file_path, id="site_a", verbose=10
    # )
    # sm = tb.SemanticModel(
    #     rdf_file=file_path, id="site_b", verbose=10
    # )
    sm = tb.SemanticModel(
        rdf_file=file_path, id="site_b", verbose=10
    )


    # Site A
    # weather_station_class = core.namespace.BRICK.Weather_Station
    # custom_class = URIRef("https://brickschema.org/schema/Brick#ec2ab3b8_518f_41d7_81c4_3b396f0d9d23")
    # # Add the equivalentClass relationship to the ontology graph
    # sm.ontology_graph.add((custom_class, core.namespace.OWL.equivalentClass, weather_station_class))
    

    # HTR
    # Map fso:feedsFluidTo to brick:feeds to enable reasoning across ontologies
    # fso_feeds_fluid_to = core.namespace.FSO.feedsFluidTo
    # brick_feeds = core.namespace.BRICK.feeds
    # # Add the equivalentProperty relationship to the ontology graph
    # sm.ontology_graph.add((fso_feeds_fluid_to, core.namespace.OWL.equivalentProperty, brick_feeds))
    # print(f"Added equivalence: fso:feedsFluidTo ≡ brick:feeds")
    


    analyze_types = True
    if analyze_types:

        points = sm.get_instances_of_type(tb.ontologies.BRICK.Point)
        # sm.parse_namespaces(sm.graph, namespaces={"BRICK": core.namespace.BRICK})
        types = [t for p_ in points for t in p_.type]
        specific_types = list()
        for p in points:
            specific_types.append(p.get_most_specific_type())

        assert len(specific_types) == len(points)
        for p, t in zip(points, specific_types):
            print(f"Point: {p.get_short_name()}", f"      Type: {t}")

        # Get the most specific class for each point
        unique_types = set(specific_types)
        
        # Print unique types in different formats for easy copy-paste
        print("\n" + "="*80)
        print("UNIQUE TYPES - DIFFERENT FORMATS")
        print("="*80)
        
        # Python list format - one per line (JSON compatible with double quotes)
        print("\nPython/JSON list format (one per line):")
        print("[")
        sorted_types = sorted(unique_types)
        for i, t in enumerate(sorted_types):
            comma = "," if i < len(sorted_types) - 1 else ""
            print(f'    "{t}"{comma}')
        print("]")
        
        # Compact Python list
        print("\nPython list format (compact):")
        print(sorted(unique_types))
        
        # Comma-separated
        print("\nComma-separated:")
        print(", ".join(sorted(unique_types)))
        
        # With counts
        print("\nWith counts:")
        for t in sorted(unique_types):
            count = len(sm.get_instances_of_type(core.namespace.BRICK.__getitem__(t)))
            print(f"  {t}: {count}")
        
        print(f"\nTotal unique types: {len(unique_types)}")
        print(f"Total points: {len(points)}")
        print("="*80 + "\n")
        

    # sm.reason()

    # sm.parse_namespaces(sm.graph, namespaces={"BRICK": tb.ontology.BRICK})
    # sm.serialize()

    query = """
    CONSTRUCT {
    ?s ?p ?o
    }
    WHERE {
        ?s ?p ?o .
        FILTER (?p = brick:feeds
                )
    }
    """ 
    # Debug: First check if there are any brick:feeds relationships at all
    # query = """
    # CONSTRUCT { ?s brick:feeds ?o }
    # WHERE { ?s brick:feeds ?o . }
    # """
    
    # Debug: Check if subjects have types
    # query = """
    # CONSTRUCT { ?s ?p ?o }
    # WHERE { 
    #     ?s brick:feeds ?o .
    #     ?s rdf:type ?sType .
    # }
    # """
    


    # Site A
    # initial_node = sm.namespaces["p2b104292_1925_4929_9986_bae1c0029526".upper()] + "1022f474_6f92_4f4c_a2fb_dd557af40739" # Weather station
    # initial_node = sm.namespaces["p2b104292_1925_4929_9986_bae1c0029526"] + "7b077b33_ad6d_422b_bb77_240fe37a82da" #Room
    # initial_node = sm.namespaces["p2b104292_1925_4929_9986_bae1c0029526".upper()] + "a33eec24_6ed3_4c48_a426_1a6bb4deac93" # Exhaust fan
    # initial_node = sm.namespaces["p2b104292_1925_4929_9986_bae1c0029526"] + "1022f474_6f92_4f4c_a2fb_dd557af40739" # Overall AHU

    
    # Site B
    #|| ?p = brick:hasPoint || ?p = brick:hasUnit || ?p = brick:adjacentTo  ?p = brick:hasPart || ?p = brick:feeds
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "3e61e901_9d3d_44a4_80aa_22259f31bc40" # Weater station
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623"] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6" # AHU
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "fd5601fa_5513_4b0a_b935_9817c84319e6.5e4ae5df_1476_48a7_a752_cbd08af4c677" # Air temperature sensor
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "13211186_beb4_4227_bd2d_0644e860886e" # Buildings
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "d02c6fdd_ce6b_44a1_b193_1ede85e7d4c8.d0a77ae4_734b_4a4c_bff7_66ecdfd8e3af" # Room
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "5c2c591b_b04c_47d6_a586_7b323df99c9a" # HVAC zone

    # HTR
    initial_node = sm.namespaces["inst"] + "HTR9_VEN02"


    sm.reason()
    sm.serialize()
    sm.visualize(query, dpi=30000, include_full_uri=False, slice_uri=(-8, None), generate_subgraphs=False, traversal_mode="bfs", random_seed=None, node_limit=300, initial_node=initial_node)
    # sm.visualize(query, dpi=30000, include_full_uri=False, generate_subgraphs=False, traversal_mode=None, random_seed=None, node_limit=200, initial_node=None)
    a
    
    # https://example.com/inst#de757268-ae7f-4271-9846-adbe8ec919b3-004acc59-2
    # sm.get_instance(initial_node).get_most_specific_type()
    # sm.serialize()
    # aa
    
    translator = tb.Translator()
    sim_model = translator.translate(sm, systems_=[tb.BuildingSpaceTorchSystem])
    
    sim_model.visualize()

    # model = tb.Model(id="site_b")
    # model.load(semantic_model_filename=file_path, verbose=10, draw_semantic_model=False, draw_simulation_model=True)
    # # model.semantic_model.visualize()
    # model.simulation_model.visualize()



    # query = """
    # CONSTRUCT {
    #     ?s ?p ?o
    # }
    # WHERE {
    #     ?s ?p ?o .
    #     FILTER (?p = brick:hasPoint && ?s = brick:Weather_Station)
    # }
    # """
    # sm.visualize(query, dpi=30000, include_full_uri=False)

    # p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623:c10f79c9_744d_40a5_a272_7cde7ca1f5a6

    
    
    # Database configuration
    db_config = {
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "postgres",
        "db_user": "postgres",
        "db_password": "password",
    }

    # Time range for data fetching - Single day analysis: August 10, 2021
    start_time = datetime(2021, 6, 10, 18, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2021, 8, 11, 18, 0, 0, tzinfo=timezone.utc)

    # # ============================================================================
    # # EXAMPLE 1: Plot a specific sensor by name
    # # ============================================================================
    # print("\n" + "=" * 60)
    # print("EXAMPLE 1: Plot a specific sensor by name")
    # print("=" * 60)

    
    # df = load_from_database(
    #     table_name="bts_site_b",
    #     sensor_uuid=sensor_uuid,  # Use sensor_name instead of sensor_uuid
    #     start_time=start_time,
    #     end_time=end_time,
    #     step_size=60,
    #     resample=True,
    #     resample_method="linear",
    #     clip=True,
    #     cache=True,
    #     # tz="UTC",
    #     **db_config,
    # )

    # print(f"df: {df}")

    # # Plot the data
    # data_dict = {sensor_uuid: df}
    # plot_timeseries_data(data_dict, f"Single Sensor: {sensor_uuid}")

    # ============================================================================
    # EXAMPLE 2: Plot multiple specific sensors
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Plot multiple specific sensors")
    print("=" * 60)



    """
    brick:hasPoint p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623:c10f79c9_744d_40a5_a272_7cde7ca1f5a6.4b524f12_0c6f_4abd_a677_51db619f635f, #Average Room Air Temperature 
        p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623:c10f79c9_744d_40a5_a272_7cde7ca1f5a6.bb042843_86c6_4d69_9e92_ef556849911b, #Warmest Room Air Temperature
        p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623:c10f79c9_744d_40a5_a272_7cde7ca1f5a6.eb67d269_f4a3_40c0_851e_66bd89efb0da ;
    """



    # List of sensors you want to plot
    sensor_ids = [
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "08ff7ae0_acef_4f4d_8328_2de484a6d5e1.18999260_6d8c_4cd3_91fd_c52f89e25b4b", #heating command
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "47f9140d_38d8_4391_9aa1_08973d4b6370.b71b369a_bc34_4ff1_9a24_95de01fca685",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.012b93b4_cb2c_4f56_86fa_8bc6b4d39cdd",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.0654bfbd_6249_4bb5_a8fd_4a19da72ad73",
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.0d760e87_4d3f_4b5a_8580_8a5fc28d0113", # Enable status
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.15a55f68_5a9d_4cf6_a73a_67b57b63b704",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.1b47bdc9_74ed_405c_8653_aecb4fe76c9b",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.91e817fb_0048_4353_88f9_dc551aaaa440",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.d2de1941_7ba3_40ef_88a7_5e71b79bf1fa",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.04fa8942_55b0_41fb_9cfc_66d087d70245",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.292353e0_4db5_4ee1_a0b6_99a2446e5a68",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.345dacf8_07e0_45c0_b151_c22f407fcc13",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.75d51cf9_fc4c_4f0b_9987_0d2ceb58e294",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.9661bdcc_e598_421d_82f4_97a0dc5863de",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.c1ddc422_0faa_4dfb_9eb9_074f8b9281e6", #brick:point
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.c534d2ad_f93c_4741_8503_e4fa0c79ccae",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.ca485062_e6af_41f3_8e24_eced3bef0c87", # point
        [sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.cc4b2230_9874_4b09_92d4_3b0fd4ff9318", #min temp setpoint
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.7204a748_bc51_4070_bb1e_4a50b901df00", # Discharge
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.f043bd1e_905e_45bb_b70b_33cc422e7a27"], #max temp setpoint
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.fbf312d9_84e6_4d92_90e4_66457501404d", #point
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e48f2638_d690_4f23_907e_f19a534c7f34",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.eb67d269_f4a3_40c0_851e_66bd89efb0da",
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.f59da42f_3979_47fa_9a62_7d073323a985", # system status
        # 
                # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.d4cb6566_30c1_470d_9777_8c2d71c6bd8d",
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.d851e5bd_1766_4aae_a571_d5070bcc3493", #temperature setpoint
        # sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.e3794860_91e1_4d33_b59e_6d32b3608901.df34251f_78ab_4f7b_8e8b_75ed453cbe89", 
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.ac15587f_8b12_478f_bba8_652bfd9b9fd2", #return damper
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.36ab6c42_36b6_45af_a65e_fbd6d119b6f4", #outside damper
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.0feaf9d9_924e_42dc_9bfc_aad29a54d56f", # heationg valve
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.f81f9843_b9c5_45ba_b1fc_00ecb34928ff", # coolling valve
        [sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.eb67d269_f4a3_40c0_851e_66bd89efb0da", # room air temperature setpoint
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.4b524f12_0c6f_4abd_a677_51db619f635f",# average room air temperature
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.fb20d0cb_185c_4d94_b1a1_37df9bc44074", #heating temperature setpoiint] 
        sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6.bb042843_86c6_4d69_9e92_ef556849911b"],  # warmest zone
    ]


    # Outdoor temperature UUID provided by user
    outdoor_temp_uuid = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "bc3a295e_af26_46d7_9445_63325ab035b8.d33b6eb3_5b84_4e98_a1cd_546105794982"
    
    # sensor_ids = [sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "bc3a295e_af26_46d7_9445_63325ab035b8.6fe38251_759a_43cf_94d2_d33296c8b4c8",
    #     sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "bc3a295e_af26_46d7_9445_63325ab035b8.a803cc3d_3abe_4025_b543_59663bfdee18",
    #     sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "bc3a295e_af26_46d7_9445_63325ab035b8.d17aa466_7355_4d3e_9087_ccdc4c6777ad",
    #     outdoor_temp_uuid ] 


    # sensor_ids = ["714c7aa4_2e2e_45e3_966d_919072ce254a",
    # "b34b56ec_8b0e_44d7_9c8c_163046963890",
    # "54ba0cea_8d18_4610_be10_532f8c8387bd",
    # "2fd1c1ab_7bfe_44d1_b1b9_01488a83f05f", ##Average Zone Air Temperature
    # "73700c2d_cc14_4743_a2a3_b1cfe2eb032b", #Warmest zone air temperature
    # "d26e2136_8c4d_4f0b_9b38_5681724bb725"]  #Supply air setpoint

    plot_data = False
    if plot_data:
        all_data = []
        for sensor_item in sensor_ids:
            # Check if this is a nested list (group of sensors to plot together)
            if isinstance(sensor_item, list):
                print(f"\nFetching grouped sensors ({len(sensor_item)} sensors):")
                group_data = []
                for sensor_uuid in sensor_item:
                    print(f"  Fetching data for sensor: {sensor_uuid}")
                    sensor_id, sensor_type = get_sensor_id(sm, sensor_uuid)
                    try:
                        df = load_from_database(
                            table_name="bts_site_b",
                            sensor_id=sensor_id,
                            start_time=start_time,
                            end_time=end_time,
                            step_size=60,
                            resample=True,
                            resample_method="linear",
                            clip=True,
                            cache=True,
                            tz="UTC",
                            **db_config,
                        )
                        
                        sensor_type = sensor_type + " | " + str(sensor_uuid).split(".")[-1][:8]
                        group_data.append((df, sensor_type))
                        
                    except Exception as e:
                        print(f"  Error fetching data for {sensor_uuid}: {e}")
                        group_data.append((None, sensor_type))
                
                # Add the entire group as a single item
                all_data.append(group_data)
            else:
                # Single sensor
                sensor_uuid = sensor_item
                print(f"\nFetching data for sensor: {sensor_uuid}")
                sensor_id, sensor_type = get_sensor_id(sm, sensor_uuid)
                try:
                    df = load_from_database(
                        table_name="bts_site_b",
                        sensor_id=sensor_id,
                        start_time=start_time,
                        end_time=end_time,
                        step_size=60,
                        resample=True,
                        resample_method="linear",
                        clip=True,
                        cache=True, 
                        tz="UTC",
                        **db_config,
                    )

                    sensor_type = sensor_type + " | " + str(sensor_uuid).split(".")[-1][:8]

                    all_data.append((df, sensor_type))

                except Exception as e:
                    print(f"Error fetching data for {sensor_uuid}: {e}")
                    all_data.append((None, sensor_type))

        # Plot all the data
        plot_timeseries_data(all_data, "Multiple Sensors", max_plots_per_figure=8)
        # plt.show()





    do_symbolic_regression = False
    if do_symbolic_regression:


        # ============================================================================
        # SYMBOLIC REGRESSION: Discover heating valve control logic
        # ============================================================================
        print("\n" + "=" * 80)
        print("SYMBOLIC REGRESSION: Heating Valve Control Discovery")
        print("=" * 80)

        # Define namespace prefix for convenience
        NS = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()]
        AHU_PREFIX = "c10f79c9_744d_40a5_a272_7cde7ca1f5a6"
        
        # TARGET: Heating valve position (what we want to predict)
        heating_valve_uuid = NS + f"{AHU_PREFIX}.0feaf9d9_924e_42dc_9bfc_aad29a54d56f"
        
        # INPUT FEATURES: Potential control inputs
        feature_sensors = {
            # "discharge_temp": NS + f"{AHU_PREFIX}.7204a748_bc51_4070_bb1e_4a50b901df00",
            # "discharge_temp_setpoint": NS + f"{AHU_PREFIX}.e3794860_91e1_4d33_b59e_6d32b3608901.d851e5bd_1766_4aae_a571_d5070bcc3493",
            # "min_temp_setpoint": NS + f"{AHU_PREFIX}.e3794860_91e1_4d33_b59e_6d32b3608901.cc4b2230_9874_4b09_92d4_3b0fd4ff9318",
            # "max_temp_setpoint": NS + f"{AHU_PREFIX}.e3794860_91e1_4d33_b59e_6d32b3608901.f043bd1e_905e_45bb_b70b_33cc422e7a27",
            "avg_room_temp": NS + f"{AHU_PREFIX}.4b524f12_0c6f_4abd_a677_51db619f635f",
            # "warmest_zone_temp": NS + f"{AHU_PREFIX}.bb042843_86c6_4d69_9e92_ef556849911b",
            "heating_temp_setpoint": NS + f"{AHU_PREFIX}.fb20d0cb_185c_4d94_b1a1_37df9bc44074",
            # "return_damper": NS + f"{AHU_PREFIX}.ac15587f_8b12_478f_bba8_652bfd9b9fd2",
            # "outside_damper": NS + f"{AHU_PREFIX}.36ab6c42_36b6_45af_a65e_fbd6d119b6f4",
            # "cooling_valve": NS + f"{AHU_PREFIX}.f81f9843_b9c5_45ba_b1fc_00ecb34928ff",
            # "heating_valve": NS + f"{AHU_PREFIX}.0feaf9d9_924e_42dc_9bfc_aad29a54d56f",
        }
        
        # Outdoor temperature
        # feature_sensors["outdoor_temp"] = outdoor_temp_uuid
        
        # Enable status - used for filtering, not as a feature
        enable_status_uuid = NS + f"{AHU_PREFIX}.0d760e87_4d3f_4b5a_8580_8a5fc28d0113"
        
        # Step 1: Load target data (heating valve position)
        print("\n1. Loading TARGET: Heating Valve Position")
        print("-" * 80)
        target_id, target_type = get_sensor_id(sm, heating_valve_uuid)
        # target_id, target_type = get_sensor_id(sm, NS + f"{AHU_PREFIX}.7204a748_bc51_4070_bb1e_4a50b901df00")

        print(f"   Sensor ID: {target_id}")
        print(f"   Type: {target_type}")
        
        y_df = load_from_database(
            table_name="bts_site_b",
            sensor_id=target_id,
            start_time=start_time,
            end_time=end_time,
            step_size=60,
            resample=True,
            resample_method="linear",
            clip=True,
            cache=True,
            tz="UTC",
            **db_config,
        )
        
        print(f"   Loaded {len(y_df)} data points")
        print(f"   Value range: {y_df.iloc[:, 0].min():.2f} to {y_df.iloc[:, 0].max():.2f}")
        
        # Step 2: Load feature data
        print("\n2. Loading INPUT FEATURES:")
        print("-" * 80)
        feature_data = {}
        
        for feature_name, feature_uuid in feature_sensors.items():
            print(f"\n   Loading: {feature_name}")
            try:
                sensor_id, sensor_type = get_sensor_id(sm, feature_uuid)
                print(f"     Sensor ID: {sensor_id}")
                
                df = load_from_database(
                    table_name="bts_site_b",
                    sensor_id=sensor_id,
                    start_time=start_time,
                    end_time=end_time,
                    step_size=60,
                    resample=True,
                    resample_method="linear",
                    clip=True,
                    cache=True,
                    tz="UTC",
                    **db_config,
                )
                
                feature_data[feature_name] = df
                print(f"     Loaded {len(df)} points, range: {df.iloc[:, 0].min():.2f} to {df.iloc[:, 0].max():.2f}")
                
            except Exception as e:
                print(f"     ERROR: {e}")
                feature_data[feature_name] = None
        
        # Step 2b: Load enable_status for filtering
        print("\n2b. Loading FILTER: Enable Status")
        print("-" * 80)
        try:
            enable_status_sensor_id, enable_status_type = get_sensor_id(sm, enable_status_uuid)
            print(f"   Sensor ID: {enable_status_sensor_id}")
            
            enable_status_df = load_from_database(
                table_name="bts_site_b",
                sensor_id=enable_status_sensor_id,
                start_time=start_time,
                end_time=end_time,
                step_size=60,
                resample=True,
                resample_method="linear",
                clip=True,
                cache=True,
                tz="UTC",
                **db_config,
            )
            
            print(f"   Loaded {len(enable_status_df)} points")
            print(f"   Value range: {enable_status_df.iloc[:, 0].min():.2f} to {enable_status_df.iloc[:, 0].max():.2f}")
            print(f"   Enabled timesteps: {(enable_status_df.iloc[:, 0] > 0).sum()} / {len(enable_status_df)}")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            print("   WARNING: Proceeding without filtering by enable_status")
            enable_status_df = None
        
        # Step 3: Prepare data for symbolic regression
        print("\n3. Preparing Data for Symbolic Regression")
        print("-" * 80)
        
        # Combine all dataframes with common timestamps
        all_dfs = {"heating_valve": y_df}
        all_dfs.update({k: v for k, v in feature_data.items() if v is not None})
        
        # Add enable_status for filtering (will be removed later)
        if enable_status_df is not None:
            all_dfs["_enable_status_filter"] = enable_status_df
        
        # Merge all dataframes on timestamp (inner join to get common timestamps)
        combined_df = None
        for name, df in all_dfs.items():
            if df is not None and not df.empty:
                df_renamed = df.copy()
                df_renamed.columns = [name]
                if combined_df is None:
                    combined_df = df_renamed
                else:
                    combined_df = combined_df.join(df_renamed, how='inner')
        
        print(f"   Combined dataset shape: {combined_df.shape}")
        print(f"   Features available: {list(combined_df.columns)}")
        print(f"   Total data points: {len(combined_df)}")
        
        # Remove any rows with NaN values
        combined_df = combined_df.dropna()
        print(f"   After removing NaN: {len(combined_df)} data points")
        
        # Filter by enable_status (keep only timesteps where system is enabled)
        if "_enable_status_filter" in combined_df.columns:
            print(f"\n   Filtering by enable_status (keeping only enabled timesteps)...")
            before_filter = len(combined_df)
            combined_df = combined_df[combined_df["_enable_status_filter"] > 0]
            after_filter = len(combined_df)
            print(f"   Data points after filtering: {after_filter} (removed {before_filter - after_filter} disabled timesteps)")
            
            # Remove the enable_status column (no longer needed)
            combined_df = combined_df.drop(columns=["_enable_status_filter"])
            print(f"   Final features: {list(combined_df.columns)}")
        
        # Lag features by 1 timestep (predict valve position at time t using features at time t-1)
        print(f"\n   Applying 1-timestep lag to features...")
        print(f"   This means: predict valve_position[t] using features[t-1]")
        
        # Get the target column (heating_valve) - this stays at time t
        target_col = combined_df["heating_valve"].copy()
        
        # Get feature columns - these will be lagged to time t-1
        feature_columns = [col for col in combined_df.columns if col != "heating_valve"]
        
        # Shift features by 1 timestep (lag them)
        # shift(1) moves data down by 1 row, so row i gets the value from row i-1
        combined_df_lagged = combined_df[feature_columns].shift(1)
        
        # Combine lagged features with the target (at time t)
        combined_df = pd.concat([target_col, combined_df_lagged], axis=1)
        
        # Remove the first row (which has NaN after shifting)
        before_lag_drop = len(combined_df)
        combined_df = combined_df.dropna()
        after_lag_drop = len(combined_df)
        print(f"   Data points after removing first row (NaN due to lag): {after_lag_drop} (removed {before_lag_drop - after_lag_drop})")
        
        # Split into X (features) and y (target)
        feature_cols = [col for col in combined_df.columns if col != "heating_valve"]
        X = combined_df[feature_cols].values
        y = combined_df["heating_valve"].values

        # X = X[3:,:]
        # y = y[3:]
        
        print(f"\n   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Feature names: {feature_cols}")
        
        # Step 4: Apply Symbolic Regression
        print("\n4. Running Symbolic Regression with PySR")
        print("=" * 80)
        
        try:
            from pysr import PySRRegressor
            
            # Configure the symbolic regression model
            model = PySRRegressor(
                niterations=50,
                binary_operators=["+", "-", "*", "max", "min"],
                unary_operators=[],
                maxsize=20,
                populations=30,
                population_size=50,
                ncyclesperiteration=550,
                constraints={
                    # "-": (1, 1),
                    # "*": (1, 1),
                    # "max": (1, 1),
                },
                verbosity=1,
                progress=True,
                temp_equation_file=True,
                complexity_of_constants=1.01,
            )
            
            print("\nFitting model (this may take a few minutes)...")
            model.fit(X, y, variable_names=feature_cols)

            pd.set_option('display.max_colwidth', None)
            
            # Display results
            print("\n" + "=" * 80)
            print("RESULTS: Best Equations Found")
            print("=" * 80)
            print(model)
            
            # Save the best equation
            best_equation = model.get_best()
            print("\n" + "=" * 80)
            print("BEST EQUATION:")
            print("=" * 80)
            print(best_equation)

            
            # Calculate predictions and metrics
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            print("\n" + "=" * 80)
            print("MODEL PERFORMANCE:")
            print("=" * 80)
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Plot results
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Actual vs Predicted
            axes[0].scatter(y, y_pred, alpha=0.5, s=10)
            axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
            axes[0].set_xlabel('Actual Heating Valve Position')
            axes[0].set_ylabel('Predicted Heating Valve Position')
            axes[0].set_title(f'Actual vs Predicted (R² = {r2:.4f})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Time series comparison
            time_index = combined_df.index[:len(y)]
            axes[1].plot(time_index, y, label='Actual', linewidth=1, alpha=0.7)
            axes[1].plot(time_index, y_pred, label='Predicted', linewidth=1, alpha=0.7)
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Heating Valve Position')
            axes[1].set_title('Actual vs Predicted Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("\nERROR: PySR is not installed!")
            print("Please install it with: pip install pysr")
            print("You may also need to install Julia first: https://julialang.org/downloads/")
            print("\nAfter installing Julia, run:")
            print("  python -c 'import pysr; pysr.install()'")
            
        except Exception as e:
            print(f"\nERROR during symbolic regression: {e}")
            import traceback
            traceback.print_exc()

    