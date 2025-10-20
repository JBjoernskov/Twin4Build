# Standard library imports
import os
import sys
from datetime import datetime, timezone

# Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2

# Local application imports
import twin4build as tb
from twin4build.utils.data_loaders.load import load_from_database


def plot_timeseries_data(data_dict, title="Timeseries Data"):
    """
    General function to plot any timeseries data
    Args:
        data_dict: Dictionary with point names as keys and DataFrames as values
        title: Title for the plot
    """
    if not data_dict:
        print("No data available for plotting")
        return

    points = list(data_dict.keys())
    num_points = len(points)

    # Create subplots for all points
    fig, axes = plt.subplots(num_points, 1, figsize=(12, 3 * num_points), sharex=True)
    if num_points == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16)

    for i, (point_label, df) in enumerate(data_dict.items()):
        if df is not None and not df.empty:
            # Plot the data
            ax = axes[i]
            ax.plot(df.index, df.iloc[:, 0], label=point_label, linewidth=1)
            ax.set_title(f"{point_label}")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Only show x-axis labels on the bottom subplot
            if i == num_points - 1:
                ax.set_xlabel("Time")
                # Rotate x-axis labels for better readability
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.set_xlabel("")

            print(f"  - {point_label}: {len(df)} data points")
            print(
                f"    Value range: {df.iloc[:, 0].min():.2f} to {df.iloc[:, 0].max():.2f}"
            )
        else:
            print(f"  - {point_label}: No data available")
            axes[i].text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].set_title(f"{point_label} (No Data)")
            if i == num_points - 1:
                axes[i].set_xlabel("Time")
            else:
                axes[i].set_xlabel("")

    plt.tight_layout()
    plt.show()


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
    file_path = r"C:\Users\jabj\Documents\python\Datasets\Building Timeseries dataset\Site_B.ttl"

    sm = tb.SemanticModel(
        rdf_file=file_path, id="site_b", parse_namespaces=False, verbose=10
    )
    sm.reason()

    points = sm.get_instances_of_type(tb.ontologies.BRICK.Point)
    print(f"Number of points: {len(points)}")
    
    # sm.reason()

    # sm.parse_namespaces(sm.graph, namespaces={"BRICK": tb.ontology.BRICK})
    # sm.serialize()

    query = """
    CONSTRUCT {
        ?s ?p ?o
    }
    WHERE {
        ?s ?p ?o .
        FILTER ((?p = brick:feeds || ?p = brick:hasPart || ?p = brick:hasPoint || ?p = brick:isLocationOf) &&
                ?p != rdf:type && 
                ?p != rdfs:subClassOf)
    }
    """ #|| ?p = brick:hasPoint || ?p = brick:hasUnit || ?p = brick:adjacentTo  
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "3e61e901_9d3d_44a4_80aa_22259f31bc40" # Weater station
    initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "c10f79c9_744d_40a5_a272_7cde7ca1f5a6" # AHU
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "fd5601fa_5513_4b0a_b935_9817c84319e6.5e4ae5df_1476_48a7_a752_cbd08af4c677" # Air temperature sensor
    # initial_node = sm.namespaces["p33f3e0c2_f2cd_471c_b5a0_4655c2bd4623".upper()] + "13211186_beb4_4227_bd2d_0644e860886e" # Building

    # sm.visualize(query, dpi=30000, include_full_uri=False, generate_subgraphs=False, traversal_mode="bfs", random_seed=None, node_limit=100, initial_node=initial_node)


    model = tb.Model(id="site_b")
    model.load(semantic_model_filename=file_path, verbose=2, draw_semantic_model=False, draw_simulation_model=True)
    # model.semantic_model.visualize()
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

    # Time range for data fetching
    start_time = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    # ============================================================================
    # EXAMPLE 1: Plot a specific sensor by name
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Plot a specific sensor by name")
    print("=" * 60)

    sensor_uuid = "7baae988_b38f_4f0a_a79c_e91d928f4311" #fd5601fa_5513_4b0a_b935_9817c84319e6.5e4ae5df_1476_48a7_a752_cbd08af4c677
    df = load_from_database(
        table_name="bts_site_b",
        sensor_uuid=sensor_uuid,  # Use sensor_name instead of sensor_uuid
        start_time=start_time,
        end_time=end_time,
        step_size=60,
        resample=True,
        resample_method="linear",
        clip=True,
        cache=True,
        # tz="UTC",
        **db_config,
    )

    print(f"df: {df}")

    # Plot the data
    data_dict = {sensor_uuid: df}
    plot_timeseries_data(data_dict, f"Single Sensor: {sensor_uuid}")

    # ============================================================================
    # EXAMPLE 2: Plot multiple specific sensors
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Plot multiple specific sensors")
    print("=" * 60)

    # List of sensors you want to plot
    sensor_names = [
        "bldg1.ZONE.AHU02.RM112.Zone_Air_Temp",
        "bldg1.ZONE.AHU02.RM112.Zone_Air_Temp_Setpoint",
        "bldg1.ZONE.AHU02.RM112.Zone_Supply_Air_Flow",
    ]

    all_data = {}
    for sensor_name in sensor_names:
        print(f"Fetching data for sensor: {sensor_name}")

        try:
            df = load_from_database(
                building_name="bldg1",
                sensor_name=sensor_name,
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

            all_data[sensor_name] = df

        except Exception as e:
            print(f"Error fetching data for {sensor_name}: {e}")
            all_data[sensor_name] = None

    # Plot all the data
    plot_timeseries_data(all_data, "Multiple Sensors")

    # ============================================================================
    # EXAMPLE 3: Use VAV UUIDs from semantic model (for testing)
    # ============================================================================
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Use VAV UUIDs from semantic model (for testing)")
    print("=" * 60)

    # Get VAVs and their points with timeseries IDs
    print("Getting VAVs and their points...")
    vav_data = get_vav_points_with_timeseries(sm)

    # Print available VAVs
    print(f"\nFound {len(vav_data)} VAVs in the semantic model:")
    for vav_name in sorted(vav_data.keys()):
        num_points = len(vav_data[vav_name])
        print(f"  - {vav_name}: {num_points} points")

    # Select a VAV to test with
    sample_vav = "VAVRM103"

    if sample_vav in vav_data:
        points = vav_data[sample_vav]
        print(f"\nTesting with VAV: {sample_vav} ({len(points)} points)")

        # Fetch data for each point using UUIDs
        vav_data_dict = {}
        for point in points:
            point_label = point["label"]
            timeseries_id = point["timeseries_id"]

            print(f"Fetching data for {point_label} (UUID: {timeseries_id})")

            try:
                df = load_from_database(
                    building_name="bldg1",
                    sensor_uuid=timeseries_id,  # Use UUID from semantic model
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

                vav_data_dict[point_label] = df

            except Exception as e:
                print(f"  - Error fetching data: {e}")
                vav_data_dict[point_label] = None

        # Plot the VAV data
        plot_timeseries_data(vav_data_dict, f"VAV Data: {sample_vav}")
    else:
        print(f"VAV '{sample_vav}' not found in the semantic model")
