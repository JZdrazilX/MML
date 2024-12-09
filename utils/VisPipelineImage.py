import os
import pandas as pd
import numpy as np
import random
import geomstats.backend as gs
from typing import Dict, Any
from gtda.mapper import make_mapper_pipeline, CubicalCover
from gtda.mapper.visualization import plot_static_mapper_graph
import hdbscan
from utils.ExtractAssignCol import generate_color_mapping, map_compound_embeddings
from utils.TargetCounts import (
    custom_nutrition,
    most_frequent_value,
    most_frequent_timestamp,
    most_frequent_concentration,
    most_frequent_compound,
)
from utils.EmbeddingPass import EmbeddingPassthrough
from utils.HypDBSCAN import HyperbolicDBSCAN
from utils.HypKmeansCover import GeomstatsKMeansCover

def visualize_mapper_graph(
    config: Dict[str, Any],
    target_column: str,
    random_seed: int,
    space_type: str,  # 'euclidean' or 'poincareball'
):
    """
    Generate a Mapper graph for a specified target variable and save it as an image file.

    Args:
        config (Dict[str, Any]): Configuration parameters.
        target_column (str): Target variable column name.

    Raises:
        ValueError: If the target column does not exist in the data.
    """
    random.seed(random_seed)
    gs.random.seed(random_seed)
    
    df = pd.read_csv(config['mapper_params']['path_to_data'])
    data_columns = df.columns[1:3]
    data_for_mapper = df[data_columns].values

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the data.")
    
    cover_params = config['hyper_params']['Cover']
    cluster_params = config['hyper_params']['Cluster']
    
    if space_type == 'euclidean':
        filter_func = EmbeddingPassthrough()
        cover = CubicalCover(**cover_params)
        clusterer = hdbscan.HDBSCAN(**cluster_params)
        
    elif space_type == 'poincareball':
        initial_centroids= None
        #initial_centroids = np.load(config['centers_path'])
        filter_func = EmbeddingPassthrough()
        cover = GeomstatsKMeansCover(**cover_params, precomputed_centers=initial_centroids)
        clusterer = HyperbolicDBSCAN(**cluster_params)

    pipeline = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=config.get('verbose', True),
        n_jobs=-1,
    )

    if target_column == 'Compound':
        df['Compound_Emb'] = map_compound_embeddings(df, compound_column='Compound')
        color_variable = df['Compound_Emb'].values
        color_map = generate_color_mapping(df, 'Compound_Emb')
        node_statistic = most_frequent_compound
    elif target_column == 'Timestamp':
        color_variable = df[target_column].values
        color_map = generate_color_mapping(df, target_column)
        node_statistic = most_frequent_timestamp
    elif target_column == 'Compound_Concentration':
        color_variable = df[target_column].values
        color_map = generate_color_mapping(df, target_column)
        node_statistic = most_frequent_concentration
    elif target_column == 'Growth_Condition_Binary':
        color_variable = df[target_column].values
        color_map = generate_color_mapping(df, target_column)
        node_statistic = custom_nutrition
    else:
        color_variable = df[target_column].values
        color_map = generate_color_mapping(df, target_column)
        node_statistic = most_frequent_value

    mapper_fig = plot_static_mapper_graph(
        pipeline,
        data_for_mapper,
        color_data=color_variable,
        node_color_statistic=node_statistic,
        layout=config['mapper_params']['layout'],
        layout_dim=config['mapper_params']['layout_dim'],
        node_scale=config['mapper_params']['node_scale'],
        plotly_params={"width": 1200, "height": 700, "showlegend": True},
    )

    for trace in mapper_fig.data:
        if trace.mode == 'markers':
            if trace.marker.color is not None:
                trace.marker.color = [color_map.get(val, '#000000') for val in trace.marker.color]

    for trace in mapper_fig.data:
        if trace.mode == 'markers':
            trace.marker.showscale = False
        trace.showlegend = False

    mapper_fig.update_layout(
        title_text=f"Mapper Graph for {target_column}",
        showlegend=True,
        template="plotly_white"
    )

    output_dir = 'Visualisations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = f"mapper_graph_{target_column}seed{random_seed}.png"
    output_path = os.path.join(output_dir, output_filename)
    mapper_fig.write_image(output_path)
