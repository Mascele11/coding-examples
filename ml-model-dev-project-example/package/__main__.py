#  Main for Temporal Data Creation:
import os
import torch
import pickle
import logging
import networkx
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

# ------- custom modules -------
from preprocessing.pytorch_dataloader import DatasetLoader

# ======================================================================================================================
#   Global Variables
# ======================================================================================================================

# ======================================================================================================================
#   Preparation Functions
# ======================================================================================================================

def transform_df_to_npy(timeseries: pd.DataFrame, number_of_features: int, save: bool = False) -> np.array:
    """
    This function transform an ORDERED IN ASCENDING ORDER BY TIME timeseries df to .npy 3D.
    Also, it arranges in order the nodes to be sorted in the same way  for each snapshot

    Parameters
    ----------
    timeseries
    number_of_features
    save

    Returns
    -------
    node_values_ercot
    """
    node_number = number_of_features
    slice_temp = []
    node_values_list = []
    count = 0
    for i, row in timeseries.iterrows():
        if count == node_number:
            slice_temp_df = pd.DataFrame(slice_temp)
            slice_temp_df.sort_values(by=[1], inplace=True)
            slice_temp_lmp = slice_temp_df[0]
            slice_temp_tuple = tuple(slice_temp_lmp.values.tolist())
            node_values_list.append(slice_temp_tuple)
            slice_temp = []
            count = 0
            values = row[["LMP", "nodeID"]]
            edgeList = values.values.tolist()
            slice_temp.append(edgeList)
        else:
            values = row[["LMP", "nodeID"]]
            edgeList = tuple(values.values.tolist())
            slice_temp.append(edgeList)
        count += 1
    node_values_ercot = np.array(node_values_list)

    if save:
        with open('./data/node_values_ercot.npy', 'wb') as f:
            np.save(f, node_values_ercot)

    return node_values_ercot




# ======================================================================================================================
#   Analytical Functions
# ======================================================================================================================

def get_adj_matrix(dependencies: pd.DataFrame, save: bool = False) -> np.array:
    """
    This function create the network of the model and extract the adj matrix.
    Parameters
    ----------
    dependencies
    save

    Returns
    -------
    A
    """
    weighted_dependencies = dependencies[["zoneID", "depends_from_ID", "dependence_type"]]
    edge_list = weighted_dependencies.values.tolist()
    G = networkx.Graph()

    for i in range(len(edge_list)):
        G.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
    A0 = networkx.adjacency_matrix(G).A

    # Weights normalization:
    A0 = abs(A0)
    row_sums = A0.sum(axis=1)
    A = A0 / row_sums[:, np.newaxis]

    if save:
        with open('./data/node_values_ercot.npy', 'wb') as f:
            np.save(f, A)

    return A


# ======================================================================================================================
#  Debug Entrypoint
# ======================================================================================================================
if __name__ == '__main__':
    # local variables
    PROJECT_ROOT: Path = Path(__file__)  # root folder of the repository #
    # 1 - Transform the timeseries from df to numpy:
    timeseries_df = pd.read_csv("./data/timeseries_one_year.csv")  # select manually the dataframe
    #node_values_ercot = transform_df_to_npy(timeseries=timeseries_df, number_of_features=36, save=False)

    # 2 - Get the adj matrix from dependencies' df:
    dependencies_df = pd.read_csv('./data/dependencies.csv')
    adj_matrix = get_adj_matrix(dependencies=dependencies_df, save=False)

    # 3 - Get the temporal tGNN using Pytorch Class:
    loader = DatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=4, num_timesteps_out=1, save=False)

    # 4 - Open Notebook tGNN.ipynb for Model Training and Evaluation:
    # TODO: evaluate coding sample

    print("Dataset type:  ", dataset)
    print("Number of samples / sequences: ", len(set(dataset)))
    print(next(iter(dataset)))

    logging.info('-----> done <-----')
