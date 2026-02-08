#  Main for Temporal Graph Attention Model Pipeline:
import logging
import networkx
import utils
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# ------- custom modules -------
from preprocessing.pytorch_dataloader import DatasetLoader
from models.pytorch import TemporalGNN
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from scipy.sparse import csr_matrix

# ======================================================================================================================
#   Global Variables
# ======================================================================================================================
torch_extensions: [str] = ['.pth']
np_extensions: [str] = ['.npy']

# ======================================================================================================================
#   Preparation Functions
# ======================================================================================================================

def transform_df_to_npy(timeseries: pd.DataFrame, number_of_nodes: int, save: bool = False) -> np.array:
    """
    This function transform an ORDERED IN ASCENDING ORDER BY TIME timeseries df to .npy 3D.
    Also, it arranges in order the nodes to be sorted in the same way for each snapshot
    Parameters
    ----------
    :param timeseries:
    :param number_of_nodes:
    :param save:
    Returns
    -------
    node_values
    """
    slice_temp = []
    node_values_list = []
    count = 0
    for i, row in timeseries.iterrows():
        if count == number_of_nodes:
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
    node_values_np = np.array(node_values_list)

    if save:
        with open('data/nodes_values.npy', 'wb') as f:
            np.save(f, node_values_np)

    return node_values_np

def get_graph(dependencies: pd.DataFrame, save: bool = False, normalize: bool = True, plot: bool = False) -> csr_matrix:
    """
    This function create the network of the model and extract the adj matrix.
    -------
    :param save:
    :param dependencies: df with node relationship and edges features (e.g., distances -> see drawing method)
    :param normalize:
    :param plot:
    """
    weighted_dependencies = dependencies[["zoneID", "depends_from_ID", "dependence_type"]]
    weighted_dependencies.loc[:,"dependence_type"] = weighted_dependencies['dependence_type'].abs()
    edge_list = weighted_dependencies.values.tolist()
    nodes = weighted_dependencies['zoneID'].unique()
    G = networkx.Graph()
    G.add_nodes_from(nodes)

    for i in range(len(edge_list)):
        G.add_edge(edge_list[i][0], edge_list[i][1], weight=edge_list[i][2])
    A = networkx.adjacency_matrix(G)

    # Weights normalization:
    if normalize:
        A = abs(A)
        row_sums = A.sum(axis=1)
        A = A / row_sums[:, np.newaxis]

    if plot:
        # Shell method: not representative
        plt.figure(figsize=(15, 15))
        networkx.draw_shell(G, with_labels=True)
        plt.draw()
        plt.show()

        # Kamada-Kawai method: representative
        dist = dict(networkx.all_pairs_dijkstra_path_length(G, weight="weight"))
        pos = networkx.kamada_kawai_layout(G, dist=dist)

        plt.figure(figsize=(10, 10))
        deg = dict(G.degree())
        node_sizes = [100 + 80 * deg[n] for n in G.nodes()]

        plt.figure(figsize=(15, 15))
        networkx.draw_networkx_edges(G, pos, alpha=0.25)
        networkx.draw_networkx_nodes(G, pos, node_size=node_sizes)
        networkx.draw_networkx_labels(G, pos, font_size=9)
        plt.axis("off")
        plt.show()

    if save:
        A = A.toarray()
        with open('data/A.npy', 'wb') as f:
            np.save(f, A)
    return A



# ======================================================================================================================
#   Analytical Functions
# ======================================================================================================================

def train_model(model: TemporalGNN, dataset: StaticGraphTemporalSignal,
                train_ratio: float = 0.8, lr: float=0.01, subset: int = 9000,
                epochs: int = 100, save_weights: bool = True) -> TemporalGNN:

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    print("Running training...")
    for epoch in range(epochs):
        loss = 0
        step = 0
        for snapshot in train_dataset:

            # Handle single feature tensors:
            x_input = snapshot.x.unsqueeze(1) if snapshot.x.ndim == 2 else snapshot.x

            # Get model predictions
            y_hat = model(x=x_input, edge_index=snapshot.edge_index,
                          edge_attributes=snapshot.edge_attr)

            # Root Mean squared error
            loss = loss + torch.sqrt(torch.mean((y_hat - snapshot.y) ** 2))
            step += 1
            if step > subset:
                break
        # pred_epoch.append(np.mean(y_list))
        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {} train RMSE: {:.4f}".format(epoch, loss.item()))

    if save_weights:
        torch.save(model.state_dict(), "model_weights.pth")
    return model

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):

    model.eval()
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    #FIXME: complete evaluation step
    with torch.inference_mode():
        pass
    # gather the stats from all processes

# ======================================================================================================================
#  Debug Entrypoint
# ======================================================================================================================
if __name__ == '__main__':
    # local variables
    PROJECT_ROOT: Path = Path(__file__).parents[1]  # root folder of the repository
    DATA_LOCATION: Path = PROJECT_ROOT / 'data'
    RAW_DATA_LOCATION: Path = DATA_LOCATION / 'raw'
    #PREPROC_DATA_LOCATION: Path = DATA_LOCATION / 'preprocessed'  # root folder of the repository #

    # 0 - Graph Info Required:
    number_of_nodes: int = 36
    number_of_features: int = 1 # number of features for each node

    # 1 - Forecasting specifications:
    time_window = 4 # 60 minutes (15 min of sample time)
    forecasting_window  = 1

    # 2 - Transform the timeseries from df to numpy:
    #timeseries_df = pd.read_csv("./data/timeseries_one_year.csv")  # select manually the dataframe for your experiments
    #node_values_ercot = transform_df_to_npy(timeseries=timeseries_df, number_of_nodes=number_of_nodes, save=False)

    # 3 - Get the adj matrix from dependencies' df:
    dependencies_df = pd.read_csv('./data/dependencies.csv')
    adj_matrix = get_graph(dependencies=dependencies_df, save=True, plot=True, normalize=True)

    # 4 - Get the temporal tGNN using Pytorch Class:
    loader = DatasetLoader()
    dataset = loader.get_dataset(num_timesteps_in=time_window, num_timesteps_out=forecasting_window, save=False)

    # 5 - Model Training and Evaluation:
    model = TemporalGNN(node_features=number_of_features, periods=forecasting_window)
    model = train_model(model, dataset, train_ratio=0.8, epochs=200, save_weights = True)

    # 5 - Prediction:
    # TODO: add prediction step


    logging.info('-----> done <-----')
