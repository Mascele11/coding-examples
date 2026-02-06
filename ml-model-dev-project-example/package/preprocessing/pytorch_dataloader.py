import os
import torch
import pickle

import numpy as np
import pandas as pd
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class DatasetLoader(object):
    """A DatasetLoader class to generate a StaticGraphTemporalSignal dataset readable by pytorch_geometric_temporal
    graph models.

    The DatasetLoader is by default handling real-time collected each 15 minutes (sample-time) for all the nodes provided.
    It takes input the adjency matrix of the graph the real-tine features.

    Parameters
    ----------
        adj_mat_file_name.npy = [# nodes x # nodes] {it resumes the static graph architecture}
        nodes_values_file_name.npy  = [# of samples(at 15 minutes) x # nodes x # features] {3-D vector}
   ----------
   """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"), adj_mat_file_name = "A.npy",
                 node_values_file_name = "nodes_values.npy"):
        super(DatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self.A_name = adj_mat_file_name
        self.node_values_file_name = node_values_file_name
        self._read_raw_data()

    def _read_raw_data(self):
        A = np.load(os.path.join(self.raw_data_dir, self.A_name))
        X = np.load(os.path.join(self.raw_data_dir, self.node_values_file_name)).transpose(
            (1, 0))  # transpose into: [#nodes x #features x #sample] - Expected by StatiGraphTemporalSignal
        X = X.astype(np.float32) # conversion for GPU compatibility

        """
        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        """

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """

        # Rolling Time Window Indices Generation: prediction is consecutive
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[1] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations (features, target):
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, i: i + num_timesteps_in]).numpy())
            target.append((self.X[:, i + num_timesteps_in: j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, save: bool = False) -> StaticGraphTemporalSignal:
        """Returns data iterator for specified data as an instance of the
        static graph temporal signal class.

        Return types:
            * **data** *(StaticGraphTemporalSignal)* - The forecasting data (node regression).
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        if save:
            with open('./data_temporal/tGNN_dataset_temporal.pickle', 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return dataset
