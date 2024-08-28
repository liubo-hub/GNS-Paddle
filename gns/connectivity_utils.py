import numpy as np
from sklearn import neighbors


def _compute_connectivity(positions, radius):
      tree = neighbors.KDTree(positions)
      receivers_list = tree.query_radius(positions, r=radius)
      num_nodes = len(positions)
      senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
      receivers = np.concatenate(receivers_list, axis=0)
      return senders, receivers


def _compute_connectivity_for_batch(positions, n_node, radius):

      # Separate the positions corresponding to particles in different graphs.
      positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
      receivers_list = []
      senders_list = []
      n_edge_list = []
      num_nodes_in_previous_graphs = 0

      # Compute connectivity for each graph in the batch.
      for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius)

        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        # Because the inputs will be concatenated, we need to add offsets to the
        # sender and receiver indices according to the number of nodes in previous
        # graphs in the same batch.
        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_graph_i = len(positions_graph_i)
        num_nodes_in_previous_graphs += num_nodes_graph_i

      # Concatenate all of the results.
      senders = np.concatenate(senders_list, axis=0).astype(np.int32)
      receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
      n_edge = np.stack(n_edge_list).astype(np.int32)

      return senders, receivers, n_edge


