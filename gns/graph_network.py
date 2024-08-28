import paddle
import paddle.nn as nn

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    """
    Given the input size, hidden size and output size, build a MLP with ReLU activation and LayerNorm
    """
    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Layer):
    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, node_attr, edge_attr):

        # (x, edge_index, edge_attr, global_attr)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return node_, edge_


class GnBlock(nn.Layer):
    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, x: paddle.Tensor, edge_index: paddle.Tensor, edge_features: paddle.Tensor):
        x_residual = x
        edge_features_residual = edge_features

        x, edge_features = self.eb_module(x, edge_index, edge_features)
        x, edge_features = self.nb_module(x, edge_index, edge_features)
        edge_attr = edge_features_residual + edge_features

        x = x_residual + x
        return x, edge_attr


class Decoder(nn.Layer):
    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(
            hidden_size, hidden_size, output_size, lay_norm=False
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.decode_module(x)


class EncoderProcesserDecoder(nn.Layer):
    def __init__(
        self, message_passing_num, node_input_size, edge_input_size, nparticle_dimensions, hidden_size=128
    ):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
        )

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))

        self.processer_list = nn.LayerList(processer_list)

        self.decoder = Decoder(hidden_size=hidden_size, output_size=nparticle_dimensions)

    def forward(self, x: paddle.Tensor, edge_index: paddle.Tensor, edge_features: paddle.Tensor):
        x, edge_features = self.encoder(x, edge_features)
        for model in self.processer_list:
            x, edge_features = model(x, edge_index, edge_features)
        x = self.decoder(x)

        return x





class EdgeBlock(nn.Layer):
    def __init__(self, custom_func=None):

        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, node_attr, edge_index, edge_attr):

        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = paddle.concat(edges_to_collect, axis=1)

        edge_attr_ = self.net(collected_edges)  # Update

        return node_attr, edge_attr_


class NodeBlock(nn.Layer):
    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, node_attr, edge_index, edge_attr):
        nodes_to_collect = []

        receivers_idx = edge_index[1]
        num_nodes = len(node_attr)
        # OK the scatter add, might need to switch to paddle's scatter_nd_add
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim_size=num_nodes)
        nodes_to_collect.append(node_attr)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = paddle.concat(nodes_to_collect, axis=-1)
        if self.net is not None:
            x = self.net(collected_nodes)
        else:
            x = collected_nodes
        return x, edge_attr

def scatter_add(src, index, dim_size=None):
    if dim_size is None:
        dim_size = paddle.max(index) + 1

    indices = paddle.unsqueeze(index, axis=1)
    x = paddle.zeros_like(src)
    y = paddle.scatter_nd_add(x, indices, src)
    return y[:dim_size]
