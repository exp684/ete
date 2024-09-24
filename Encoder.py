import copy
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch_geometric.nn.models.basic_gnn
from accelerate import Accelerator
from torch_geometric.nn import conv


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embedding_dim, q_dim, k_dim, v_dim):
        super(MultiHeadAttention, self).__init__()
        #-----------------------------------------------------------------
        self.n_head = n_heads
        self.hidden_dim = embedding_dim // self.n_head

        self.queries_transformation = nn.Linear(q_dim, embedding_dim, bias=False)
        self.keys_transformation = nn.Linear(k_dim, embedding_dim, bias=False)
        self.values_transformation = nn.Linear(v_dim, embedding_dim, bias=False)
        self.out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        #------------------------------------------------------   

        #------------------------------------------------------

    def forward(self, query, key, value, mask=None):
        """
        :param query: [batch_size]
        :param key: [seq_len]
        :param value: [embedding_dim]
        :param mask : [batch_size, seq_len]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        batch_size = query.size(0)

        q1 = self.queries_transformation(query)
        q = q1.reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)
        k = self.keys_transformation(key).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)
        v = self.values_transformation(value).reshape(batch_size, -1, self.n_head, self.hidden_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(3, 2)) / sqrt(self.hidden_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attention = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(attention, v)

        output_contiguous = output.transpose(1, 2).contiguous()
        concat_output = output_contiguous.view(batch_size, -1, self.n_head * self.hidden_dim)

        out = self.out(concat_output)

        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, dim_feedforward):
        super(FeedForwardLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.dim_feedforward = dim_feedforward

        self.lin1 = nn.Linear(self.embedding_dim, self.dim_feedforward)
        self.lin2 = nn.Linear(self.dim_feedforward, self.embedding_dim)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, embedding_dim]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        return self.lin2(functional.relu(self.lin1(inputs)))


class EncoderLayer(nn.Module):
    def __init__(self, n_head, embedding_dim, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(n_head,
                                                       embedding_dim,
                                                       embedding_dim,
                                                       embedding_dim,
                                                       embedding_dim)
        self.feed_forward_layer = FeedForwardLayer(embedding_dim, dim_feedforward)

        self.dropout1 = nn.Dropout(dropout)  # MHA dropout
        self.dropout2 = nn.Dropout(dropout)  # FFL dropout

        self.bn1 = nn.BatchNorm1d(embedding_dim, affine=True)
        self.bn2 = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, embedding_dim]
        :return: out : [batch_size, seq_len, embedding_dim]
        """
        x = x + self.dropout1(self.multi_head_attention(x, x, x))
        x = self.bn1(x.view(-1, x.size(-1))).view(*x.size())

        x = x + self.dropout2(self.feed_forward_layer(x))
        x = self.bn2(x.view(-1, x.size(-1))).view(*x.size())

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_head, embedding_dim, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.scores_embedding = nn.Linear(1, embedding_dim)
        self.city_embedding = nn.Linear(2, embedding_dim)

        self.layers = [EncoderLayer(n_head, embedding_dim, dim_feedforward, dropout) for _ in
                       range(n_layers)]

        self.transformer_encoder = nn.Sequential(*self.layers)

    def forward(self, inputs, scor):
        """
        :param inputs, scor: [batch_size, seq_len, embedding_dim]
        :return: [batch_size, seq_len, embedding_dim]
        """

        embedded_data = self.city_embedding(inputs) + self.scores_embedding(scor)

        return self.transformer_encoder(embedded_data)

    def get_name(self):
        return f"{self.__class__.__name__}"


def get_graph(deltas, embedded_data):
    device = deltas.device

    batch_size, seq_len = embedded_data.shape[:2]
    kappas = [5] * batch_size
    data_list = []
    for b in range(batch_size):
        distances = copy.deepcopy(deltas[b]).to('cpu')
        distances.fill_diagonal_(float('inf'))

        edge_indices_list = []

        # Depot 0 (departure) to all others except itself and depot 1
        depot_0_neighbors = torch.arange(2, seq_len, device='cpu')  # starts from node 2 onwards
        depot_0_src = torch.zeros(depot_0_neighbors.size(0), device='cpu', dtype=torch.long)
        edge_indices_list.append(torch.stack([depot_0_src, depot_0_neighbors], dim=0))

        # Clients and depot 1 to depot 1 (destination)
        for i in range(2, seq_len):  # from each client
            depot_1_dst = torch.ones(1, device='cpu', dtype=torch.long) * 1  # always connect to depot 1
            src_nodes = torch.full((1,), i, device='cpu', dtype=torch.long)
            edge_indices_list.append(torch.stack([src_nodes, depot_1_dst], dim=0))

        # Clients to kappa nearest clients excluding self and both depots
        for i in range(2, seq_len):
            distances_to_others = copy.deepcopy(distances[i]).to('cpu')
            _, indices = torch.topk(distances_to_others, kappas[b] + 2, largest=False)
            indices = indices[(indices != i) & (indices >= 2)]  # Exclude self and depots

            src_nodes = torch.full((indices.size(0),), i, device='cpu', dtype=torch.long)
            edge_indices_list.append(torch.stack([src_nodes, indices], dim=0))

        edge_index = torch.cat(edge_indices_list, dim=1).to(device)

        data = torch_geometric.data.Data(x=embedded_data[b],
                                         edge_index=edge_index,
                                         device=device
                                         )
        data.num_nodes = seq_len
        data_list.append(data)
    batch = torch_geometric.data.Batch.from_data_list(data_list)
    return batch


class GNNEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_head, embedding_dim, dim_feedforward, dropout, gnn_model=None):
        super(GNNEncoder, self).__init__()

        in_channels = embedding_dim
        out_channels = embedding_dim
        self.partial_graph = True
        self.scores_embedding = nn.Linear(1, embedding_dim)

        self.city_embedding = nn.Linear(2, embedding_dim)

        if gnn_model is None:
            gnn_model = torch_geometric.nn.models.basic_gnn.GCN

        if gnn_model in [torch_geometric.nn.models.basic_gnn.GAT]:
            hidden_dim = embedding_dim * n_head
            self.GNNModel = gnn_model(in_channels, hidden_dim, n_layers, out_channels, dropout,
                                      heads=n_head, concat=True)
        else:
            hidden_dim = dim_feedforward
            self.GNNModel = gnn_model(in_channels, hidden_dim, n_layers, out_channels, dropout)

    def forward(self, inputs, scor):
        batch_size, seq_len = inputs.shape[:2]
        deltas = torch.stack([torch.cdist(inputs[i], inputs[i]) for i in range(batch_size)])
        embedded_data = self.city_embedding(inputs) + self.scores_embedding(scor)
        batch = get_graph(deltas, embedded_data)
        node_features = batch.x
        edge_index = batch.edge_index

        return self.GNNModel(node_features, edge_index=edge_index).view(batch_size, seq_len, -1)

    def get_name(self):
        return f"{self.__class__.__name__}-{self.GNNModel.__class__.__name__}"


class GraphEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_head, embedding_dim, dim_feedforward, dropout, gnn_layer=None):
        super(GraphEncoder, self).__init__()
        if gnn_layer is None:
            gnn_layer = conv.TransformerConv

        in_channels = embedding_dim
        out_channels = dim_feedforward
        self.partial_graph = True
        self.scores_embedding = nn.Linear(1, embedding_dim)

        self.city_embedding = nn.Linear(2, embedding_dim)

        layers = []
        for l in range(n_layers):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(dropout)
            ]
            in_channels = dim_feedforward
        layers += [
            gnn_layer(in_channels=in_channels,
                      out_channels=embedding_dim)
        ]
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs, scor):
        import torch_geometric
        batch_size, seq_len = inputs.shape[:2]

        deltas = torch.stack([torch.cdist(inputs[i], inputs[i]) for i in range(batch_size)])
        embedded_data = self.city_embedding(inputs) + self.scores_embedding(scor)
        batch = (
            get_graph(deltas, embedded_data))
        node_features = batch.x
        edge_index = batch.edge_index

        for layer in self.layers:
            if isinstance(layer, torch_geometric.nn.MessagePassing):
                node_features = layer(node_features, edge_index)
            else:
                node_features = layer(node_features)

        return node_features.view(batch_size, seq_len, -1)

    def get_name(self):
        return f"GraphEncoder-{self.layers[0].__class__.__name__}"
