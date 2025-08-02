import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import TransformerConv, GATConv
from torch_geometric.utils import dense_to_sparse

#@title Batch normalization layers
class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)

        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
        return x_bn


class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_trans = e.transpose(1, 3).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()  # Reshape to original
        return e_bn

class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        """
        Args:
            x: Input features (batch_size, hidden_dim)

        Returns:
            y: Output predictions (batch_size, output_dim)
        """
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y

class NodeFeatures(nn.Module):
    """Convnet features for nodes.

    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]

    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # Extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new

class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new

class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Define net parameters
        self.num_nodes = config['num_nodes']
        self.node_dim = config['node_dim']
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['gcn_hidden_dim']
        self.num_layers = config['gcn_num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']
        # Node and edge embedding layers/lookups

        # We are using TransformerConv layer from torch geometric library!
        self.nodes_coord_embedding = TransformerConv(self.node_dim, self.hidden_dim)

        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def loss_edges(self, y_pred_edges, y_edges, edge_cw):
        """
        Loss function for edge predictions.

        Args:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            loss_edges: Value of loss function

        """
        # Edge loss
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = nn.NLLLoss(edge_cw)
        loss = loss_edges(y.contiguous(), y_edges)
        return loss

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            loss: Value of loss function
        """
        batch_size = x_edges.size(0)
        actual_num_nodes = x_edges.size(1)  # Gunakan ukuran aktual dari input
        device = x_edges.device

        # Create batched graph for TransformerConv - more GPU efficient
        # Flatten batch and create offset edge indices
        x_nodes_coord_flat = x_nodes_coord.view(-1, x_nodes_coord.size(-1))  # (B*V, node_dim)

        # Create batched edge indices with proper offsets
        edge_indices = []
        for i in range(batch_size):
            edge_index = x_edges[i].nonzero().t().contiguous()
            edge_index = edge_index + i * actual_num_nodes  # offset by graph position
            edge_indices.append(edge_index)

        if edge_indices:
            edge_index_batched = torch.cat(edge_indices, dim=1)  # (2, total_edges)
        else:
            edge_index_batched = torch.empty((2, 0), dtype=torch.long, device=device)

        # Apply TransformerConv to batched graph
        x_flat = self.nodes_coord_embedding(x_nodes_coord_flat, edge_index_batched)

        # Reshape back to batch format
        x = x_flat.view(batch_size, actual_num_nodes, -1)  # B x V x H

        # Edge embedding
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)

        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out

        loss = np.inf
        if y_edges is not None and edge_cw is not None:
            loss = self.loss_edges(y_pred_edges.to(device), y_edges.to(device), edge_cw.to(device))

        return y_pred_edges, loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

class GATNodeFeatures(nn.Module):
    """GAT-based node feature update with attention mechanism.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(GATNodeFeatures, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # Multi-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Updated node features (batch_size, num_nodes, hidden_dim)
        """
        B, V, H = x.shape

        # Multi-head attention
        Q = self.query(x).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)  # B x num_heads x V x head_dim
        K = self.key(x).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)    # B x num_heads x V x head_dim
        V_att = self.value(x).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)  # B x num_heads x V x head_dim

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B x num_heads x V x V

        # Apply edge gates as attention mask (average across hidden dim)
        edge_mask = torch.mean(edge_gate, dim=-1).unsqueeze(1)  # B x 1 x V x V
        attention_scores = attention_scores * edge_mask

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V_att)  # B x num_heads x V x head_dim

        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(B, V, H)

        # Final projection
        x_new = self.output_proj(attended_values)

        return x_new

class ResidualGATLayer(nn.Module):
    """GAT layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(ResidualGATLayer, self).__init__()
        self.node_feat = GATNodeFeatures(hidden_dim, num_heads, dropout)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x

        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H

        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)

        # Node convolution with GAT
        x_tmp = self.node_feat(x_in, edge_gate)

        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)

        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)

        # Residual connection
        x_new = x_in + x
        e_new = e_in + e

        return x_new, e_new

class ResidualGATModel(nn.Module):
    """Residual GAT Model for outputting predictions as edge adjacency matrices.
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGATModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        # Define net parameters
        self.num_nodes = config['num_nodes']
        self.node_dim = config['node_dim']
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['gat_hidden_dim']
        self.num_layers = config['gat_num_layers']
        self.mlp_layers = config['mlp_layers']

        # GAT specific parameters
        self.num_heads = 8
        self.dropout = 0.1

        # Node and edge embedding layers
        # Using GATConv for initial node coordinate embedding
        self.nodes_coord_embedding = GATConv(
            self.node_dim,
            self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout
        )

        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)

        # Define GAT Layers
        gat_layers = []
        for layer in range(self.num_layers):
            gat_layers.append(ResidualGATLayer(self.hidden_dim, self.num_heads, self.dropout))
        self.gat_layers = nn.ModuleList(gat_layers)

        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def loss_edges(self, y_pred_edges, y_edges, edge_cw):
        """
        Loss function for edge predictions.

        Args:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            loss_edges: Value of loss function
        """
        # Edge loss
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = nn.NLLLoss(edge_cw)
        loss = loss_edges(y.contiguous(), y_edges)
        return loss

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            loss: Value of loss function
        """
        batch_size = x_edges.size(0)
        actual_num_nodes = x_edges.size(1)  # Gunakan ukuran aktual dari input
        device = x_edges.device

        # Create batched graph for GATConv - GPU efficient
        # Flatten batch and create offset edge indices
        x_nodes_coord_flat = x_nodes_coord.view(-1, x_nodes_coord.size(-1))  # (B*V, node_dim)

        # Create batched edge indices with proper offsets
        edge_indices = []
        for i in range(batch_size):
            edge_index = x_edges[i].nonzero().t().contiguous()
            edge_index = edge_index + i * actual_num_nodes  # offset by graph position
            edge_indices.append(edge_index)

        if edge_indices:
            edge_index_batched = torch.cat(edge_indices, dim=1)  # (2, total_edges)
        else:
            edge_index_batched = torch.empty((2, 0), dtype=torch.long, device=device)

        # Apply GATConv to batched graph
        x_flat = self.nodes_coord_embedding(x_nodes_coord_flat, edge_index_batched)

        # Reshape back to batch format
        x = x_flat.view(batch_size, actual_num_nodes, -1)  # B x V x H

        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)

        # GAT layers
        for layer in range(self.num_layers):
            x, e = self.gat_layers[layer](x, e)  # B x V x H, B x V x V x H

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out

        loss = np.inf
        if y_edges is not None and edge_cw is not None:
            loss = self.loss_edges(y_pred_edges.to(device), y_edges.to(device), edge_cw.to(device))

        return y_pred_edges, loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

class GATv2NodeFeatures(nn.Module):
    """GATv2-based node feature update with improved attention mechanism.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(GATv2NodeFeatures, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        # GATv2 style attention - using linear transformation after concatenation
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.Parameter(torch.empty(size=(1, num_heads, 2 * self.head_dim)))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Initialize parameters
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_gate):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Updated node features (batch_size, num_nodes, hidden_dim)
        """
        B, V, H = x.shape

        # Linear transformation
        h = self.W(x)  # B x V x H
        h = h.view(B, V, self.num_heads, self.head_dim)  # B x V x num_heads x head_dim

        # GATv2 attention mechanism
        # Create all pairs for attention computation
        h_i = h.unsqueeze(2).expand(-1, -1, V, -1, -1)  # B x V x V x num_heads x head_dim
        h_j = h.unsqueeze(1).expand(-1, V, -1, -1, -1)  # B x V x V x num_heads x head_dim

        # Concatenate for GATv2 style attention
        h_concat = torch.cat([h_i, h_j], dim=-1)  # B x V x V x num_heads x (2*head_dim)

        # Apply attention weights
        attention_scores = torch.einsum('bijnk,hnk->bijh', h_concat, self.a)  # B x V x V x num_heads
        attention_scores = self.leaky_relu(attention_scores)

        # Apply edge gates as attention mask (average across hidden dim)
        edge_mask = torch.mean(edge_gate, dim=-1).unsqueeze(-1)  # B x V x V x 1
        attention_scores = attention_scores * edge_mask

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=2)  # B x V x V x num_heads
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        h_values = h.unsqueeze(1).expand(-1, V, -1, -1, -1)  # B x V x V x num_heads x head_dim
        attended_values = torch.einsum('bijh,bijhk->bihk', attention_weights, h_values)  # B x V x num_heads x head_dim

        # Concatenate heads
        attended_values = attended_values.contiguous().view(B, V, H)

        # Final projection
        x_new = self.output_proj(attended_values)

        return x_new

class ResidualGATv2Layer(nn.Module):
    """GATv2 layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(ResidualGATv2Layer, self).__init__()
        self.node_feat = GATv2NodeFeatures(hidden_dim, num_heads, dropout)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)

        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        e_in = e
        x_in = x

        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H

        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)

        # Node convolution with GATv2
        x_tmp = self.node_feat(x_in, edge_gate)

        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)

        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)

        # Residual connection
        x_new = x_in + x
        e_new = e_in + e

        return x_new, e_new

class ResidualGATv2Model(nn.Module):
    """Residual GATv2 Model for outputting predictions as edge adjacency matrices.
    """

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGATv2Model, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        # Define net parameters
        self.num_nodes = config['num_nodes']
        self.node_dim = config['node_dim']
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['gat_v2_hidden_dim']
        self.num_layers = config['gat_v2_num_layers']
        self.mlp_layers = config['mlp_layers']

        # GATv2 specific parameters
        self.num_heads = 8
        self.dropout = 0.1

        # Node and edge embedding layers
        # Using GATv2Conv for initial node coordinate embedding
        self.nodes_coord_embedding = GATv2Conv(
            self.node_dim,
            self.hidden_dim,
            heads=self.num_heads,
            concat=False,
            dropout=self.dropout,
            share_weights=False  # GATv2 specific parameter
        )

        self.edges_values_embedding = nn.Linear(1, self.hidden_dim//2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim//2)

        # Define GATv2 Layers
        gat_layers = []
        for layer in range(self.num_layers):
            gat_layers.append(ResidualGATv2Layer(self.hidden_dim, self.num_heads, self.dropout))
        self.gat_layers = nn.ModuleList(gat_layers)

        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)

    def loss_edges(self, y_pred_edges, y_edges, edge_cw):
        """
        Loss function for edge predictions.

        Args:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            loss_edges: Value of loss function
        """
        # Edge loss
        y = F.log_softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
        y = y.permute(0, 3, 1, 2)  # B x voc_edges x V x V
        loss_edges = nn.NLLLoss(edge_cw)
        loss = loss_edges(y.contiguous(), y_edges)
        return loss

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None):
        """
        Args:
            x_edges: Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
            x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
            x_nodes: Input nodes (batch_size, num_nodes)
            x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
            y_edges: Targets for edges (batch_size, num_nodes, num_nodes)
            edge_cw: Class weights for edges loss

        Returns:
            y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
            loss: Value of loss function
        """
        batch_size = x_edges.size(0)
        actual_num_nodes = x_edges.size(1)  # Gunakan ukuran aktual dari input
        device = x_edges.device

        # Create batched graph for GATv2Conv - GPU efficient
        # Flatten batch and create offset edge indices
        x_nodes_coord_flat = x_nodes_coord.view(-1, x_nodes_coord.size(-1))  # (B*V, node_dim)

        # Create batched edge indices with proper offsets
        edge_indices = []
        for i in range(batch_size):
            edge_index = x_edges[i].nonzero().t().contiguous()
            edge_index = edge_index + i * actual_num_nodes  # offset by graph position
            edge_indices.append(edge_index)

        if edge_indices:
            edge_index_batched = torch.cat(edge_indices, dim=1)  # (2, total_edges)
        else:
            edge_index_batched = torch.empty((2, 0), dtype=torch.long, device=device)

        # Apply GATv2Conv to batched graph
        x_flat = self.nodes_coord_embedding(x_nodes_coord_flat, edge_index_batched)

        # Reshape back to batch format
        x = x_flat.view(batch_size, actual_num_nodes, -1)  # B x V x H

        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        e = torch.cat((e_vals, e_tags), dim=3)

        # GATv2 layers
        for layer in range(self.num_layers):
            x, e = self.gat_layers[layer](x, e)  # B x V x H, B x V x V x H

        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out

        loss = np.inf
        if y_edges is not None and edge_cw is not None:
            loss = self.loss_edges(y_pred_edges.to(device), y_edges.to(device), edge_cw.to(device))

        return y_pred_edges, loss
    
#@title Hyperparameters

num_nodes = -1 #@param # Could also be 10, 20, or 30!
gcn_hidden_dim = 300 #@param
gat_hidden_dim = 256 #@param
gat_v2_hidden_dim = 256+48 #@param
gcn_num_layers = 5 #@param
gat_num_layers = 5 #@param
gat_v2_num_layers = 5 #@param
mlp_layers = 2 #@param
learning_rate = 0.001 #@param
batches_per_epoch = 128
load_gcn = False #@param {type:"boolean"}
load_gat = False #@param {type:"boolean"}
train_gcn = True #@param {type:"boolean"}
train_gat = True #@param {type:"boolean"}

variables = {
             'num_nodes': num_nodes,
             'node_dim': 2 ,
             'voc_nodes_in': 2,
             'voc_nodes_out': 2,
             'voc_edges_in': 3,
             'voc_edges_out': 2,
             'gcn_hidden_dim': gcn_hidden_dim,
             'gat_hidden_dim': gat_hidden_dim,
             'gat_v2_hidden_dim': gat_v2_hidden_dim,
             'gcn_num_layers': gcn_num_layers,
             'gat_num_layers': gat_num_layers,
             'gat_v2_num_layers': gat_v2_num_layers,
             'mlp_layers': mlp_layers,
             'aggregation': 'mean',
             'val_every': 5,
             'test_every': 5,
             'batches_per_epoch': batches_per_epoch,
             'accumulation_steps': 1,
             'learning_rate': learning_rate,
             'decay_rate': 1.01
             }