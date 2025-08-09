import torch.nn as nn
import torch
from .mlp import MultiLayerPerceptron, GraphMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, ):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)
        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.argumented_linear = nn.Linear(model_dim, model_dim)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2, c=None, augment=False):
        x = x.transpose(dim, -2)
        augmented = None
        # x: (batch_size, ..., length, model_dim)
        if c is not None:
            residual = c
        else:
            residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
            if augment is True:
                augmented = self.act1(self.argumented_linear(residual))
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)

        out = self.dropout1(out)

        if augmented is not None and augment is not False:
            out = self.ln1(residual + out + augmented)
        else:
            out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class DSTRformer(nn.Module):
    """
    Paper: STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
    Link: https://arxiv.org/abs/2308.10425
    Official Code: https://github.com/XDZhelheim/STAEformer
    """

    def __init__(
            self,
            num_nodes,
            adj_mx,
            in_steps,
            out_steps,
            steps_per_day,
            input_dim,
            output_dim,
            input_embedding_dim,
            tod_embedding_dim,
            ts_embedding_dim,
            dow_embedding_dim,
            time_embedding_dim,
            adaptive_embedding_dim,
            node_dim,
            feed_forward_dim,
            out_feed_forward_dim,
            num_heads,
            num_layers,
            num_layers_m,
            mlp_num_layers,
            dropout,
            use_mixed_proj,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.adj_mx = adj_mx
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.ts_embedding_dim = ts_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.node_dim = node_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + adaptive_embedding_dim
                + ts_embedding_dim
                + time_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.num_layers_m = num_layers_m
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(7 * steps_per_day, self.time_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        self.adj_mx_forward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )

        self.adj_mx_backward_encoder = nn.Sequential(
            GraphMLP(input_dim=self.num_nodes, hidden_dim=self.node_dim)
        )


        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        ####################

        self.attn_layers_c = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            ]
        )
        self.ar_attn = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, out_feed_forward_dim, num_heads,
                                   dropout)
                for _ in range(num_layers_m)
            ]
        )
        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps, out_channels=self.ts_embedding_dim, kernel_size=(1, 1),
                bias=True)

        self.fusion_model = nn.Sequential(
            *[MultiLayerPerceptron(input_dim=self.adaptive_embedding_dim + 2 * self.node_dim,
                                   hidden_dim=self.adaptive_embedding_dim + 2 * self.node_dim,
                                   dropout=0.2)
              for _ in range(mlp_num_layers)],
            nn.Linear(in_features=self.adaptive_embedding_dim + 2 * self.node_dim, out_features=self.adaptive_embedding_dim, bias=True)
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)

        x = history_data
        batch_size, _, num_nodes, _ = x.shape

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        if self.time_embedding_dim > 0:
            tod = x[..., 1]
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        if self.ts_embedding_dim > 0:
            input_data = x.transpose(1, 2).contiguous()
            input_data = input_data.view(
                batch_size, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            # B L*3 N 1
            time_series_emb = self.time_series_emb_layer(input_data)
            time_series_emb = time_series_emb.transpose(1, -1).expand(batch_size, self.in_steps, self.num_nodes,
                                                                      self.ts_embedding_dim)
        # B ts_embedding_dim N 1

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)

        if self.tod_embedding_dim > 0:
            tod_index = (tod * self.steps_per_day).clamp(0, self.steps_per_day - 1).long()
            tod_emb = self.tod_embedding(tod_index)
            features.append(tod_emb)

        if self.dow_embedding_dim > 0:
            dow_index = dow.clamp(0, 6).long()
            dow_emb = self.dow_embedding(dow_index)
            features.append(dow_emb)

        if self.time_embedding_dim > 0:
            tod_index = (tod * self.steps_per_day).clamp(0, self.steps_per_day - 1).long()
            dow_index = dow.clamp(0, 6).long()
            time_index = (dow_index * self.steps_per_day + tod_index).clamp(0, 7 * self.steps_per_day - 1)
            time_emb = self.time_embedding(time_index)
            features.append(time_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)

        for i, f in enumerate(features):
            print(f"[DEBUG] feature[{i}] shape: {f.shape}")

        temporal_x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        spatial_x = temporal_x.clone()

        for index, (attn_t, attn_s) in enumerate(zip(self.attn_layers_t, self.attn_layers_s)):
            temporal_x = attn_t(temporal_x, dim=1)
            spatial_x = attn_s(spatial_x, dim=2)

        for attn in self.attn_layers_c:
            x = attn(temporal_x, spatial_x, dim=2)
        if self.node_dim > 0:
            adp_graph = x[..., -self.adaptive_embedding_dim:]
            x = x[..., :self.model_dim - self.adaptive_embedding_dim]

            node_forward = self.adj_mx[0].to(device)
            node_forward = self.adj_mx_forward_encoder(node_forward.unsqueeze(0)).expand(batch_size, self.in_steps, -1,
                                                                                         -1)
            node_backward = self.adj_mx[1].to(device)
            node_backward = self.adj_mx_backward_encoder(node_backward.unsqueeze(0)).expand(batch_size, self.in_steps,
                                                                                           -1,
                                                                                           -1)
            graph = torch.cat([adp_graph, node_forward, node_backward], dim=-1)
            graph = self.fusion_model(graph)

            x = torch.cat([x, graph], dim=-1)
        else:
            for attn in self.attn_out:
                x = attn(x, dim=2)

        for attn in self.ar_attn:
            x = attn(x, dim=2, augment=True)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        return out
