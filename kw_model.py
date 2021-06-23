import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv


class KW_GNN(torch.nn.Module):
    def __init__(self, embed_size, vocab_size, keyword_vocab_size, hidden_size, output_size, n_layers, gnn, aggregation,
                 n_heads=0, dropout=0, bidirectional=False, \
                 utterance_encoder="", keywordid2wordid=None, keyword_mask_matrix=None, nodeid2wordid=None,
                 keywordid2nodeid=None, concept_encoder="mean", \
                 combine_node_emb="mean"):
        super(KW_GNN, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.keyword_vocab_size = keyword_vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gnn = gnn
        self.aggregation = aggregation
        self.n_heads = n_heads
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.utterance_encoder_name = utterance_encoder
        self.keywordid2wordid = keywordid2wordid
        self.keyword_mask_matrix = keyword_mask_matrix
        self.nodeid2wordid = nodeid2wordid
        self.keywordid2nodeid = keywordid2nodeid
        self.concept_encoder = concept_encoder
        self.combine_node_emb = combine_node_emb
        self.num_nodes = nodeid2wordid.shape[0]

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GNN learning
        if gnn == "GatedGraphConv":
            self.conv1 = GatedGraphConv(hidden_size, num_layers=n_layers)
            output_size = hidden_size

        if n_layers == 1:
            output_size = hidden_size

        # aggregation
        if aggregation in ["mean", "max"]:
            output_size = output_size

        # utterance encoder
        if self.utterance_encoder_name == "HierGRU":
            self.utterance_encoder = nn.GRU(embed_size, hidden_size, 1, batch_first=True, dropout=dropout,
                                            bidirectional=bidirectional)
            self.context_encoder = nn.GRU(2 * hidden_size if bidirectional else hidden_size, hidden_size, 1,
                                          batch_first=True, dropout=dropout, bidirectional=bidirectional)
            output_size = output_size + 2 * hidden_size if bidirectional else output_size + hidden_size

        # final linear layer
        self.mlp = nn.Linear(output_size, keyword_vocab_size)

    def forward_gnn(self, emb, edge_index):
        # emb: (keyword_vocab_size, emb_size)
        # edge_index: (2, num_edges)
        # edge_type: None or (num_edges, )
        # edge_weight: None or (num_edges, )
        if self.gnn in ["GatedGraphConv"]:
            out = self.conv1(emb, edge_index)  # (keyword_vocab_size, hidden_size)
        return out

    def forward_aggregation(self, out, x):
        # out: (keyword_vocab_size, output_size)
        # x: (batch_size, seq_len)
        if self.aggregation == "mean":
            x_mask = x.ne(0).float()  # (batch_size, seq_len)
            out = out[x]  # (batch_size, seq_len, output_size)
            out = (out * x_mask.unsqueeze(-1)).sum(dim=1) / x_mask.sum(dim=-1, keepdim=True).clamp(
                min=1)  # (batch_size, output_size)
        if self.aggregation == "max":
            x_mask = x.ne(0).float()  # (batch_size, seq_len)
            out = out[x]  # (batch_size, seq_len, output_size)
            out = torch.max(out * x_mask.unsqueeze(-1) + (-5e4) * (1 - x_mask.unsqueeze(-1)), dim=1)[
                0]  # (batch_size, output_size)
        return out

    def forward_utterance(self, x):
        # x: None or (batch_size, context_len, seq_len)
        batch_size, context_len, seq_len = x.shape
        # print(x.shape)
        # print(x.max())
        # print(self.embedding.weight.shape)
        if self.utterance_encoder_name == "HierGRU":
            seq_lengths = x.reshape(-1, seq_len).ne(0).sum(dim=-1)  # (batch_size*context_len, )
            context_lengths = seq_lengths.reshape(batch_size, -1).ne(0).sum(dim=-1)  # (batch_size, )
            out = self.embedding(x)  # (batch_size, context_len, seq_len, emb_size)
            out, _ = self.utterance_encoder(out.reshape(batch_size * context_len, seq_len,
                                                        -1))  # out: (batch_size*context_len, seq_len, num_directions * hidden_size)
            out = out[torch.arange(batch_size * context_len), (seq_lengths - 1).clamp(min=0),
                  :]  # out: (batch_size*context_len, num_directions * hidden_size)
            out, _ = self.context_encoder(out.reshape(batch_size, context_len,
                                                      -1))  # out: (batch_size, context_len, num_directions * hidden_size)
            out = out[torch.arange(batch_size), (context_lengths - 1).clamp(min=0),
                  :]  # out: (batch_size, num_directions * hidden_size)
            return out
        return out

    def forward_concept(self, emb, nodeid2wordid):
        # emb: (vocab_size, emb_size)
        # nodeid2wordid: (num_nodes, 10)
        mask = nodeid2wordid.ne(0).float()  # (num_nodes, 10)
        if self.concept_encoder == "mean":
            node_emb = (emb[nodeid2wordid] * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(
                min=1)  # (num_nodes, emb_size)
        if self.concept_encoder == "max":
            node_emb = (emb[nodeid2wordid] * mask.unsqueeze(-1) + (-5e4) * (1 - mask.unsqueeze(-1))).max(dim=1)[
                0]  # (num_nodes, emb_size)
        return node_emb

    def forward(self, CN_hopk_edge_index, x, x_utter=None, x_concept=None):
        # CN_hopk_edge_index: (2, num_edges)
        # x: (batch_size, seq_len)
        # x_utter: None or (batch_size, context_len, max_sent_len)
        # x_concept: None or (batch_size, max_sent_len)

        # graph convolution
        emb = self.embedding.weight  # (keyword_vocab_size, emb_size)
        CN_hopk_out = None
        if CN_hopk_edge_index is not None:
            node_emb = self.forward_concept(emb.to('cuda'), self.nodeid2wordid.to('cuda'))
            CN_hopk_out = self.forward_gnn(node_emb, CN_hopk_edge_index)

        # aggregation
        if CN_hopk_edge_index is not None:
            x = self.keywordid2nodeid[x]  # (batch_size, keyword_seq_len)
            CN_hopk_keyword_out = self.forward_aggregation(CN_hopk_out, x)

        # concept aggregation
        if CN_hopk_edge_index is not None and x_concept is not None:
            CN_hopk_concept_out = self.forward_aggregation(CN_hopk_out, x_concept)  # (batch_size, output_size)
            # print("CN_hopk_concept_out: ", CN_hopk_concept_out.shape)

            if self.combine_node_emb == "mean":
                CN_hopk_out = (CN_hopk_keyword_out + CN_hopk_concept_out) / 2
            if self.combine_node_emb == "max":
                CN_hopk_out = torch.stack([CN_hopk_keyword_out, CN_hopk_concept_out], dim=0).max(dim=0)[0]

        # combine two graphs
        if CN_hopk_edge_index is not None:
            if x_concept is None:
                out = CN_hopk_keyword_out
            else:
                out = CN_hopk_out

        # utterance encoder
        if self.utterance_encoder_name != "":
            utter_out = self.forward_utterance(x_utter)
            out = torch.cat([out, utter_out], dim=-1)  # (batch_size, *)

        # final linear layer
        out = self.mlp(out)  # out: (batch_size, keyword_vocab_size)
        return out

    def init_embedding(self, embedding, fix_word_embedding):
        print("initializing word embedding layer...")
        self.embedding.weight.data.copy_(embedding)
        if fix_word_embedding:
            self.embedding.weight.requires_grad = False
