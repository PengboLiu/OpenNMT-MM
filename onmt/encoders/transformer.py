"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as f

from onmt.encoders.encoder import EncoderBase
from onmt.modules import Capsule
from onmt.modules import MultiHeadedAttention
from onmt.modules import GumbelMultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerGumbelSelectLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerGumbelSelectLayer, self).__init__()

        self.self_attn = GumbelMultiHeadedAttention(
            heads, d_model, dropout=attention_dropout, 
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs_image, inputs_text):
        """
        Args:
            inputs_image (FloatTensor): ``(batch_size, image_len, model_dim)``
            inputs_text (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm_image = self.layer_norm(inputs_image)
        input_norm_text = self.layer_norm(inputs_text)
        context, _ = self.self_attn(inputs_image, inputs_image, inputs_text ,
                                    attn_type="context")
        out = self.dropout(context) + input_norm_text
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        
        self.transformer_image = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.gumbel_layer_0 = TransformerGumbelSelectLayer(d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
        
        # self.gumbel_layer_1 = TransformerGumbelSelectLayer(d_model, heads, d_ff, dropout, attention_dropout,
        #         max_relative_positions=max_relative_positions)

        # multi-modal
        self.image_linear = nn.Linear(4096, 128)
        self.image_dropout = nn.Dropout(0.4)

        # self.lamb0 = nn.Parameter(torch.Tensor([1]))
        
        self.multi_modal_linear = nn.Linear(128, 128)
        self.multi_dropout = nn.Dropout(0.4)

        self.local_image_linear = nn.Linear(512, 128)

        # gate module
        self.gate_linear_text_1 = nn.Linear(128, 128)
        self.gate_linear_image_1 = nn.Linear(128, 128)

        self.multi_modal_gate = nn.Sigmoid()

        

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, img_feats, img_feats_local, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        mask = ~sequence_mask(lengths).unsqueeze(1)
        
        emb = self.embeddings(src)
        
        # text_feats_0 = emb

        img_feats_local = img_feats_local.transpose(1, 2)
        img_feats_local = self.local_image_linear(img_feats_local)
        img_feats_local = self.image_dropout(img_feats_local)
        
        out = emb.transpose(0, 1).contiguous()

        text_aware_image = self.gumbel_layer_0(img_feats_local, out)

        # Run the forward pass of every layer of the tranformer.
        layer_num = 0
        for layer in self.transformer:
            # out: [batch, len, dim]
            out = layer(out, mask)
            # text_aware_image = layer(text_aware_image, mask)
            text_aware_image = self.transformer_image[layer_num](text_aware_image, mask)
            layer_num += 1

        text_out = out
        image_out = text_aware_image
        
        multi_gate = self.multi_modal_gate(self.gate_linear_image_1(text_aware_image) + self.gate_linear_text_1(out))
        out = out + multi_gate*text_aware_image

        out = self.layer_norm(out)
        out = out.transpose(0, 1).contiguous()

        return emb, out, lengths, text_out, image_out

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
