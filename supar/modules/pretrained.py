# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from supar.utils.fn import pad
from supar.utils.tokenizer import TransformerTokenizer


class TransformerEmbedding(nn.Module):
    r"""
    Bidirectional transformer embeddings of words from various transformer architectures :cite:`devlin-etal-2019-bert`.

    Args:
        name (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 10.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        mix_dropout (float):
            The dropout ratio of BERT layers. This value will be passed into the :class:`ScalarMix` layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        name: str,
        n_layers: int,
        n_out: int = 0,
        stride: int = 256,
        pooling: str = 'mean',
        pad_index: int = 0,
        mix_dropout: float = .0,
        finetune: bool = False
    ) -> TransformerEmbedding:
        super().__init__()

        from transformers import AutoModel
        try:
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=True)
        except Exception:
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=False)
        self.model = self.model.requires_grad_(finetune)
        self.tokenizer = TransformerTokenizer(name)

        self.name = name
        self.n_layers = n_layers or self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.pooling = pooling
        self.pad_index = pad_index
        self.mix_dropout = mix_dropout
        self.finetune = finetune
        self.max_len = int(max(0, self.model.config.max_position_embeddings) or 1e12) - 2
        self.stride = min(stride, self.max_len)

        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.name}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.mix_dropout > 0:
            s += f", mix_dropout={self.mix_dropout}"
        if self.finetune:
            s += f", finetune={self.finetune}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            tokens (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.

        Returns:
            ~torch.Tensor:
                Contextualized token embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        mask = tokens.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        tokens = pad(tokens[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        token_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        x = self.model(tokens[:, :self.max_len], attention_mask=token_mask[:, :self.max_len].float())[-1]
        # [batch_size, max_len, hidden_size]
        x = self.scalar_mix(x[-self.n_layers:])
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride, (tokens.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.model(tokens[:, i:i+self.max_len], attention_mask=token_mask[:, i:i+self.max_len].float())[-1]
            x = torch.cat((x, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        lens = lens.masked_fill_(lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        x = x.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), x[token_mask])
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            x = x[:, :, 0]
        elif self.pooling == 'last':
            x = x.gather(2, (lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        elif self.pooling == 'mean':
            x = x.sum(2) / lens.unsqueeze(-1)
        else:
            raise RuntimeError(f'Unsupported pooling method "{self.pooling}"!')
        return self.projection(x)

class TransformerEmbedWithRelations(nn.Module):
    r"""
    Bidirectional transformer embeddings with realtions extends of words from various transformer architectures :cite:`devlin-etal-2019-bert`.
    
    Args:
        name (str):
            Path or name of the pretrained models registered in `transformers`_, e.g., ``'bert-base-cased'``.
        n_layers (int):
            The number of BERT layers to use. If 0, uses all layers.
        n_out (int):
            The requested size of the embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        stride (int):
            A sequence longer than max length will be splitted into several small pieces
            with a window size of ``stride``. Default: 10.
        pooling (str):
            Pooling way to get from token piece embeddings to token embedding.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        pad_index (int):
            The index of the padding token in BERT vocabulary. Default: 0.
        mix_dropout (float):
            The dropout ratio of BERT layers. This value will be passed into the :class:`ScalarMix` layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        name: str,
        n_layers: int,
        n_out: int = 0,
        stride: int = 256,
        pooling: str = 'mean',
        pad_index: int = 0,
        mix_dropout: float = .0,
        atten_layer: int = 3,
        finetune: bool = False
    ) -> TransformerEmbedding:
        super().__init__()

        from transformers import AutoModel
        try:
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=True)
        except Exception:
            self.model = AutoModel.from_pretrained(name, output_hidden_states=True, local_files_only=False)
        self.model = self.model.requires_grad_(finetune)
        self.tokenizer = TransformerTokenizer(name)

        self.name = name
        self.n_layers = n_layers or self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.n_out = n_out or self.hidden_size
        self.pooling = pooling
        self.pad_index = pad_index
        self.mix_dropout = mix_dropout
        self.finetune = finetune
        self.max_len = int(max(0, self.model.config.max_position_embeddings) or 1e12) - 2
        self.stride = min(stride, self.max_len)
        self.hook_results = {}
        
        self.model.encoder.layer[atten_layer].attention.self.register_forward_hook(self.relations_hook)
        self.scalar_mix = ScalarMix(self.n_layers, mix_dropout)
        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def relations_hook(self, model, input, output):
        self.hook_results['model'] = model
        self.hook_results['input'] = input
        self.hook_results['output'] = output
        
    def get_raw_attentions(self):
        query_layer = self.hook_results['model'] \
            .transpose_for_scores(self.hook_results['model'] \
                .query(self.hook_results['input'][0]))
        key_layer = self.hook_results['model'] \
            .transpose_for_scores(self.hook_results['model'] \
                .key(self.hook_results['input'][0]))
            
        return torch.matmul(query_layer, key_layer.transpose(-1, -2)).detach().clone()
            
    def __repr__(self):
        s = f"{self.name}, n_layers={self.n_layers}, n_out={self.n_out}, "
        s += f"stride={self.stride}, pooling={self.pooling}, pad_index={self.pad_index}"
        if self.mix_dropout > 0:
            s += f", mix_dropout={self.mix_dropout}"
        if self.finetune:
            s += f", finetune={self.finetune}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            tokens (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.

        Returns:
            ~torch.Tensor:
                Contextualized token embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        mask = tokens.ne(self.pad_index)
        lens = mask.sum((1, 2))
        # [batch_size, n_subwords]
        tokens = pad(tokens[mask].split(lens.tolist()), self.pad_index, padding_side=self.tokenizer.padding_side)
        token_mask = pad(mask[mask].split(lens.tolist()), 0, padding_side=self.tokenizer.padding_side)

        # return the hidden states of all layers
        x = self.model(tokens[:, :self.max_len], attention_mask=token_mask[:, :self.max_len].float())[-1]
        # get raw attention score from inputs [batch_size, n_heads, n_subwords, n_subwords]
        # todo: remember to devide by d_k
        attention_scores = self.get_raw_attentions() / math.sqrt(self.hook_results['model'].attention_head_size)
        # attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        # [batch_size, max_len, hidden_size]
        x = self.scalar_mix(x[-self.n_layers:])
        # [batch_size, n_subwords, hidden_size]
        for i in range(self.stride, (tokens.shape[1]-self.max_len+self.stride-1)//self.stride*self.stride+1, self.stride):
            part = self.model(tokens[:, i:i+self.max_len], attention_mask=token_mask[:, i:i+self.max_len].float())[-1]
            part_attentions = self.get_raw_attentions() / math.sqrt(self.hook_results['model'].attention_head_size)
            part_attentions = nn.functional.softmax(part_attentions, dim=-1)
            # pad inexistent relations
            passed_stride = i//self.stride
            num_pad_eles = self.stride if tokens.shape[-1] > (passed_stride+1)*self.max_len else tokens.shape[-1] - passed_stride 
            attention_scores = F.pad(attention_scores, (0, num_pad_eles,0, num_pad_eles), "constant", -1e-9)
            attention_scores[:, :, -part_attentions.shape[-1]:, -num_pad_eles:] = part_attentions[:, :, :, self.max_len-self.stride:]
            attention_scores[:, :, -num_pad_eles:, -part_attentions.shape[-1]:] = part_attentions[:, :, self.max_len-self.stride:, :]
            x = torch.cat((x, self.scalar_mix(part[-self.n_layers:])[:, self.max_len-self.stride:]), 1)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        lens = lens.masked_fill_(lens.eq(0), 1)
        # [batch_size, seq_len, fix_len, hidden_size]
        x = x.new_zeros(*mask.shape, self.hidden_size).masked_scatter_(mask.unsqueeze(-1), x[token_mask])
        # define dimensions 
        num_heads = attention_scores.shape[1]
        batch_size = attention_scores.shape[0]
        num_tokens = attention_scores.shape[-1]
        # mask raw attention scores batch x 1 x 1 x num_toks
        attention_mask = token_mask[:, None, None, :].float()
        # filter out zero scores
        attention_scores[attention_scores == 0] = -1e-9
        # keep the scores at none zero mask elements only
        masked_attentions = attention_scores*attention_mask
        # get attention score for filling
        temp_att_mask = masked_attentions.ne(0)
        # expand mask to rationale for filling scores 
        expand_mask = mask.repeat_interleave(torch.tensor([num_tokens*num_heads]*batch_size, device=mask.get_device()), dim=0).bool()
        # fill attentions 
        filled_attn_scores = attention_scores.new_zeros(*expand_mask.shape).masked_scatter_(expand_mask,
                                                                                            masked_attentions[temp_att_mask])
        # todo: consider this thing 
        filled_attn_scores[filled_attn_scores == -1e-9] = 0
        # reshape attentions to batch_sizexnum_headsxnum_toksxseq_lenxfix_len
        filled_attn_scores = filled_attn_scores.reshape(batch_size, num_heads, num_tokens, mask.shape[1], mask.shape[-1])
        # currently consider mean summerize over subwords only
        # to do: extend to first subword and last subword
        filled_attn_scores = filled_attn_scores.mean(-1)
        # compute mask to get rid of irrelevant rows
        # batch x seq_len
        final_indices = mask.sum(-1).cumsum(-1) - 1 
        # expand dim 1 to hold the heads
        final_indices = final_indices[:, None, :]
        final_indices = final_indices.repeat_interleave(num_heads, dim=1) 
        final_indices = final_indices[:, :, :, None].expand(batch_size, num_heads, mask.shape[1], mask.shape[1])
    
        # attentions score after removing irrelevant rows batch x heads x seq_len x seq_len 
        final_attn_scores = filled_attn_scores.gather(-2, final_indices)
        # [batch_size, seq_len, hidden_size]
        if self.pooling == 'first':
            x = x[:, :, 0]
        elif self.pooling == 'last':
            x = x.gather(2, (lens-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        elif self.pooling == 'mean':
            x = x.sum(2) / lens.unsqueeze(-1)
        else:
            raise RuntimeError(f'Unsupported pooling method "{self.pooling}"!')
        return self.projection(x), final_attn_scores

class ELMoEmbedding(nn.Module):
    r"""
    Contextual word embeddings using word-level bidirectional LM :cite:`peters-etal-2018-deep`.

    Args:
        name (str):
            The name of the pretrained ELMo registered in `OPTION` and `WEIGHT`. Default: ``'original_5b'``.
        bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of sentence outputs.
            Default: ``(True, True)``.
        n_out (int):
            The requested size of the embeddings. If 0, uses the default size of ELMo outputs. Default: 0.
        dropout (float):
            The dropout ratio for the ELMo layer. Default: 0.
        finetune (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.
    """

    OPTION = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',  # noqa
    }
    WEIGHT = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',  # noqa
    }

    def __init__(
        self,
        name: str = 'original_5b',
        bos_eos: Tuple[bool, bool] = (True, True),
        n_out: int = 0,
        dropout: float = 0.5,
        finetune: bool = False
    ) -> ELMoEmbedding:
        super().__init__()

        from allennlp.modules import Elmo

        self.elmo = Elmo(options_file=self.OPTION[name],
                         weight_file=self.WEIGHT[name],
                         num_output_representations=1,
                         dropout=dropout,
                         finetune=finetune,
                         keep_sentence_boundaries=True)

        self.name = name
        self.bos_eos = bos_eos
        self.hidden_size = self.elmo.get_output_dim()
        self.n_out = n_out or self.hidden_size
        self.dropout = dropout
        self.finetune = finetune

        self.projection = nn.Linear(self.hidden_size, self.n_out, False) if self.hidden_size != n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.name}, n_out={self.n_out}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        if self.finetune:
            s += f", finetune={self.finetune}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, chars: torch.LongTensor) -> torch.Tensor:
        r"""
        Args:
            chars (~torch.LongTensor): ``[batch_size, seq_len, fix_len]``.

        Returns:
            ~torch.Tensor:
                ELMo embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        x = self.projection(self.elmo(chars)['elmo_representations'][0])
        if not self.bos_eos[0]:
            x = x[:, 1:]
        if not self.bos_eos[1]:
            x = x[:, :-1]
        return x


class ScalarMix(nn.Module):
    r"""
    Computes a parameterized scalar mixture of :math:`N` tensors, :math:`mixture = \gamma * \sum_{k}(s_k * tensor_k)`
    where :math:`s = \mathrm{softmax}(w)`, with :math:`w` and :math:`\gamma` scalar parameters.

    Args:
        n_layers (int):
            The number of layers to be mixed, i.e., :math:`N`.
        dropout (float):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjusts its softmax weight mass to 0
            with the dropout probability (i.e., setting the unnormalized weight to -inf).
            This effectively redistributes the dropped probability mass to all other weights.
            Default: 0.
    """

    def __init__(self, n_layers: int, dropout: float = .0) -> ScalarMix:
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        return f"{self.__class__.__name__}({s})"

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            tensors (List[~torch.Tensor]):
                :math:`N` tensors to be mixed.

        Returns:
            The mixture of :math:`N` tensors.
        """

        return self.gamma * sum(w * h for w, h in zip(self.dropout(self.weights.softmax(-1)), tensors))
