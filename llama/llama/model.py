# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 128256 # defined later by tokenizer
    multiple_of: int = 1024  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    prune_layer: int = 32
    max_batch_size: int = 1
    max_seq_len: int = 128000
    chai_activate: bool = True
    chai_layers: list = field(
        default_factory=lambda: [
        48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 
        36, 36, 36, 36, 36, 36, 36, 36, 
        24, 24, 24, 24, 24, 24, 
        8, 8, 
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 ]
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.layer_id = layer_id
        self.chai_activate = args.chai_activate
        self.prune_layer = args.prune_layer
        self.chai_layer_param = args.chai_layers[layer_id]

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # store the Query again
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):

        # NOTE: The first sequence needs to be atleast of size 6
        # if not we throw an error.
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        cluster_assignment_log_per_example = dict()
        if self.layer_id >= self.prune_layer:
            # CHAI
            if start_pos == 0:
                # first sentence

                xk = self.cache_k[:bsz, : start_pos + seqlen]
                values = self.cache_v[:bsz, : start_pos + seqlen]
                xq = xq.view(bsz, self.n_local_heads, seqlen, self.head_dim)
                xk = xk.view(bsz, self.n_local_heads, seqlen, self.head_dim)
                num_examples, num_org_heads, seq_len, head_dim = xq.shape
                xq_four = xq[:, :, :5, :]
                xk_four = xk[:, :, :5, :]

                scores_four = F.softmax(
                    (
                        torch.matmul(xq_four, xk_four.transpose(2, 3))
                        / math.sqrt(self.head_dim)
                    ).float(),
                    dim=-1,
                )
                scores_four_numpy = scores_four.cpu().numpy()
                scores_new_xk_xq = torch.zeros(
                    [num_examples, num_org_heads, seq_len, seq_len],
                    device=xq.device,
                    dtype=xq.dtype,
                )
                xk_new = torch.zeros(
                    [num_examples, self.chai_layer_param, seq_len, head_dim],
                    dtype=xk.dtype,
                    device=xk.device,
                )
                xq_new = torch.zeros(
                    [num_examples, self.chai_layer_param, seq_len, head_dim],
                    dtype=xq.dtype,
                    device=xq.device,
                )

                for ex_id in range(num_examples):
                    assert num_examples == 1
                    temp_data = dict()
                    ex_id_score = scores_four_numpy[ex_id, :]
                    sequence_length_example = ex_id_score.shape[1]
                    # if ex_id_score.shape[1] > 4:
                    # use_small = False
                    num_heads = ex_id_score.shape[0]
                    first_sample_score = ex_id_score.reshape((num_heads, -1))
                    dist_arr = cdist(
                        first_sample_score, first_sample_score, metric="cosine"
                    )
                    cluster = AgglomerativeClustering(
                        n_clusters=self.chai_layer_param,
                        metric="precomputed",
                        linkage="average",
                    )
                    try:
                        cluster = cluster.fit(dist_arr)
                    except:
                        import ipdb

                        ipdb.set_trace()
                    cluster_assignment = cluster.labels_
                    self.grouping = cluster_assignment
                    for cluster_idx in range(self.chai_layer_param):
                        grouped_heads = np.where(cluster_assignment == cluster_idx)[
                            0
                        ].tolist()
                        xk_new[ex_id, cluster_idx, :, :] = xk[
                            ex_id, grouped_heads[0], :, :
                        ]
                        xq_new[ex_id, cluster_idx, :, :] = xq[
                            ex_id, grouped_heads[0], :, :
                        ]
                        temp_data[cluster_idx] = grouped_heads
                    cluster_assignment_log_per_example[ex_id] = temp_data
                    # else:
                    # cluster_assignment_log_per_example[ex_id] = temp_data
                    # xk_new = xk
                    # xq_new = xq
            else:
                # scores
                xk = self.cache_k[:bsz, : start_pos + seqlen]
                values = self.cache_v[:bsz, : start_pos + seqlen]
                xq = xq.view(bsz, self.n_local_heads, 1, self.head_dim)
                xk = xk.view(bsz, self.n_local_heads, start_pos + seqlen, self.head_dim)
                num_examples, num_org_heads, seq_len, head_dim = xk.shape
                scores_new_xk_xq = torch.zeros(
                    [num_examples, num_org_heads, 1, seq_len],
                    device=xq.device,
                    dtype=xq.dtype,
                )
                xk_new = torch.zeros(
                    [num_examples, self.chai_layer_param, seq_len, head_dim],
                    dtype=xk.dtype,
                    device=xk.device,
                )
                xq_new = torch.zeros(
                    [num_examples, self.chai_layer_param, 1, head_dim],
                    dtype=xq.dtype,
                    device=xq.device,
                )
                cluster_assignment = self.grouping
                for ex_id in range(num_examples):
                    temp_data = dict()
                    for cluster_idx in range(self.chai_layer_param):
                        grouped_heads = np.where(cluster_assignment == cluster_idx)[
                            0
                        ].tolist()
                        xk_new[ex_id, cluster_idx, :, :] = xk[
                            ex_id, grouped_heads[0], :, :
                        ]
                        xq_new[ex_id, cluster_idx, :, :] = xq[
                            ex_id, grouped_heads[0], :, :
                        ]
                        temp_data[cluster_idx] = grouped_heads
                    cluster_assignment_log_per_example[ex_id] = temp_data

            scores_new_temp = torch.matmul(xq_new, xk_new.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            # if use_small:
            # putting them back together
            for ex_id in range(num_examples):
                for cluster_idx in range(self.chai_layer_param):
                    scores_new_xk_xq[
                        ex_id,
                        cluster_assignment_log_per_example[ex_id][cluster_idx],
                        :,
                        :,
                    ] = scores_new_temp[ex_id, cluster_idx, :, :]
            # else:
            # scores_new_xk_xq = scores_new_temp
            if mask is not None:
                scores_new_xk_xq = scores_new_xk_xq + mask
            scores_new_xk_xq = F.softmax(scores_new_xk_xq.float(), dim=-1).type_as(xq)
            scores = scores_new_xk_xq
            values = values.transpose(1, 2)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(output)
        else:
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.chai_activate = args.chai_activate
        self.prune_layer = args.prune_layer
        self.chai_layer_param = args.chai_layers[layer_id]
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        self.prune_layer = params.prune_layer
        self.chai_activate = params.chai_activate
        self.chai_layers = params.chai_layers
        for layer_id in range(params.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id,
                    params,
                )
            )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float()
