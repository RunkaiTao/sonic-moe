# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os

import torch
import torch.nn.functional as F

from quack.gemm_interface import gemm

from ..count_cumsum import count_cumsum
from ..quack_utils import gemm_dgated, gemm_gated
from .backward import _down_projection_backward, _softmax_topk_bwd, _token_broadcast_backward, _up_projection_backward
from .forward import _down_projection_forward, _router_forward, _softmax_topk_fwd, _up_projection_forward
from .utils import enable_quack_gemm, is_using_quack_gemm


def TC_topk_router_metadata(
    topk_router_indices: torch.Tensor, expert_offset, K: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_scatter_idx = torch.argsort(topk_router_indices.view(-1)).int()
    expert_offset = torch.cat([torch.zeros(1, device=expert_offset.device, dtype=expert_offset.dtype), expert_offset])
    # s_reverse_scatter_idx = torch.argsort(s_scatter_idx).int()
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.shape[0], device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    topk_x_offset = torch.arange(
        0, topk_router_indices.shape[0] * K + 1, K, dtype=torch.int32, device=topk_router_indices.device
    )
    x_gather_idx = s_scatter_idx // K

    return expert_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, topk_x_offset


def general_routing_router_metadata(
    router_scores_selected: torch.Tensor, sorted_selected_T: torch.Tensor, selected_E: torch.Tensor, T: int, E: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    device = router_scores_selected.device

    expert_frequency, expert_offset = count_cumsum(selected_E, E, do_cumsum=True)
    expert_offset = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), expert_offset])

    s_scatter_idx = selected_E.argsort().int()
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.size(0), device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    x_gather_idx = sorted_selected_T[s_scatter_idx]

    if T % 4 == 0:
        _, topk_token_offset = count_cumsum(sorted_selected_T, T, do_cumsum=True)
    else:
        topk_token_offset = torch.bincount(sorted_selected_T, minlength=T).int()

    topk_token_offset = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), topk_token_offset])

    return expert_frequency, expert_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, topk_token_offset


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, router_logits: torch.Tensor, E: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        # change this to router_logits.dtype (bfloat16) increase another 5 tflops at fwd at the cost of numerical accuracy
        topk_router_score = torch.empty(T, K, dtype=torch.float32, device=router_logits.device)
        topk_router_indices = torch.empty(T, K, dtype=torch.int32, device=router_logits.device)

        _softmax_topk_fwd(router_logits, topk_router_score, topk_router_indices, E, K)

        ctx.save_for_backward(topk_router_score, topk_router_indices)
        ctx.E = E
        ctx.dtype = router_logits.dtype

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(ctx, dtopk_score: torch.Tensor, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, K = dtopk_score.size()

        topk_router_score, topk_router_indices = ctx.saved_tensors
        dlogits = torch.zeros(T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device)

        _softmax_topk_bwd(dlogits, None, dtopk_score, topk_router_score, topk_router_indices, K)

        return dlogits, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        topk_token_offset: torch.Tensor,
        is_varlen_K: bool,
        is_inference_mode_enabled: bool,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        I //= 2
        TK = total_expert_freq

        if is_using_quack_gemm():
            assert not torch.compiler.is_compiling()

            z, y1 = gemm_gated(
                x,
                w1.permute(2, 1, 0),
                activation="swiglu",
                cu_seqlens_m=expert_offset,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
            )
        else:
            z = torch.empty(TK, 2 * I, dtype=x.dtype, device=x.device)
            y1 = torch.empty(TK, I, dtype=x.dtype, device=x.device)
            _up_projection_forward(
                x=x,
                w1=w1,
                z=z,
                y1=y1,
                b1=b1,
                expert_offset=expert_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
                is_inference_mode_enabled=is_inference_mode_enabled,
            )

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_varlen_K = is_varlen_K
        ctx.stream_id = stream_id

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            topk_token_offset,
        )

        ctx.mark_non_differentiable(y1)
        ctx.set_materialize_grads(False)

        return y1, z

    @staticmethod
    def backward(ctx, _: None, dz: torch.Tensor):
        is_compiling = torch.compiler.is_compiling()

        if not is_compiling:
            assert _ is None

        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_varlen_K = ctx.is_varlen_K
        stream_id = ctx.stream_id

        (
            x,
            w1,
            b1,
            expert_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            topk_token_offset,
        ) = ctx.saved_tensors

        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        if is_using_quack_gemm():
            assert not is_compiling

            gemm(
                x.T,
                dz,
                out=dw1.permute(2, 1, 0),
                cu_seqlens_k=expert_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )
            dx_expanded = gemm(dz, w1.permute(2, 0, 1), cu_seqlens_m=expert_offset, dynamic_scheduler=False)
        else:
            dx_expanded = torch.empty(TK, H, dtype=dz.dtype, device=dz.device)
            _up_projection_backward(
                x=x,
                w1=w1,
                dx_expanded=dx_expanded,
                dw1=dw1,
                dz=dz,
                db1=db1,
                expert_offset=expert_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                stream_id=stream_id,
            )

        dx_reduced = torch.empty(T, H, dtype=dz.dtype, device=dz.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            topk_token_offset=topk_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        return dx_reduced, dw1, db1, *[None] * 10


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y1: torch.Tensor,
        z: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_offset: torch.Tensor,
        T: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        topk_token_offset: torch.Tensor,
        is_varlen_K: bool,
    ) -> torch.Tensor:
        TK = y1.size(0)
        H, I, E = w2.shape

        if is_using_quack_gemm():
            assert not torch.compiler.is_compiling()

            assert b2 is None
            y2 = gemm(y1, w2.permute(2, 1, 0), cu_seqlens_m=expert_offset)
        else:
            y2 = torch.empty(TK, H, dtype=y1.dtype, device=y1.device)
            _down_projection_forward(
                w2=w2,
                y1=y1,
                y2=y2,
                b2=b2,
                expert_offset=expert_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
            )

        o = torch.empty(T, H, device=z.device, dtype=z.dtype)
        topk_scores = topk_scores.flatten()

        _router_forward(
            y2=y2,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            topk_token_offset=topk_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.stream_id = stream_id

        ctx.save_for_backward(
            z,
            w2,
            b2,
            topk_scores,
            expert_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        stream_id = ctx.stream_id
        is_varlen_K = ctx.is_varlen_K

        (
            z,
            w2,
            b2,
            topk_scores,
            expert_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        ) = ctx.saved_tensors

        dw2 = torch.empty_like(w2)
        db2 = None if b2 is None else torch.empty_like(b2)
        dz = torch.empty_like(z)

        if is_using_quack_gemm():
            assert not torch.compiler.is_compiling()

            s = topk_scores[s_scatter_idx]
            _, y1s, ds = gemm_dgated(
                dout,
                w2.permute(2, 0, 1),
                PreAct=z,
                activation="swiglu",
                dx_out=dz,
                colvec_scale=s,
                colvec_reduce=True,
                cu_seqlens_m=expert_offset,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
            )
            gemm(
                dout.T,
                y1s,
                out=dw2.permute(2, 0, 1),
                cu_seqlens_k=expert_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )

            ds = ds[s_reverse_scatter_idx]
        else:
            ds = torch.empty_like(topk_scores)
            _down_projection_backward(
                dout=dout,
                z=z,
                w2=w2,
                dw2=dw2,
                dz=dz,
                ds=ds,
                b2=b2,
                db2=db2,
                topk_scores=topk_scores,
                expert_offset=expert_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                stream_id=stream_id,
            )

        # TC top-K routing
        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dz, dw2, db2, ds, *[None] * 9


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    is_inference_mode_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(router_logits, router_w.size(0), K)
    expert_frequency, expert_offset = count_cumsum(topk_indices.view(-1), router_w.size(0), do_cumsum=True)

    expert_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, topk_token_offset = TC_topk_router_metadata(
        topk_indices, expert_offset, K
    )

    T = x.size(0)

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_offset,
        T * K,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        topk_token_offset,
        False,  # is_varlen_K
        is_inference_mode_enabled,
    )

    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        topk_scores,
        expert_offset,
        T,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        topk_token_offset,
        False,  # is_varlen_K
    )

    return o, router_logits, expert_frequency


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# We assume sorted_selected_T is already SORTED ascendingly !!!
#   and len(sorted_selected_T) = len(selected_E) = len(router_scores_selected)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Runkai's Remark #25
# Function: moe_general_routing_inputs
# Purpose: Execute MoE forward pass using precomputed general routing assignments (from hybrid TC/EC routing)
#
# Inputs:
# - x: torch.Tensor [T, H] - Input token embeddings (T tokens, H hidden dimensions)
# - router_scores_selected: torch.Tensor [TK] - Selected routing scores after hybrid TC/EC routing
# - sorted_selected_T: torch.Tensor [TK] - Token IDs sorted by token index for coalesced memory access
# - selected_E: torch.Tensor [TK] - Corresponding expert IDs for each token-expert assignment
# - w1: torch.Tensor [2*I, H, E] - Up-projection weights for all experts (2*I for gate and up, I = intermediate size)
# - b1: torch.Tensor [2*I, E] or None - Up-projection biases (optional)
# - w2: torch.Tensor [H, I, E] - Down-projection weights for all experts
# - b2: torch.Tensor [H, E] or None - Down-projection biases (optional)
# - E: int - Total number of experts
# - stream_id: int - CUDA stream ID for kernel synchronization
# - is_inference_mode_enabled: bool - Whether inference mode is enabled (default: False)
#
# Outputs:
# - o: torch.Tensor [T, H] - Final MoE output after weighted aggregation of expert results
# - expert_frequency: torch.Tensor [E] - Number of tokens assigned to each expert

def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores_selected: torch.Tensor,
    sorted_selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    stream_id: int,
    is_inference_mode_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"

    T = x.size(0)
    TK = router_scores_selected.size(0)
    E = w2.size(-1)
    # Runkai's Remark #28
    # Compute routing metadata needed for efficient MoE execution from general routing results
    # This function is defined at line 38-62 of this file
    #
    # Function: general_routing_router_metadata (see lines 38-62)
    # Inputs:
    # - router_scores_selected: [TK] Selected routing scores
    # - sorted_selected_T: [TK] Token IDs sorted by token index
    # - selected_E: [TK] Corresponding expert IDs
    # - T: Number of tokens, E: Number of experts
    #
    # Outputs (6 metadata tensors):
    # - expert_frequency: [E] Count of tokens assigned to each expert (computed via count_cumsum)
    # - expert_offset: [E+1] Cumulative sum for expert workload offsets (0-indexed, starts with 0)
    # - x_gather_idx: [TK] Token indices after sorting by expert ID (for gathering input tokens)
    # - s_scatter_idx: [TK] Indices to sort assignments by expert ID (for scattering to experts)
    # - s_reverse_scatter_idx: [TK] Reverse mapping to unsort after expert computation
    # - topk_token_offset: [T+1] Cumulative sum for each token's assignment count (for aggregation)
    #
    # Example:
    # Input: sorted_selected_T = [0,0,1,1,2,2,3,3], selected_E = [2,3,0,3,0,3,2,3]
    # Step 1: Count expert frequency via count_cumsum(selected_E, E=4)
    #   expert_frequency = [2, 0, 2, 4] (expert 0: 2 tokens, expert 1: 0 tokens, expert 2: 2 tokens, expert 3: 4 tokens)
    #   expert_offset (cumsum) = [0, 2, 2, 4, 8] (positions where each expert's assignments start)
    # Step 2: Sort by expert ID: s_scatter_idx = [2,4,0,6,1,3,5,7] (dest2src mapping)
    #   (positions 2,4 have expert 0; positions 0,6 have expert 2; positions 1,3,5,7 have expert 3)
    #   After sorting: selected_E becomes [0,0,2,2,3,3,3,3] (grouped by expert)
    # Step 3: Reverse mapping: s_reverse_scatter_idx[s_scatter_idx] = [0,1,2,3,4,5,6,7]
    #   s_reverse_scatter_idx = [2,4,0,5,1,6,3,7] (to restore original order after expert computation) 
    #   source2dest mapping
    # Step 4: x_gather_idx = sorted_selected_T[s_scatter_idx] = [1,2,0,3,0,1,2,3] (s_scatter_idx // 2)
    #   (token IDs in expert-sorted order for gathering inputs)
    # Step 5: Count token assignment frequency via count_cumsum or bincount
    #   topk_token_offset = [0, 2, 4, 6, 8] (token 0 has 2 assignments, token 1 has 2, etc.)
    (expert_frequency, expert_offset, x_gather_idx, s_scatter_idx, s_reverse_scatter_idx, topk_token_offset) = (
        general_routing_router_metadata(router_scores_selected, sorted_selected_T, selected_E, T, E)
    )

    # Runkai's Remark #29
    # Execute the up-projection (first FFN layer) using custom autograd function _UpProjection
    # This implements the gated activation: output = silu(gate) * up, where gate and up come from w1
    # The _UpProjection class is defined at lines 94-239 of this file
    #
    # Function: _UpProjection.apply (torch.autograd.Function)
    #
    # Outputs:
    # - y1: [TK, I] Intermediate activations after SwiGLU gated activation
    #       This is the result of silu(gate) * up, where both gate and up are I-dimensional
    # - z: [TK, 2*I] Combined pre-activation values (concatenation of gate and up projections)
    #       Contains both the gate projection and up projection before activation
    #       Saved for backward pass to recompute gradients efficiently
    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_offset,
        TK,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        topk_token_offset,
        True,  # is_varlen_K
        is_inference_mode_enabled,
    )

    # Runkai's Remark #30
    # Execute the down-projection (second FFN layer) with weighted aggregation using custom autograd function
    # This projects intermediate activations from I dimensions back to H (hidden size) and aggregates expert outputs
    # The _DownProjection class is defined at lines 242-408 of this file
    #
    # Function: _DownProjection.apply (torch.autograd.Function)
    # Inputs:
    #
    # Output:
    # - o: [T, H] Final MoE output after weighted aggregation
    #      Each token's output is the weighted sum of its assigned experts' outputs
    #
    # Note: The actual implementation uses efficient CUDA kernels (_down_projection_forward)
    #       with expert-batched GEMMs and custom weighted reduction kernels
    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        router_scores_selected,
        expert_offset,
        T,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        topk_token_offset,
        True,  # is_varlen_K
    )

    # Runkai's Remark #31
    # Return the final MoE output and expert frequency statistics
    return o, expert_frequency
