# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import random
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from tqdm.auto import tqdm

from sonicmoe import MoE
from sonicmoe.functional import count_cumsum, moe_general_routing_inputs
from triton.testing import do_bench


@torch.autocast(device_type="cuda", dtype=torch.float32)
def ref_moe_token_rounding(
    x: torch.Tensor,
    router_scores_selected: torch.Tensor,
    selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E,
):
    T, D = x.shape  # # B, L, # total expert

    ref_o = torch.zeros_like(x, dtype=torch.float32)

    for i in range(E):
        pos = selected_E == i
        T_idx = selected_T[pos]

        if T_idx.numel() > 0:

            w1_out = F.linear(x[T_idx, :], w1[i, :, :].squeeze(), bias=(b1[i] if b1 is not None else None))
            w1_out = F.silu(w1_out[:, ::2]) * w1_out[:, 1::2]

            w2_out = F.linear(w1_out, w2[i, :, :].squeeze(), bias=(b2[i] if b2 is not None else None))

            ref_o[T_idx, :] += w2_out * router_scores_selected[pos, None]

    return ref_o.view(T, D)


def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of SonicMoE (arbitrary routing inputs).")

    parser.add_argument(
        "--thiekq",
        type=parse_comma_separated_ints,
        default=(16384, 4096, 1024, 256, 8, 128),
        help="T, H, I, E, K, tileM dimensions (comma-separated)",
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--routing",
        type=str,
        choices=["top_k", "nr", "up", "down"],
        default="top_k",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if len(args.thiekq) != 6:
        parser.error("--thiekq must contain exactly 6 values")

    return args


def our_e2e_fwd_bwd_call(x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id, dout):
    o, _ = moe_general_routing_inputs(
        x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id, False
    )
    torch.autograd.grad(o, [x, router_scores_selected, w1, w2], dout, retain_graph=True)
    router_scores_selected.grad = x.grad = w1.grad = w2.grad = None


def our_fwd_call(x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id):
    return moe_general_routing_inputs(
        x, router_scores_selected, sorted_selected_T, selected_E, w1, b1, w2, b2, E, stream_id, False
    )


# Runkai's Remark #10
# Hybrid Token-Choice/Expert-Choice routing with tile-based rounding for hardware efficiency
# Function: forward_token_choice_rounding
# Purpose: Implements a routing strategy that combines Token-Choice (TC) and Expert-Choice (EC)
#          routing, rounding expert workloads to multiples of Mtile for efficient GPU computation
# Inputs:
#   - x: Input token embeddings, shape (T, D) where T=num_tokens, D=hidden_dim
#   - router_w: Router weights, shape (E, D) for computing expert scores
#   - E: Number of experts (e.g., 256)
#   - K: Top-K experts per token in first stage (e.g., 8)
#   - Mtile: Tile size for rounding (e.g., 128) - workloads rounded to multiples of this
#   - routing: Rounding strategy - "up" (ceil), "down" (floor), or "nr" (nearest)
# Outputs:
#   - router_scores_selected: Routing scores for selected token-expert pairs, shape (num_selected,)
#   - selected_T: Token indices (sorted), shape (num_selected,)
#   - selected_E: Expert indices, shape (num_selected,)
# Example: T=1000 tokens, K=8, Mtile=128:
#   - Stage 1 (TC): Each token picks top-8 experts → 8000 assignments
#   - Stage 2 (EC): Each expert's workload rounded (e.g., expert 0: 145→128 or 256 tokens)
#   - Output: Variable number of assignments based on rounding strategy
def forward_token_choice_rounding(
    x: torch.Tensor, router_w: torch.Tensor, E, K, Mtile, routing
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Runkai's Remark #11
    # Extract input dimensions and override Mtile parameter
    # T: Number of tokens, D: Hidden dimension (should equal router_w.shape[1])
    # Note: Mtile parameter is overridden to 128 (hardcoded), ignoring the passed value
    # CONSISTENT EXAMPLE: We'll trace through T=4 tokens, E=4 experts, K=2, Mtile=128, routing="down"
    T, D = x.shape  # # B, L, # total expert
    Mtile = 128

    device = x.device
    dtype = x.dtype

    # Runkai's Remark #12
    # Compute router logits and scores for all experts
    # router_logits: Linear projection x @ router_w.T, shape (T, E)
    #   Each row contains unnormalized scores for all E experts for one token
    # router_scores: Softmax-normalized scores, shape (T, E)
    #   Each row sums to 1.0, representing probability distribution over experts
    # EXAMPLE: After softmax, router_scores (T=4, E=4):
    #   [[0.10, 0.05, 0.30, 0.55],   # token 0: expert 3 is highest
    #    [0.60, 0.15, 0.05, 0.20],   # token 1: expert 0 is highest
    #    [0.25, 0.30, 0.25, 0.20],   # token 2: expert 1 is highest
    #    [0.15, 0.10, 0.50, 0.25]]   # token 3: expert 2 is highest
    router_logits = F.linear(x, router_w)
    router_scores = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(dtype)

    # Runkai's Remark #13
    # STAGE 1: Token-Choice (TC) routing - each token selects its top-K experts
    # topk_values: Top-K scores for each token, shape (T, K)
    # topk_indices: Indices of top-K experts for each token, shape (T, K)
    # EXAMPLE: Each token picks top-K=2 experts:
    #   topk_values (T=4, K=2):
    #   [[0.55, 0.30],   # token 0 picks experts 3, 2
    #    [0.60, 0.20],   # token 1 picks experts 0, 3
    #    [0.30, 0.25],   # token 2 picks experts 1, 0 (or 2)
    #    [0.50, 0.25]]   # token 3 picks experts 2, 3
    #   topk_indices (T=4, K=2):
    #   [[3, 2],   # token 0 → experts 3, 2
    #    [0, 3],   # token 1 → experts 0, 3
    #    [1, 0],   # token 2 → experts 1, 0
    #    [2, 3]]   # token 3 → experts 2, 3
    # first sorting, similar to TC
    topk_values, topk_indices = router_scores.topk(K, dim=-1)

    # Runkai's Remark #14
    # Count how many tokens selected each expert and compute cumulative sum
    # Function: count_cumsum (defined in sonicmoe/count_cumsum/__init__.py:16-29)
    # Kernel origin: sonicmoe/count_cumsum/kernel.cu (CUDA kernel for efficient counting)
    # Input: topk_indices.view(-1) flattens (T, K) → (T*K,) expert IDs
    # Output: expert_freq, shape (E,) - histogram showing how many tokens chose each expert
    # EXAMPLE: Flattened topk_indices = [3,2, 0,3, 1,0, 2,3]
    #   Expert 0 appears 2 times (from tokens 1, 2)
    #   Expert 1 appears 1 time (from token 2)
    #   Expert 2 appears 2 times (from tokens 0, 3)
    #   Expert 3 appears 3 times (from tokens 0, 1, 3)
    #   expert_freq = [2, 1, 2, 3] (counts per expert, shape (E=4,))
    # Note: Uses custom CUDA kernel with shared memory atomics for fast counting
    expert_freq = count_cumsum(topk_indices.view(-1), E, do_cumsum=True)[0]
    # Runkai's Remark #15
    # Compute rounded expert frequencies for three strategies
    # expert_freq_rounded_up: Ceil to next Mtile multiple, shape (E,)
    # expert_freq_rounded_down: Floor to previous Mtile multiple, shape (E,)
    # Purpose: Aligning workloads to tile boundaries improves GPU kernel efficiency
    #   Most GEMM kernels are optimized for multiples of 128/256
    # EXAMPLE: With expert_freq = [2, 1, 2, 3] and Mtile=128:
    #   expert_freq_rounded_up   = [128, 128, 128, 128] (ceil to next multiple)
    #   expert_freq_rounded_down = [0, 0, 0, 0] (floor to previous multiple)
    # Note: For this small example, all values round to 0 or 128
    expert_freq_rounded_up = (torch.ceil(expert_freq / Mtile) * Mtile).type(torch.int32)
    expert_freq_rounded_down = expert_freq // Mtile * Mtile

    # Runkai's Remark #16
    # Re-normalize topk_values so they sum to 1 per token (in case of numerical drift)
    # This ensures routing scores remain valid probabilities after top-K selection
    # Shape: (T, K) - each row sums to 1.0
    # EXAMPLE: Before normalization: topk_values = [[0.55, 0.30], [0.60, 0.20], [0.30, 0.25], [0.50, 0.25]]
    #   After normalization (each row sums to 1):
    #   [[0.647, 0.353],   # 0.55/(0.55+0.30), 0.30/(0.55+0.30)
    #    [0.75, 0.25],     # 0.60/0.80, 0.20/0.80
    #    [0.545, 0.455],   # 0.30/0.55, 0.25/0.55
    #    [0.667, 0.333]]   # 0.50/0.75, 0.25/0.75
    topk_values /= topk_values.sum(dim=-1, keepdim=True)

    # Runkai's Remark #17
    # Update router_scores with re-normalized top-K values
    # scatter_(-1, topk_indices, topk_values) writes topk_values into router_scores
    #   at positions specified by topk_indices along dimension -1 (expert dimension)
    # Result: router_scores now has normalized values for top-K experts, others unchanged
    # EXAMPLE: After scatter, router_scores becomes:
    #   [[0.10, 0.05, 0.353, 0.647],   # token 0: positions 2,3 updated
    #    [0.75, 0.15, 0.05, 0.25],     # token 1: positions 0,3 updated
    #    [0.455, 0.545, 0.25, 0.20],   # token 2: positions 0,1 updated
    #    [0.15, 0.10, 0.667, 0.333]]   # token 3: positions 2,3 updated
    # Non-topK positions keep original softmax values, topK positions get renormalized values
    router_scores.scatter_(-1, topk_indices, topk_values)

    # Runkai's Remark #18
    # Prepare combined score matrix for Expert-Choice (EC) stage
    # Strategy: Create a score matrix where EC candidates have lower scores than TC selections
    # router_TC_EC_combined_val: Clone of router_scores, shape (T, E)
    # Step 1: Subtract 1 from all scores → all values become negative
    # Step 2: Restore original topk_values for TC selections via scatter_
    # Result: TC selections have positive scores (0-1), EC candidates have negative scores (-1 to 0)
    # Purpose: When sorting, TC selections (positive) appear before EC candidates (negative)
    #   This implements the hybrid routing: prioritize tokens' choices, then fill with EC
    # EXAMPLE: Starting from router_scores (Remark #17):
    #   After Step 1 (subtract 1), router_TC_EC_combined_val:
    #   [[-0.90, -0.95, -0.647, -0.353],   # all negative
    #    [-0.25, -0.85, -0.95, -0.75],
    #    [-0.545, -0.455, -0.75, -0.80],
    #    [-0.85, -0.90, -0.333, -0.667]]
    #
    #   After Step 2 (scatter topk_values), router_TC_EC_combined_val:
    #   [[-0.90, -0.95, 0.353, 0.647],    # token 0: TC experts 2,3 are positive
    #    [0.75, -0.85, -0.95, 0.25],      # token 1: TC experts 0,3 are positive
    #    [0.455, 0.545, -0.75, -0.80],    # token 2: TC experts 0,1 are positive
    #    [-0.85, -0.90, 0.667, 0.333]]    # token 3: TC experts 2,3 are positive
    # Result: Each row has K positive values (TC) and (E-K) negative values (EC candidates)
    router_TC_EC_combined_val = router_scores.detach().clone()
    router_TC_EC_combined_val -= 1  # make sure EC's score is lower than TC & EC keeps the score order
    router_TC_EC_combined_val.scatter_(1, topk_indices, topk_values)  # mask out original TC score

    # Runkai's Remark #19
    # STAGE 2: Expert-Choice (EC) sorting - experts select their top tokens
    # argsort(dim=0, descending=True): Sort along token dimension (dim=0) for each expert
    # Result: topk_indices now contains token rankings per expert, shape (T, E)
    # For expert e, topk_indices[:, e] lists tokens in descending score order
    # EXAMPLE: Sorting each column (expert) by score descending:
    #   Expert 0 column: [0.75, 0.455, -0.85, -0.90] → sorted order: [1, 2, 3, 0]
    #   Expert 1 column: [-0.85, 0.545, -0.90, -0.95] → sorted order: [2, 1, 0, 3]
    #   Expert 2 column: [0.353, -0.95, -0.75, 0.667] → sorted order: [3, 0, 2, 1]
    #   Expert 3 column: [0.647, 0.25, -0.80, 0.333] → sorted order: [0, 3, 1, 2]
    #   topk_indices (T=4, E=4) after argsort:
    #   [[1, 2, 3, 0],   # row 0: each column shows the top token for that expert
    #    [2, 1, 0, 3],   # row 1: second-best token for each expert
    #    [3, 0, 2, 1],   # row 2: third-best token for each expert
    #    [0, 3, 1, 2]]   # row 3: fourth-best token for each expert
    # Note: TC selections (positive scores) appear first in each expert's ranking
    # second sorting, similar to EC
    topk_indices = router_TC_EC_combined_val.argsort(dim=0, descending=True).int()  # type: ignore

    # Runkai's Remark #20
    # Select rounding strategy based on 'routing' parameter
    # Three strategies for handling fractional workloads:
    # - "down": Floor to Mtile multiple (conservative, may underutilize)
    # - "up": Ceil to Mtile multiple (aggressive, ensures capacity)
    # - "nr": Nearest Mtile multiple (balanced approach)
    # EXAMPLE: With routing="nr", Mtile=2, expert_freq=[2, 1, 2, 3]:
    #   expert_freq / Mtile = [2/2, 1/2, 2/2, 3/2] = [1.0, 0.5, 1.0, 1.5]
    #   torch.round() = [1, 0 (rounds 0.5 to nearest even), 1, 2]
    #   Multiply by Mtile: [1*2, 0*2, 1*2, 2*2] = [2, 0, 2, 4]
    #   expert_freq_rounded = [2, 0, 2, 4]
    # Result: Expert 0 gets 2 tokens, Expert 1 gets 0, Expert 2 gets 2, Expert 3 gets 4
    # Note: In production with Mtile=128, typical expert_freq values would be hundreds/thousands
    if routing == "down":
        expert_freq_rounded = expert_freq_rounded_down

    elif routing == "up":
        expert_freq_rounded = expert_freq_rounded_up

    elif routing == "nr":
        expert_freq_rounded = torch.round(expert_freq / Mtile).type(torch.int32) * Mtile

    else:
        raise NotImplementedError()

    # Runkai's Remark #21
    # Create selection mask based on rounded expert frequencies
    # expert_freq_mask: Boolean mask, shape (T, E)
    # For expert e, selects top expert_freq_rounded[e] tokens from sorted ranking
    # Construction: torch.arange(T)[:, None] creates column [0,1,2,...,T-1], shape (T, 1)
    #   expand(-1, E) broadcasts to (T, E), expert_freq_rounded[None, :] creates row, shape (1, E)
    #   Comparison: position < capacity for each expert
    # EXAMPLE: With routing="nr", Mtile=2, expert_freq_rounded=[2, 0, 2, 4]:
    #   Position matrix (T=4):       Capacity (broadcast):
    #   [[0, 0, 0, 0],               [[2, 0, 2, 4],
    #    [1, 1, 1, 1],                [2, 0, 2, 4],
    #    [2, 2, 2, 2],       <        [2, 0, 2, 4],
    #    [3, 3, 3, 3]]                [2, 0, 2, 4]]
    #
    #   expert_freq_mask (T=4, E=4):
    #   [[True,  False, True,  True ],   # row 0: select for experts 0, 2, 3
    #    [True,  False, True,  True ],   # row 1: select for experts 0, 2, 3
    #    [False, False, False, True ],   # row 2: select only for expert 3
    #    [False, False, False, True ]]   # row 3: select only for expert 3
    # Result: Expert 0 gets top 2 tokens, Expert 1 gets 0, Expert 2 gets top 2, Expert 3 gets all 4
    expert_freq_mask = torch.arange(T, device=device, dtype=torch.int32)[:, None].expand(-1, E) < expert_freq_rounded[None, :]  # type: ignore

    # Runkai's Remark #22
    # Extract selected token and expert indices using the mask
    # selected_T: Token indices for selected assignments, shape (num_selected,)
    #   topk_indices[expert_freq_mask] extracts token IDs where mask is True
    # selected_E: Expert indices for selected assignments, shape (num_selected,)
    #   torch.arange(E) expanded to (T, E), then masked to get expert IDs
    # EXAMPLE: With routing="nr", Mtile=2, from Remark #19's topk_indices:
    #   topk_indices = [[1, 2, 3, 0],   # Expert 0's top is token 1, Expert 3's top is token 0, etc.
    #                   [2, 1, 0, 3],
    #                   [3, 0, 2, 1],
    #                   [0, 3, 1, 2]]
    #   expert_freq_mask has 8 True values (2+0+2+4=8 total selections):
    #     Expert 0 column: True at rows [0,1] → extracts tokens [1, 2]
    #     Expert 1 column: all False → no selections
    #     Expert 2 column: True at rows [0,1] → extracts tokens [3, 0]
    #     Expert 3 column: True at rows [0,1,2,3] → extracts tokens [0, 3, 1, 2]
    #   selected_T = [1, 2, 3, 0, 0, 3, 1, 2] (8 values, order: expert0, expert2, expert3)
    #   selected_E = [0, 0, 2, 2, 3, 3, 3, 3] (corresponding expert IDs)
    # Note: selected_T is NOT sorted by token ID yet (sorted by expert)
    selected_T = topk_indices[expert_freq_mask]
    selected_E = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(T, -1)[expert_freq_mask]  # type: ignore

    # Runkai's Remark #23
    # Sort all assignments by token index for efficient reduction operations
    # Requirement: Downstream MoE kernels expect assignments sorted by token ID
    # selected_T_order: Indices that would sort selected_T, shape (num_selected,)
    # After reordering: selected_T is monotonically increasing (e.g., [0, 1, 2, 3, ...])
    #   selected_E reordered correspondingly to maintain pairing
    # Purpose: Sorted token order enables efficient memory coalescing in CUDA kernels
    # EXAMPLE: With routing="nr", Mtile=2, before sorting (from Remark #22):
    #   selected_T = [1, 2, 3, 0, 0, 3, 1, 2]
    #   selected_E = [0, 0, 2, 2, 3, 3, 3, 3]
    #
    #   selected_T.argsort() finds indices to sort: [3, 4, 0, 6, 1, 7, 2, 5]
    #     (token 0 is at positions 3,4; token 1 is at positions 0,6; etc.)
    #
    #   After reordering:
    #   selected_T = [0, 0, 1, 1, 2, 2, 3, 3]   # sorted by token ID
    #   selected_E = [2, 3, 0, 3, 0, 3, 2, 3]   # reordered to maintain token-expert pairing
    # Result: Sorted for efficient sequential access in CUDA kernels
    # implicit assumption: selected_T should be sorted in my reduction code
    selected_T_order = selected_T.argsort().int()
    selected_T = selected_T[selected_T_order]
    selected_E = selected_E[selected_T_order]

    # Runkai's Remark #24
    # Extract final routing scores and return all routing information
    # router_scores[selected_T, selected_E]: Indexing to get scores for each assignment
    # .contiguous(): Ensure memory layout is contiguous for efficient kernel access
    # Returns:
    #   1. Routing scores for selected pairs, shape (num_selected,)
    #   2. Sorted token indices, shape (num_selected,)
    #   3. Corresponding expert indices, shape (num_selected,)
    # EXAMPLE: With routing="nr", Mtile=2, after sorting (from Remark #23):
    #   selected_T = [0, 0, 1, 1, 2, 2, 3, 3]
    #   selected_E = [2, 3, 0, 3, 0, 3, 2, 3]
    #   router_scores (from Remark #17):
    #   [[0.10, 0.05, 0.353, 0.647],
    #    [0.75, 0.15, 0.05, 0.25],
    #    [0.455, 0.545, 0.25, 0.20],
    #    [0.15, 0.10, 0.667, 0.333]]
    #
    #   Extracted scores (router_scores[selected_T, selected_E]):
    #   [router_scores[0,2], router_scores[0,3], router_scores[1,0], router_scores[1,3],
    #    router_scores[2,0], router_scores[2,3], router_scores[3,2], router_scores[3,3]]
    #   = [0.353, 0.647, 0.75, 0.25, 0.455, 0.20, 0.667, 0.333]
    #
    #   Returns: (scores=[0.353, 0.647, 0.75, 0.25, 0.455, 0.20, 0.667, 0.333],
    #             selected_T=[0, 0, 1, 1, 2, 2, 3, 3],
    #             selected_E=[2, 3, 0, 3, 0, 3, 2, 3])
    # Total: 8 token-expert assignments (2+0+2+4 from expert_freq_rounded)
    return router_scores[selected_T, selected_E].contiguous(), selected_T, selected_E


def forward_topk(x: torch.Tensor, router_w: torch.Tensor, E, K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = x.shape[0]

    router_logits = F.linear(x, router_w)

    top_logits, topk_indices = router_logits.topk(K, dim=1)
    router_scores = F.softmax(top_logits, dim=-1, dtype=torch.float32)

    # first sorting, similar to TC
    return (
        router_scores.view(-1),
        torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(K),
        topk_indices.view(-1).int(),
    )


def run(
    thiekq: Tuple[int, int, int, int, int, int],
    dtype: Type[cutlass.Numeric],
    rep: int,
    routing: str,
    skip_test: Type[bool],
    add_bias: Type[bool],
    **kwargs,
):

    cutlass_dtype = dtype
    torch_dtype = {cutlass.BFloat16: torch.bfloat16, cutlass.Float16: torch.float16}[dtype]

    # Runkai's Remark #1
    # Unpack the 6-element configuration tuple 'thiekq' into individual variables:
    # - T (Token count): Total number of input tokens to process (default: 16384)
    # - H (Hidden size): Dimension of token embeddings/hidden features (default: 4096)
    #
    # - I (Intermediate size): Dimension of FFN intermediate layer (default: 1024)
    # - E (Experts count): Total number of expert networks in the MoE (default: 256)
    # - K (Top-K): Number of experts activated per token (default: 8)
    # - Mtile (Tile size): Block size for rounding token allocations (default: 128)
    T, H, I, E, K, Mtile = thiekq
    TK = T * K
    print(f"T {T}, I {I}, H {H}, E {E}, K {K} | Routing {routing}")

    random.seed(1100)
    torch.manual_seed(1100)
    torch.cuda.manual_seed_all(1100)

    avg_time = torch.zeros(3)
    avg_tflops = torch.zeros(3)

    total_processed_tokens = 0
    total_hardware_tokens = 0

    # Runkai's Remark #3
    # Initialize the Mixture-of-Experts (MoE) model with GLU-based feed-forward networks
    # Function: MoE.__init__ (defined in sonicmoe/moe.py:139-176)
    # Inputs:
    #   - num_experts (E): Total number of expert networks (default: 256)
    #   - num_experts_per_tok (K): How many experts each token is routed to (default: 8)
    #   - hidden_size (H): Input/output dimension of each expert (default: 4096)
    #   - intermediate_size (I): Hidden dimension inside each expert FFN (default: 1024)
    #   - is_glu (bool): Use Gated Linear Unit activation (True = SwiGLU variant)
    #   - add_bias (bool): Whether to add bias terms in linear layers
    #   - std (float): Standard deviation for weight initialization (0.02)
    # Output: MoE module containing:
    #   - self.router: Linear layer (H → E) that scores each expert for each token
    #   - self.c_fc: Experts layer for up-projection (H → 2*I for GLU, or H → I otherwise)
    #   - self.c_proj: Experts layer for down-projection (I → H)
    #   - self.stream_id: CUDA stream ID for kernel execution
    # Example: With E=8 experts, K=2 top-k, each token's features (H-dim) get processed by
    #          its 2 highest-scoring experts, then weighted-averaged for final output
    # Note: When is_glu=True, c_fc outputs 2*I dimensions which are split and combined
    #       using SwiGLU: output = silu(gate) * up, where gate and up are I-dimensional
    moe = (
        MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            is_glu=True,
            add_bias=add_bias,
            std=0.02,
        )
        .to(dtype=torch_dtype)
        .cuda()
    )
    # Runkai's Remark #4
    # Compile the MoE module using PyTorch's JIT compiler for optimization
    moe = torch.compile(moe)

    x = 0.2 * torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    dout = 0.2 * torch.randn_like(x, requires_grad=True)

    # Runkai's Remark #5
    # Extract weight tensors from the MoE module for direct access in benchmarking
    # This line unpacks three weight matrices from different components of the MoE model:
    #
    # - w1 (from moe.c_fc.weight): Up-projection weights for all experts
    #   Origin: Created in MoE.__init__ -> Experts.__init__ at sonicmoe/moe.py:160-166
    #   Shape: (E, 2*I, H) when is_glu=True, or (E, I, H) otherwise
    #   Purpose: First linear layer in each expert's FFN, projects from hidden_size H to intermediate_size
    #   Example: With E=256 experts, H=4096, I=1024, GLU=True: shape is (256, 2048, 4096)
    #
    # - w2 (from moe.c_proj.weight): Down-projection weights for all experts
    #   Origin: Created in MoE.__init__ -> Experts.__init__ at sonicmoe/moe.py:168-174
    #   Shape: (E, H, I) for all experts
    #   Purpose: Second linear layer in each expert's FFN, projects from intermediate_size I back to hidden_size H
    #   Example: With E=256 experts, H=4096, I=1024: shape is (256, 4096, 1024)
    #
    # - router_w (from moe.router.weight): Router scoring weights
    #   Origin: Created in MoE.__init__ at sonicmoe/moe.py:158
    #   Shape: (E, H) - maps each token's hidden state to expert scores
    #   Purpose: Linear layer that computes routing logits for all experts given a token embedding
    #   Example: With E=256 experts, H=4096: shape is (256, 4096)
    #   Note: Router has no bias term (bias=False in nn.Linear initialization)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias
    # Runkai's Remark #7
    # Redundant assignment: router_w already extracted above at line 263
    # This line re-assigns the same value, likely left from code refactoring
    # router_w is the routing weight matrix used to compute expert selection scores
    router_w = moe.router.weight

    # Runkai's Remark #8
    # Extract CUDA stream ID for kernel execution synchronization
    # Origin: Created in MoE.__init__ at sonicmoe/moe.py:176
    # Type: Integer representing a CUDA stream handle
    # Purpose: Used to synchronize custom CUDA kernels with PyTorch's computation graph
    #          Ensures MoE kernels execute on the correct stream for proper async execution
    # Used in: moe_general_routing_inputs function calls (e.g., line 259, 349, 366)
    stream_id = moe.stream_id

    if add_bias:
        torch.nn.init.normal_(b1, 0, 0.01)
        torch.nn.init.normal_(b2, 0, 0.01)

    # # Ref check
    if not skip_test:
        x_clone = x.detach().clone()
        x_clone.requires_grad_(True)
        dout_clone = dout.detach().clone()
        dout_clone.requires_grad_(True)

        if routing == "top_k":
            router_scores_selected, selected_T, selected_E = forward_topk(x, router_w, E, K)
            router_scores_selected_clone, _, _ = forward_topk(x_clone, router_w, E, K)
        else:
            router_scores_selected, selected_T, selected_E = forward_token_choice_rounding(
                x, router_w, E, K, Mtile, routing
            )
            router_scores_selected_clone, _, _ = forward_token_choice_rounding(x_clone, router_w, E, K, Mtile, routing)

        o, expert_frequency = moe_general_routing_inputs(
            x,
            router_scores_selected,
            selected_T,
            selected_E,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            E,
            stream_id,
            False,
        )
        if add_bias:
            dx, dw1, db1, dw2, db2, drouter_w = torch.autograd.grad(
                o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
            )
        else:
            dx, dw1, dw2, drouter_w = torch.autograd.grad(o, [x, w1, w2, router_w], grad_outputs=dout)

        ref_o = ref_moe_token_rounding(
            x_clone,
            router_scores_selected_clone,
            selected_T,
            selected_E,
            w1,
            b1,
            w2,
            b2,
            E,
        )
        ref_expert_frequency = selected_E.view(-1).bincount(minlength=E)

        torch.testing.assert_close(expert_frequency.int(), ref_expert_frequency.int())

        o_diff = (o.float() - ref_o).abs()

        print(f"max ref o val {ref_o.abs().max():.6f}")
        print(f"mean ref o val {ref_o.abs().mean():.6f}")
        print(f"max abs diff on o {o_diff.max():.6f}")
        print(f"mean rel diff on o {(o_diff / (ref_o.abs() + 1e-6)).mean():.6f}" + "\n")

        if add_bias:
            ref_dx, ref_dw1, ref_db1, ref_dw2, ref_db2, ref_drouter_w = torch.autograd.grad(
                ref_o, [x_clone, w1, b1, w2, b2, router_w], grad_outputs=dout_clone
            )
            test_triple_list = [
                ("dx", dx, ref_dx),
                ("dw2", dw2, ref_dw2),
                ("db2", db2, ref_db2),
                ("dw1", dw1, ref_dw1),
                ("db1", db1, ref_db1),
                ("drouter_w", drouter_w, ref_drouter_w),
            ]
        else:
            ref_dx, ref_dw1, ref_dw2, ref_drouter_w = torch.autograd.grad(
                ref_o, [x_clone, w1, w2, router_w], grad_outputs=dout_clone
            )
            test_triple_list = [
                ("dx", dx, ref_dx),
                ("dw2", dw2, ref_dw2),
                ("dw1", dw1, ref_dw1),
                ("drouter_w", drouter_w, ref_drouter_w),
            ]

        for n, our, ref in test_triple_list:
            print(f"max abs ref value {n} {ref.abs().max():.6f}")
            print(f"mean abs ref value {n} {ref.abs().mean():.6f}")
            print(f"max abs diff on {n} {(our - ref).abs().max():.6f}")
            print(f"mean rel diff on {n} {((our - ref).abs() / (ref.abs() + 1e-6)).mean():.6f}" + "\n")

    TRIALS = 50
    for _ in tqdm(range(TRIALS)):
        torch.nn.init.normal_(w1, 0.0, 0.02)
        torch.nn.init.normal_(w2, 0.0, 0.02)

        if add_bias:
            torch.nn.init.normal_(b1, 0, 0.01)
            torch.nn.init.normal_(b2, 0, 0.01)

        x = torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
        dout = 0.2 * torch.randn_like(x, requires_grad=True)

        # Runkai's Remark #9
        # Select routing strategy and compute token-to-expert assignments
        # This conditional chooses between two routing methods based on the 'routing' parameter:
        #
        # Origin of 'routing' variable: Command-line argument --routing (line 79-83)
        #   Possible values: "top_k", "nr" (nearest round), "up" (round up), "down" (round down)
        #
        # Branch 1: routing == "top_k" (Standard Top-K Routing)
        # Function: forward_topk (defined at line 171-184)
        # Inputs:
        #   - x: Input tokens, shape (T, H) - all tokens with their hidden representations
        #   - router_w.detach(): Router weights (E, H), detached to prevent gradient flow during routing
        #   - E: Number of experts (e.g., 256)
        #   - K: Experts per token (e.g., 8)
        # Outputs (all three are 1D tensors of length T*K):
        #   - router_scores_selected: Normalized scores for selected experts, shape (T*K,)
        #   - selected_T: Token indices for each assignment, shape (T*K,)
        #   - selected_E: Expert indices for each assignment, shape (T*K,)
        # Example: With T=4 tokens, K=2, you get 8 assignments total
        #   router_scores_selected = [0.6, 0.4, 0.7, 0.3, ...] (probabilities sum to 1 per token)
        #   selected_T = [0, 0, 1, 1, 2, 2, 3, 3] (each token appears K times)
        #   selected_E = [5, 12, 3, 7, ...] (which experts were chosen)
        if routing == "top_k":
            router_scores_selected, selected_T, selected_E = forward_topk(x, router_w.detach(), E, K)
        # Branch 2: routing in {"nr", "up", "down"} (Token Choice with Rounding)
        # Function: forward_token_choice_rounding (defined at line 116-168)
        # Inputs:
        #   - x: Input tokens, shape (T, H)
        #   - router_w.detach(): Router weights (E, H), detached from gradient graph
        #   - E: Number of experts
        #   - K: Initial top-k for each token (used in first sorting stage)
        #   - Mtile: Tile size for rounding (default: 128)
        #   - routing: Rounding strategy ("nr"=nearest, "up"=ceil, "down"=floor)
        # Outputs (1D tensors, variable length depending on rounding):
        #   - router_scores_selected: Selected routing scores, shape (num_selected,)
        #   - selected_T: Token indices (sorted), shape (num_selected,)
        #   - selected_E: Expert indices, shape (num_selected,)
        # Purpose: Rounds expert workloads to multiples of Mtile for hardware efficiency
        #   This can change the total number of assignments (not always T*K)
        # Example: If expert 0 gets 135 tokens and Mtile=128:
        #   "up" → 256 tokens, "down" → 128 tokens, "nr" → 128 tokens (nearest)
        # Note: Uses hybrid Token-Choice/Expert-Choice routing to balance load
        else:
            router_scores_selected, selected_T, selected_E = forward_token_choice_rounding(
                x, router_w.detach(), E, K, Mtile, routing
            )

        our_e2e_fwd_bwd_call(
            x,
            router_scores_selected,
            selected_T,
            selected_E,
            w1.permute(1, 2, 0),
            b1,
            w2.permute(1, 2, 0),
            b2,
            E,
            stream_id,
            dout,
        )

        TK = router_scores_selected.shape[0]

        forward_time = do_bench(
            lambda: our_fwd_call(
                x,
                router_scores_selected,
                selected_T,
                selected_E,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                E,
                stream_id,
            ),
            warmup=10,
            rep=rep,
        )
        flops = 6 * TK * I * H
        tflops = flops / (forward_time / 1e3) / 1e12

        avg_time[0] += forward_time
        avg_tflops[0] += tflops

        flops = 18 * TK * I * H
        e2e_time = do_bench(
            lambda: our_e2e_fwd_bwd_call(
                x,
                router_scores_selected,
                selected_T,
                selected_E,
                w1.permute(1, 2, 0),
                b1,
                w2.permute(1, 2, 0),
                b2,
                E,
                stream_id,
                dout,
            ),
            warmup=10,
            rep=rep,
            grad_to_none=[x, w1, w2, router_w, dout],
        )
        tflops = flops / (e2e_time / 1e3) / 1e12

        avg_time[1] += e2e_time
        avg_tflops[1] += tflops

        flops = 12 * TK * I * H
        bwd_time = e2e_time - forward_time
        tflops = flops / (bwd_time / 1e3) / 1e12

        avg_time[2] += bwd_time
        avg_tflops[2] += tflops

        expert_frequency = torch.bincount(selected_E).int()
        total_processed_tokens += expert_frequency.sum().item()
        total_hardware_tokens += (torch.ceil(expert_frequency / Mtile).to(torch.int32) * Mtile).sum().sum().item()

    avg_time /= TRIALS
    avg_tflops /= TRIALS

    print0(f"[bold green][/bold green] {routing}, Fwd Average time: {avg_time[0]:.3f} ms, TFLOPS: {avg_tflops[0]:.1f}")
    print0(f"[bold green][/bold green] {routing}, E2E Average time: {avg_time[1]:.3f} ms, TFLOPS: {avg_tflops[1]:.1f}")
    print0(f"[bold green][/bold green] {routing}, Bwd Average time: {avg_time[2]:.3f} ms, TFLOPS: {avg_tflops[2]:.1f}")
    print0(
        f"[bold green][/bold green] {routing}, processed tokens, hardware tokens {total_processed_tokens:.1f}, {total_hardware_tokens:.1f}. wasted ratio {(total_hardware_tokens-total_processed_tokens)/total_processed_tokens:.3f}"
    )


if __name__ == "__main__":
    args = parse_arguments()
    run(args.thiekq, args.dtype, args.rep, args.routing, args.skip_test, args.add_bias)
    print("PASS")
