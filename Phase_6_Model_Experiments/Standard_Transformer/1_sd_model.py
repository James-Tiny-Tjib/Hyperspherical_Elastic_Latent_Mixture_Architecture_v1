# %%writefile model.py

##################################################
# Defines the HELM V1 architecture
# Removes all NGPT components and replaces with 
##################################################

import os
import json
import torch
import numpy as np
from safetensors.torch import load_file
import math
from math import sqrt
import random
import torch.nn.functional as F
import torch.nn as nn
try:
    from torch_xla.utils.checkpoint import checkpoint as _xla_checkpoint
except Exception:
    _xla_checkpoint = None
from transformers import AutoTokenizer
from transformers import PretrainedConfig, PreTrainedModel



# modified justnorm() function
# better than F.normalize(), max() causes micro walls during gradient descent
# better than nGPT's version, prevents division by 0 error
def justnorm(x, dim = -1, eps = 1e-12):
    res = x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    return res

# Cast the input to the correct input layer dtype
def cast_linear(x, layer):
    w = layer.weight.to(x.dtype)
    b = None if layer.bias is None else layer.bias.to(x.dtype)
    return F.linear(x,w,b)


# Hugging Face Config Class (for future deployment)
class HELMConfig(PretrainedConfig):

    model_type = "helm"

    def __init__(

        self,

        # General Model Hyperparameters
        hidden_size = 1024,
        sqrt_hidden_size = 32,
        max_position_embeddings = 4096,
        initializer_range = 0.03125,
        num_hidden_layers = 12,
        num_attention_heads = 16,
        rope_theta = 160000,
        intermediate_size = 2816,
        norm_eps = 1e-12,
        hidden_act = "swiglu",
        swiglu_s_init = 1.0,
        base_lr = 3e-4,
        min_lr = 3e-5,
        weight_decay = 0.1,
        bias = False,
        use_ckpt = False,

        # Tokenization and Data Collator Hyperparameters
        tokenizer_path = "answerdotai/ModernBERT-base",
        vocab_size = 50368,
        bos_token_id = 50281,
        eos_token_id = 50282,
        pad_token_id = 50283,
        mask_token_id = 50284,
        unk_token_id = 50285,
        mlm_probability = 0.3,
        mlm_use_span_masking = True,
        mlm_span_length = 3,

        # Router Hyperparameters
        num_router_latents = 4,
        num_permanent_heads = 2,
        selection_threshold = 0.5,
        router_init_scale = 1.0,
        use_sigmoid_scaling = False,
        jitter_noise = 0.01,
        router_grad_clip = 0.05,
        dense_warmup_steps = 0.03,


        # Router Sparsity Hyperparameters
        sparsity_lambda = 0.01,
        sparsity_warm_up_steps = 0.05,
        head_target_min = 4,
        head_target_center = 8,
        head_target_max = 16,
        easiness_cdf_breakpoints = None,
        sparsity_slack_lo = 1.0,
        sparsity_slack_hi = 2.0,

        # Router Auxiliary Hyperparameters:
        aux_coeff_start = 0.02,
        aux_coeff_floor = 0.002,
        aux_anneal_start = 0.08,
        aux_anneal_steps = 0.08,

        # ngpt self attention and ffn hyperparameters
        ngpt_sqk_init_value = 1.0,
        ngpt_sqk_init_scale = 0.03125,
        use_exclusive_attention = True,
        ngpt_alpha_value_attn = 0.05,
        ngpt_alpha_scale_attn = 0.03125,
        ngpt_alpha_value_mlp = 0.05,
        ngpt_alpha_scale_mlp = 0.03125,
        ngpt_suv_value = 1.0,
        ngpt_suv_scale = 1.0,
        ngpt_sz_init_value = 1.00,
        ngpt_sz_init_scale = 0.03125,

        # Passing total step count for warm up step calculations:
        dataset_total_steps = 65000,

        **kwargs
    ):
        # General Model Hyperparameters
        self.hidden_size = hidden_size
        self.sqrt_hidden_size = sqrt_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.intermediate_size = intermediate_size
        self.norm_eps = norm_eps
        self.hidden_act = hidden_act
        self.swiglu_s_init = swiglu_s_init
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.bias = bias
        self.use_ckpt = use_ckpt

        # Tokenization and Data Collator Hyperparameters
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.mlm_probability = mlm_probability
        self.mlm_use_span_masking = mlm_use_span_masking
        self.mlm_span_length = mlm_span_length

        # Router Hyperparameters
        self.num_router_latents = num_router_latents
        self.num_permanent_heads = num_permanent_heads
        self.selection_threshold = selection_threshold
        self.router_init_scale = router_init_scale
        self.use_sigmoid_scaling = use_sigmoid_scaling
        self.jitter_noise = jitter_noise
        self.router_grad_clip = router_grad_clip
        self.dense_warmup_steps = int(dense_warmup_steps * dataset_total_steps) 

        # Router Sparsity Hyperparameters
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_warm_up_steps = int(sparsity_warm_up_steps * dataset_total_steps) 
        self.head_target_min = head_target_min
        self.head_target_center = head_target_center
        self.head_target_max = head_target_max
        self.easiness_cdf_breakpoints = easiness_cdf_breakpoints
        self.sparsity_slack_lo = sparsity_slack_lo
        self.sparsity_slack_hi = sparsity_slack_hi

        # Router Auxiliary Hyperparameters:
        self.aux_coeff_start = aux_coeff_start
        self.aux_coeff_floor = aux_coeff_floor
        self.aux_anneal_start = int(aux_anneal_start * dataset_total_steps) 
        self.aux_anneal_steps = int(aux_anneal_steps * dataset_total_steps) 


        # ngpt self attention and ffn hyperparameters
        self.ngpt_sqk_init_value = ngpt_sqk_init_value
        self.ngpt_sqk_init_scale = ngpt_sqk_init_scale
        self.use_exclusive_attention = use_exclusive_attention
        self.ngpt_alpha_value_attn = ngpt_alpha_value_attn
        self.ngpt_alpha_scale_attn = ngpt_alpha_scale_attn
        self.ngpt_alpha_value_mlp = ngpt_alpha_value_mlp
        self.ngpt_alpha_scale_mlp = ngpt_alpha_scale_mlp
        self.ngpt_suv_value = ngpt_suv_value
        self.ngpt_suv_scale = ngpt_suv_scale
        self.ngpt_sz_init_value = ngpt_sz_init_value
        self.ngpt_sz_init_scale = ngpt_sz_init_scale

        super().__init__(**kwargs)



# Define Embedding Layer
class HELMEmbedding(nn.Module):

    # Initialize Embedding Layer
    def __init__(self, config):
        super().__init__()

        # Embedding Matrix size() : [vocab_size, hidden_size]
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

    # Forward Pass (yes, its literally 3 lines)
    def forward(self, input_ids):

        # Map input_ids from Word Embeddings (standard: raw learned embeddings, no unit-norm)
        word_embeds = self.word_embeddings(input_ids)

        # Return
        return word_embeds




# NOVEL: Multi-Latent Summary Router to decide which heads to use
class HELMMultiViewRouter(nn.Module):

    # Initialize the following:
    #   - Summary Query Matrix (q_down_proj)
    #   - Latent Importance Weights (l_i_weights)
    #   - Router_Init_Scale (tau)
    #   - Linear Router Gate (q_down_proj)
    def __init__(self, config):
        super().__init__()

        # Yoink some things from config
        self.config = config
        self.scale = config.sqrt_hidden_size
        self.num_elastic_candidates = config.num_attention_heads - config.num_permanent_heads


        # Summary Query Matrix size() : [hidden_size, num_router_latents]
        self.q_down_proj = nn.Linear(
            config.hidden_size,
            config.num_router_latents,
            bias = config.bias
        )

        # Latent Importance Weights
        # size() [num_router_latents]
        self.l_i_weights = nn.Parameter(
            torch.ones(config.num_router_latents)
        )

        # Router_Init_Scale size() : [1]
        self.tau = nn.Parameter(torch.tensor(config.router_init_scale))

        # Linear Router Gate size() : [hidden_size, num_attention_heads - num_permanent_heads]
        self.q_up_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads - config.num_permanent_heads,
            bias = config.bias
        )

    # Map BT_easiness -> Target # of heads
    # Problem: most of the BT_easiness_score are around .34 mark
    # We need to now center the BT easiness around the median, where a score with 0.34 should be assigned to 8/16 heads should, not 0.5
    def _easiness_to_target(self, sigmoid_scores, easiness_score):

        batch = sigmoid_scores.size(0)
        device = sigmoid_scores.device
        total_heads = self.config.num_attention_heads
        permanent_heads = self.config.num_permanent_heads

        # Get Bucket Target Values (or default to these )
        h_min = float(getattr(self.config, "head_target_min", permanent_heads + 2))
        h_ctr = float(getattr(self.config, "head_target_center", 0.5 * (h_min + total_heads)))
        h_max = float(getattr(self.config, "head_target_max", total_heads))

        # I clamped theses values before, but we'll do it here just in case
        # Size: [batch] (of easiness_score)
        easiness_score = easiness_score.to(torch.float32).view(batch).clamp(0.0,1)
        bp = getattr(self.config, "easiness_cdf_breakpoints", None)

        # Convert the breakpoints to device
        breaks = torch.as_tensor(bp, device = device, dtype=torch.float32)
        num_breaks = breaks.numel() - 1
        pos = torch.searchsorted(breaks, easiness_score, right=True).clamp(1, num_breaks)
        lo = breaks[pos - 1]; hi = breaks[pos]
        frac_in = (easiness_score - lo) / (hi - lo + 1e-8)
        q = ((pos - 1).to(torch.float32) + frac_in) / num_breaks
        q = q.clamp(0.0, 1.0)
        hard = q < 0.5
        t_hard = h_ctr + (h_max - h_ctr) * (0.5 - q) / 0.5
        t_easy = h_ctr + (h_min - h_ctr) * (q - 0.5) / 0.5
        t_total = torch.where(hard, t_hard, t_easy)

        return (t_total - permanent_heads).clamp(0.0, float(total_heads - permanent_heads))

    # Pass in only Hidden States
    # Don't pass in attention mask bc theres no attention here (duh)
    def forward(self, hidden_states, step_tensor, easiness_score):

        #################### FINALIZED LOGIC ####################

        # Write vars for cleaner code
        q_down_proj = self.q_down_proj
        l_i_weights = self.l_i_weights
        tau = self.tau
        q_up_proj = self.q_up_proj
        scale = self.scale
        self.selection_threshold = self.config.selection_threshold

        # Norm Query Matrix
        # Requires .weight since the matrix was defined before
        q_down_proj = justnorm(q_down_proj.weight, dim = 1).to(hidden_states.dtype)

        # Multiply the hidden_state by Down projection (q_down_proj)
        # Call it "scanner"
        # Size: [b, s, hidden_size] * [hidden_size * num_router_latents] = [b, s, num_router_latents]
        scanner = F.linear(justnorm(hidden_states), q_down_proj)

        # Apply Softmax to entire sequences (sequence level routing)
        scanner_softmax = F.softmax(scale * scanner, dim = 1)

        # Apply Transpose to allow for dimension matching
        # [b, s, num_router_latents] -> [b, num_router_latents, s]
        scanner_softmax = scanner_softmax.transpose(1,2)

        # Create Latent Vectors (Summary of the sequence in 4 vectors)
        # Size: [b,num_router_latents,s] * [b, s, hidden_size] = [b, num_router_latents, hidden_size]
        # Use bmm (batch matrix matric product) b/c [b, n_r_l, s] * [b, s, h_s] (dims don't match up normally)
        # Could've transposed, but this is more memory efficient
        latents = torch.bmm(scanner_softmax, justnorm(hidden_states))

        # Scale latents by Learnable important parameters (l_i_weights)
        # Softmax them first
        l_i_weights = F.softmax(l_i_weights, dim = 0)

        # Apply l_i_weights to latents
        # Sum the Latents together
        # Size: ([b, num_router_latents, hidden_size] * broadcast [1, num_router_latents, 1]) and sum the latents = [b, 1 (size of pooled_latents when we added them together), hidden_size]
        pooled_latents = (latents * l_i_weights.view(1, -1, 1)).sum(dim=1, keepdim = True)

        # Normalize q_up_proj
        # Requires .weight since the matrix was defined before
        # size [total_elastic_heads, hidden_size]
        q_up_proj = justnorm(q_up_proj.weight, dim = 1).to(pooled_latents.dtype)

        # Multiply the latents by the classifer (q_up_proj)
        # Call it "class_scores"
        # Size: [b, 1, hidden_size] * [hidden_size, total_elastic_heads] = [b, 1, total_elastic_heads]
        class_scores = F.linear(pooled_latents, q_up_proj)

        # Multiply this by Tau (router_init_scale) and ngpt scaler sqrt(hidden_size), or should we???
        class_scores = class_scores * tau

        # Sigmoid Scores
        # Size: still [b, 1, total_elastic_heads], but with sigmoid scores
        sigmoid_scores = torch.sigmoid(class_scores)

        #################### FINALIZED LOGIC ENDS HERE ####################

        # Hard Mask: of 1s and 0s based on whether the sigmoid score > threshold (0.5)
        # Size: [b, 1, total_elastic_heads]
        flat_mask = (sigmoid_scores > self.selection_threshold).float()

        # Dense warmup: ensure all heads are active before dense_warmup_steps
        dense_warmup_steps = self.config.dense_warmup_steps
        in_dense_warmup = step_tensor < dense_warmup_steps
        # Use where pattern to un-mask
        # torch.ones_like (copies all metadata (device, datatype)) ; torch.ones requires you to define all metadata + shape
        flat_mask = torch.where(in_dense_warmup, torch.ones_like(flat_mask), flat_mask)

        # Telemetry hooks
        self.save_flat_mask = flat_mask.detach()
        self.save_sigmoid_scores = sigmoid_scores.detach()

        # use_sigmoid_scaling = True: router_mask = Sigmoid values and 0s (Accuracy)
        # use_sigmoid_scaling = False: router_mask = 1s and 0s (Efficiency)
        flat_mask = flat_mask * sigmoid_scores if self.config.use_sigmoid_scaling else flat_mask

        # Apply STE for Dead Router Heads during backprop
        # Must happen after sigmoid_scaling or else torch could believe the sigmoid scaling are dynamically linked
        # .detach() Ignored during Backprop (Autograd doesn't see anything with.detach(), so when backprop happens, they disappear)
        # Forward pass: flat_mask- sigmoid_scores + sigmoid_scores = flat_mask
        # Backward pass: sigmoid_scores
        flat_mask = flat_mask.detach() - sigmoid_scores.detach() + sigmoid_scores

        # Add attention dimensions: [b, 1, total_elastic_heads] -> [b, num_elastic_heads, 1, 1]
        router_mask = flat_mask.view(flat_mask.size(0), -1, 1, 1)

        # If permanent_heads are used, add the columns
        # size(): [b, num_attention_heads, 1 , 1]
        if (self.config.num_permanent_heads > 0):
            permanent_head_scores = torch.ones(
                flat_mask.size(0),
                self.config.num_permanent_heads,
                1,
                1,
                device=router_mask.device,
                dtype=router_mask.dtype
            )
            router_mask = torch.cat((permanent_head_scores, router_mask), dim = 1)

        #################### LOSS CALCULATIONS ####################
        # In both previous implementations, the model could cheat by turning off all the heads to reduce loss
        # The old formulas were okay for scores passed into variant activations (softmax), but breaks at invariant activation (sigmoid)
        # We need to solve this by redefining how these losses are being calculated

        if self.training:

            # Generate the target-head count from easiness (soft PRIOR; CE can override a wrong label)
            elastic_target_num_heads =  self._easiness_to_target(sigmoid_scores, easiness_score)

            # ########## SPARSITY ##########: Ensure the correct # of heads are being activated

            # Why not count the number of heads to use > 0.5 ? Ans: > produces gradients of 0. Plus, we don't account for on edge heads (i.e .49)
            # By using the sum of the sigmoid scores along the head dimension, the gradient sees the proportion to how "almost on" heads are
            # Just think about this as how many heads should be on?
            num_head_preds = sigmoid_scores.squeeze(1).sum(-1)

            # Define our lower and upper clearances
            slack_lo = self.config.sparsity_slack_lo
            slack_hi = self.config.sparsity_slack_hi

            # WOW check this out:
            # If the computed value is < 0, then it becomes 0 (relu)
            # preds - upperbound for the over (If its less than the upperbound -> 0)
            # lowerbound - preds for the under (If its greater than the lowerbound -> 0)
            # We can define an over or under this way
            over = torch.relu(num_head_preds - (elastic_target_num_heads + slack_hi))
            under = torch.relu((elastic_target_num_heads - slack_lo) - num_head_preds)
            # Squared Sum penalty to extremely punish big changes
            raw_sparsity = (over**2 + under**2).mean()
            # Sparsity warmup
            # Denom: fixed. sparsity starts at 0 and reach 1 when step tensor reaches it
            denom = max(1, self.config.sparsity_warm_up_steps)
            # Ramp: go from 0 -> 1 starting at dense_warm_up -> sparsity_warm_up_steps + dense_warm_up
            ramp = torch.clamp((step_tensor.float() - dense_warmup_steps) / denom, 0.0, 1.0)
            self.sparsity_loss = ramp * self.config.sparsity_lambda * raw_sparsity

            # ########## SPARSITY ENDS ##########

            # ########## AUXILIARY BEGINS ##########
            # aux_loss: ensure that the router doesn't route the same head everytime (even routing)
            # This needs to be designed so its scale invariant (or else collapse will be encouraged)
            # CV^2 (coeff of variation^2) of per-head usage

            # First calculate the average sigmoid value per head in all the seqs in the batch
            # Size: [num_elastic_heads]
            head_avg_scores = sigmoid_scores.squeeze(1).mean(0)
            cv2 = head_avg_scores.var(unbiased = False) / (head_avg_scores.mean()**2 + 1e-6)

            # Yoink from config
            aux_start = self.config.aux_coeff_start
            aux_floor = self.config.aux_coeff_floor
            aux_begin = self.config.aux_anneal_start
            aux_steps = self.config.aux_anneal_steps

            # aux_frac: go from 1 -> 0 starting at aux_anneal_start -> aux_anneal_start + aux_anneal_steps
            aux_frac = torch.clamp((step_tensor.float() - aux_begin) / aux_steps, 0.0, 1.0)
            aux_coeff = aux_start + (aux_floor - aux_start) * aux_frac
            self.aux_loss = aux_coeff * cv2

            # ########## AUXILIARY ENDS ##########

        else:
            self.aux_loss = torch.tensor(0.0, device=hidden_states.device)
            self.sparsity_loss = torch.tensor(0.0, device=hidden_states.device)

        # Cast the router to matching data_type before returning:
        router_mask = router_mask.to(hidden_states.dtype)

        # Return Mask
        # [b, num_attention_heads, 1 , 1]
        return router_mask



# RoPE Class
class RotaryEmbeddings(nn.Module):

    # Initialize the Following
    # rope_theta
    # max_position_embeddings
    # sin & cos table
    def __init__(self, dim, max_position_embeddings, rope_theta = 160000):
        super().__init__()

        # Define inverse of frequencies
        # size(): [dim/2]
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))

        # Create position vector
        # size(): [max_position_embeddings]
        t = torch.arange(max_position_embeddings, dtype = inv_freq.dtype)

        freqs = torch.outer(t, inv_freq)

        freqs = torch.cat((freqs, freqs), dim = -1)


        # Save the Sine and Cosine
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    # Implement rotate_half (Allows for clean rotation mechanics)
    def rotate_half(self, x):

        # Take x as the first half
        x1 = x[..., : x.shape[-1] // 2]

        # Take y was the second half
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat((-x2, x1), dim = -1)


    # Implement apply_rotary_embeddings
    # Does RoPE
    # Expected input size: [b, num_attention_heads, seq_len, dim]
    # Output: [b, num_attention_heads, seq_len, dim]
    def forward(self, x):

        # Get token length
        seq_len = x.shape[-2]

        # Take a slice of the cos and sin tables
        x_cos = self.cos[:seq_len, ...].to(dtype=x.dtype)
        x_sin = self.sin[:seq_len, ...].to(dtype=x.dtype)

        # Return RoPE matrix
        return (x * x_cos) + (self.rotate_half(x) * x_sin)



# Self Attention
# Literally Just Self Attention
# QKV cross self attention
# Use RoPE
# Output Matrix
# Speicfics about training (masked training)
# MODIFICATION: USE FLEX ATTENTION TO ALLOW FOR BATCHED INFERENCE
class HELMSelfAttention(nn.Module):

    # Initialize the following:
    #   - QKV matrix
    #   - Output matrix
    #   - Scaling vector sqk for q and k
    #   - RoPE Module
    def __init__(self, config):
        super().__init__()

        # Grabbing config values from convience
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_permanent_heads = config.num_permanent_heads
        self.d_head = config.hidden_size // config.num_attention_heads
        self.ngpt_sqk_init_value = config.ngpt_sqk_init_value
        self.ngpt_sqk_init_scale = config.ngpt_sqk_init_scale
        self.config = config

        self._eval_backend = "dense"
        self._flex_compiled = False
        self._flex_fn = None
        self._block_mask_fn = None


        # QKV Matrix
        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias = config.bias
        )

        # RoPE Module
        self.RoPE = RotaryEmbeddings(
            self.d_head,
            config.max_position_embeddings,
            config.rope_theta
        )

        # (standard) no sqk: q/k are used as projected, scaled by the usual 1/sqrt(d_head)

        # Output Matrix
        self.output = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias = config.bias
        )

    # Configure the eval-time attention backend. Call via model.enable_efficient_inference(...).
    #   backend="flex"  : FlexAttention; set compile=True on GPU for the fused kernel (recommended).
    #   backend="gather": compact gather/scatter SDPA, no torch.compile needed.
    #   backend="dense" : compute-all-then-mask (default; what training uses).
    def set_eval_backend(self, backend="flex", compile=True):
        self._eval_backend = backend
        self._flex_compiled = bool(compile)
        self._flex_fn = None
        self._block_mask_fn = None

    def _flex_attn(self, q, k, v, block_mask, scale):
        if self._flex_fn is None:
            from torch.nn.attention.flex_attention import flex_attention
            self._flex_fn = torch.compile(flex_attention) if self._flex_compiled else flex_attention
        return self._flex_fn(q, k, v, block_mask=block_mask, scale=scale)

    def _build_block_mask(self, mask_mod, B, H, S, device):
        if self._block_mask_fn is None:
            from torch.nn.attention.flex_attention import create_block_mask
            # compiling create_block_mask avoids materializing the full SxS mask for long sequences
            self._block_mask_fn = torch.compile(create_block_mask) if self._flex_compiled else create_block_mask
        return self._block_mask_fn(mask_mod, B, H, S, S, device=device)

    # Define Training
    def forward(self, hidden_states, attention_mask, router_mask):

        # Obtain projection from hidden_states onto QKV
        # size(): [b, seq_len, hidden_size * 3]
        qkv_proj = cast_linear(hidden_states, self.qkv)

        # Obtain Hidden Size
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Split Projects
        # q, k, v size(): [b, seq_len, hidden_size]
        q, k, v = qkv_proj.split(hidden_size, dim=-1)

        # (standard) plain scaled-dot-product scale, 1/sqrt(d_head)
        attn_scale = 1.0 / math.sqrt(self.d_head)

        eval_backend = self._eval_backend

        # Reshape q,k,v
        # q, k, v size(): [b, seq_len, num_attention_heads, d_head]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.d_head)

        # Reshape q,k,v
        # q, k, v size(): [b, num_attention_heads, seq_len, d_head]
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)


        # TRAINING / TPU MODE
        if (self.training or eval_backend == "dense"):

            # (standard) RoPE on q,k; no unit-norm, no sqk
            q = self.RoPE(q)
            k = self.RoPE(k)

            # Apply Attention (standard 1/sqrt(d_head) scale)
            # final size(): [b, num_attention_heads, seq_len, d_head]
            context_layer = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask.to(q.dtype),
                scale=attn_scale,
            )

            # Add Exclusive Attention (better results?)
            if (self.config.use_exclusive_attention):
                Vn = torch.nn.functional.normalize(v, dim=-1)
                context_layer = context_layer - (context_layer * Vn).sum(dim=-1, keepdim=True) * Vn

            if router_mask is not None:
                # Apply Broadcasting Mask (expand_as() good for XLA)
                # size(): [b, num_attention_heads, seq_len, d_head]
                context_layer = context_layer * router_mask.expand_as(context_layer)

            # Apply Jitter Noise to the Permanent heads during training
            if self.training and self.num_permanent_heads > 0:

                # Take the permanent heads:
                permanent_heads = context_layer[:,:self.num_permanent_heads, :, :]

                # Take the elastic heads:
                elastic_heads = context_layer[:, self.num_permanent_heads:, :, :]

                # Apply dropout
                permanent_heads = F.dropout(permanent_heads, p = self.config.jitter_noise, training = self.training)

                # Combine back together
                context_layer = torch.cat((permanent_heads, elastic_heads),dim = 1)

            # Reshape
            # size(): [b, seq_len, num_attention_heads, d_head]
            context_reshaped = context_layer.permute(0, 2, 1, 3).contiguous()

            # Flatten the last two dimensions:
            # size(): [b, seq_len, num_hidden_size]
            context_reshaped = context_reshaped.view(batch_size, seq_len, -1)

            # Project context onto the Output Matrix
            context_layer = cast_linear(context_reshaped, self.output)

        # FLEX ATTENTION (for GPUs)
        elif eval_backend == "flex" and batch_size > 1:

            # (standard) RoPE on q,k; no unit-norm, no sqk
            q = self.RoPE(q)
            k = self.RoPE(k)

            # Router_mask scores, 1 or 0 or sigmoid scaling
            # [b, num_attention_heads, 1 , 1] -> [batch, num_attention_heads]
            active = (router_mask[:, :, 0, 0] > 0)

            # Boolean attention mask
            # [batch_size, 1, 1, seq_len] -> [batch, seq_len]
            key_valid = (attention_mask[:, 0, 0, :] >=0)

            # mask_mod: attend / calculate only if the head is on and its not a padding token
            def mask_mod(bi, hi, qi, ki):
                return active[bi, hi] & key_valid[bi, ki]

            # Prep the block to be passed into flex attention
            block_mask = self._build_block_mask(
                mask_mod, batch_size, self.num_attention_heads,seq_len, q.device
            )

            # Apply flex attention
            context_layer = self._flex_attn(
                q, k, v, block_mask = block_mask, scale = attn_scale
            )

            # Add Exclusive Attention (better results?)
            if (self.config.use_exclusive_attention):
                Vn = torch.nn.functional.normalize(v, dim=-1)
                context_layer = context_layer - (context_layer * Vn).sum(dim=-1, keepdim=True) * Vn

            # Apply router mask to 0 the heads of the context layer
            # [batch, num attention heads, seq_len, head dim] (router_mask [batch, num_attention_heads, 1,1] was broadcasted)
            context_layer = context_layer * router_mask.expand_as(context_layer)

            # Flatten the last two dimensions:
            # size(): [b, seq_len, num_hidden_size]
            context_reshaped = context_reshaped.view(batch_size, seq_len, -1)

            # Project context onto the Output Matrix
            context_layer = cast_linear(context_reshaped, self.output)

        # Single query effieincy
        else:

            # Find the heads that are on
            # nonzero(): [1, num_attention_heads, 1, 1] -> [num_active_heads, 1]
            # squeeze(): [num_active_heads, 1] -> [num_active_heads] (indices)
            active_indices = torch.nonzero(router_mask[0, :, 0, 0]).squeeze(-1)

            # Reshape q,k,v
            # q, k, v size(): [b, seq_len, num_attention_heads, hidden_size]
            q = q.view(batch_size, seq_len, self.num_attention_heads, self.d_head)
            k = k.view(batch_size, seq_len, self.num_attention_heads, self.d_head)
            v = v.view(batch_size, seq_len, self.num_attention_heads, self.d_head)

            # Reshape q,k,v
            # q, k, v size(): [b, num_attention_heads, seq_len, hidden_size]
            q = q.permute(0,2,1,3)
            k = k.permute(0,2,1,3)
            v = v.permute(0,2,1,3)

            # 2. Extract the parts used by the active heads
            # size(): [1, num_attention_heads, seq_len, d_head] ->  [1, num_active_heads, seq_len, d_head]
            q_sliced = q[:, active_indices, :, :]
            k_sliced = k[:, active_indices, :, :]
            v_sliced = v[:, active_indices, :, :]

            # (standard) RoPE on q,k; no unit-norm, no sqk
            q_sliced = self.RoPE(q_sliced)
            k_sliced = self.RoPE(k_sliced)

            # Flash Attention (only for GPUs where on-the-fly splicing can exist)
            # size(): [b, num_active_heads, seq_len, d_head]
            context_sliced = F.scaled_dot_product_attention(
                q_sliced, k_sliced, v_sliced,
                attn_mask=attention_mask.to(q.dtype),
                scale=attn_scale
            )

            # Add Exclusive Attention (better results?)
            if (self.config.use_exclusive_attention):
                Vn = torch.nn.functional.normalize(v_sliced, dim=-1)
                context_sliced = context_sliced - (context_sliced * Vn).sum(dim=-1, keepdim=True) * Vn

            # STE tie to the router
            # Note: If use_sigmoid_scaling = True: Scales the router mask back to the sigmoid values
            # (since active indices were just indices of the values, not the real values)
            # If use_sigmooid_scaling = False, then multiplying by 1 does mathimatically nothing
            active_weights = router_mask[:, active_indices, :, :]
            context_sliced = context_sliced * active_weights

            # 5. Reshape for the output linear layer
            # [1, num_active, seq_len, d_head] -> [1, seq_len, num_active, d_head]
            context_reshaped = context_sliced.permute(0, 2, 1, 3).contiguous()

            # Flatten the last two dimensions: [1, seq_len, num_active * d_head]
            context_reshaped = context_reshaped.view(batch_size, seq_len, -1)

            # 6. Map the active head indices to their exact hidden dimension indices
            # Example: Head 1 with d_head=64 generates indices 64 through 127
            dim_offsets = torch.arange(self.d_head, device=hidden_states.device)
            active_dims = (active_indices.unsqueeze(1) * self.d_head + dim_offsets).view(-1)

            # 7. Slice the input columns of the output weight matrix
            # original shape [hidden_size, hidden_size] -> [hidden_size, num_active * d_head]
            sliced_weight = self.output.weight[:, active_dims]

            # 8. Perform the compressed functional linear projection
            context_layer = F.linear(context_reshaped, sliced_weight, bias=self.output.bias)

        # Return context_layer (normalization occurs in HELMMLP)
        return context_layer



# HELMMLP - standard SwiGLU feed-forward network.
# Pre-LN + additive residual are handled by HELMBlock; this module is a pure FFN.
class HELMMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # SwiGLU: expand to 2*intermediate (gate + value), then project back down
        self.mlp_exp = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)

    def forward(self, x):
        # x is already LayerNorm'd by the block (Pre-LN). Returns FFN output (no residual).
        uv = cast_linear(x, self.mlp_exp)              # [b, s, 2*intermediate]
        u, v = torch.chunk(uv, 2, dim=-1)              # each [b, s, intermediate]
        x_mlp = u * self.silu(v)                        # SwiGLU
        return cast_linear(x_mlp, self.mlp_proj)        # [b, s, hidden]



# HELMBLOCK = HELMMultiViewRouter + HELMSelfAttention (which defines RotaryEmbeddigs) + HELMMLP
# This is 1 transformer layer
class HELMBlock(nn.Module):

    # Define the Following:
    #   - HELMMultiViewRouter
    #   - HELMSelfAttention
    #   - HELMMLP
    def __init__(self, config):
        super().__init__()
        self.mlt_vw_rtr = HELMMultiViewRouter(config)
        self.attn = HELMSelfAttention(config)
        self.mlp = HELMMLP(config)
        # (standard) Pre-LN: one norm before attention, one before the FFN
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.bias)

    # Standard Pre-LN transformer block with additive residuals.
    # The router still reads the RAW block input (unchanged from the nGPT version),
    # so its behaviour/telemetry is identical; only the backbone math changes.
    def forward(self, hidden_states, attention_mask, step_tensor, easiness_score = None):
        router_mask = self.mlt_vw_rtr(hidden_states, step_tensor, easiness_score)
        aux_loss = self.mlt_vw_rtr.aux_loss
        sparsity_loss = self.mlt_vw_rtr.sparsity_loss

        # attention sublayer: h = h + Attn(LN1(h))
        attn_output = self.attn(self.ln1(hidden_states), attention_mask, router_mask)
        hidden_states = hidden_states + attn_output

        # feed-forward sublayer: h = h + FFN(LN2(h))
        hidden_states = hidden_states + self.mlp(self.ln2(hidden_states))

        return hidden_states, aux_loss, sparsity_loss



# HELMModel - HELM without the head
class HELMModel(nn.Module):

    # Define the following:
    #   - HELMEmbedding
    #   - HELMBlock
    def __init__(self, config):
        super().__init__()

        # Get the ckpt_attribute
        self.use_ckpt = config.use_ckpt

        # Embedding layer
        self.embedding = HELMEmbedding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [HELMBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # (standard) final norm before the LM head (Pre-LN stacks need this)
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.bias)


    # Forward Pass
    def forward(self, input_ids, attention_mask, current_step = None, easiness_score = None):

        # Build additive mask for SDPA fallback
        # Reshape Additive Mask to be 4D for SDPA [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bfloat16)
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)

        # Convert the current_step to be infinity if null, or a tensor, or a tensor of the correct datatype if its already tensor
        # We did this because we don't want to pass in a non-tensor if we are using gradient_checkpointing
        if current_step is None:
            step_tensor = torch.tensor(float("inf"), device=input_ids.device)
        elif not isinstance(current_step, torch.Tensor):
            step_tensor = torch.tensor(current_step, device=input_ids.device)
        else:
            step_tensor = current_step

        # Pass input_ids through the input
        embeddings = self.embedding(input_ids)

        # Set Embeddings to be hidden_states
        hidden_states = embeddings.to(torch.bfloat16)

        # Accumulate aux_loss and sparsity_loss
        total_aux_loss = 0
        total_sparsity_loss = 0

        # Run Tranformer Blocks
        for block in self.blocks:
            # Use Gradient Checkpointing
            if self.use_ckpt and self.training:
                _ckpt = (_xla_checkpoint if (_xla_checkpoint is not None
                         and hidden_states.device.type == "xla")
                         else torch.utils.checkpoint.checkpoint)
                hidden_states, aux_loss, sparsity_loss = _ckpt(
                    block,
                    hidden_states,
                    attention_mask,
                    step_tensor,
                    easiness_score,
                    use_reentrant= True if hidden_states.device.type == "xla" else False
                )
            # Or Standard Forward Pass
            else:
                hidden_states, aux_loss, sparsity_loss = block(hidden_states, attention_mask, step_tensor, easiness_score)

            total_aux_loss += aux_loss
            total_sparsity_loss += sparsity_loss

        # (standard) final norm
        hidden_states = self.final_norm(hidden_states)

        # Return hidden state (feature extraction / context location prediction) & special losses
        # hidden_states: [b, seq_len, hidden_size]
        return hidden_states, total_aux_loss, total_sparsity_loss



# HELMModelforMaskedLM
class HELMForMaskedLM(PreTrainedModel):

    # Define the Config for the HF push_to_hub() function
    config_class = HELMConfig

    # Define the Following:
    #   - HELMModel
    #   - classifier
    #   - Head layer scaling vector
    def __init__(self, config):
        super().__init__(config)

        # Define the Model
        self.model = HELMModel(config)

        # Define the head Layer (standard: plain linear, no nGPT sz scaling)
        self.classifier = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias = config.bias
        )

        # HF Function to call _init_weights() function
        self.post_init()

    # Initialize weights (pulled from ngpt model.py)
    def _init_weights(self, module):

        # If it's an nn.Linear, initialize it with the initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # If it's an nn.Linear, initialize it with the initializer_range also)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    # Define Function to normalize_ngpt_matrices
    # Flip every self-attention layer into an efficient eval backend for GPU inference.
    # Leave this OFF for TPU training/validation (the default "dense" path is static-shape friendly).
    #   model.eval(); model.enable_efficient_inference("flex")        # GPU, fused (recommended)
    #   model.eval(); model.enable_efficient_inference("gather")      # no torch.compile needed
    # On GPU also wrap inference in torch.compile, or pass compile=True (default) to fuse flex.
    def enable_efficient_inference(self, backend="flex", compile=True):
        for block in self.model.blocks:
            block.attn.set_eval_backend(backend=backend, compile=compile)
        return self

    # Standard backbone does NOT renormalize weights each step (that was an nGPT requirement).
    # Kept as a no-op so the trainer's per-step call is harmless and needs no change.
    def normalize_ngpt_matrices(self):
        return

    # Get all necessary telemetrics & return as dict
    @torch.no_grad()
    def get_telemetry(self):

        # Define Telemetry:
        telemetry = {}

        # Iterate Through All Layers
        for i, block in enumerate(self.model.blocks):

            # ========== ROUTER ==========

            # Get the router
            router = block.mlt_vw_rtr

            # Derived / Intermediate Values:
            #   - sigmoid_scores
            #   - flat_mask
            #   - Elastic head ratio

            # sigmoid_scores
            telemetry[f"layer_{i}_sigmoid_scores"] = router.save_sigmoid_scores.float().cpu()

            # flat_mask
            telemetry[f"layer_{i}_flat_mask"] = router.save_flat_mask.float().cpu()

            # Elastic head ratio
            telemetry[f"layer_{i}_elastic_head_ratio"] = router.save_flat_mask.float().cpu().mean().item()

            # Persistent Parameters
            #   - l_i_weights
            #   - tau

            # l_i_weights (learnable importance when gather 4 latents)
            telemetry[f"layer_{i}_l_i_weights"] = router.l_i_weights.detach().cpu()

            # tau (sigmoid scaling factor)
            telemetry[f"layer_{i}_tau"] = router.tau.detach().item()

            # ============================


            # ---------- PRE-LN GAINS (standard-backbone health signals) ----------

            # ln1 gain (pre-attention LayerNorm weight)
            ln1_tensor = block.ln1.weight.detach().cpu()
            telemetry[f"layer_{i}_ln1_gain_mean"] = ln1_tensor.mean().item()
            telemetry[f"layer_{i}_ln1_gain_std"] = ln1_tensor.std().item()
            telemetry[f"layer_{i}_ln1_gain_hist"] = ln1_tensor

            # ln2 gain (pre-FFN LayerNorm weight)
            ln2_tensor = block.ln2.weight.detach().cpu()
            telemetry[f"layer_{i}_ln2_gain_mean"] = ln2_tensor.mean().item()
            telemetry[f"layer_{i}_ln2_gain_std"] = ln2_tensor.std().item()
            telemetry[f"layer_{i}_ln2_gain_hist"] = ln2_tensor

            # attention output-projection weight scale (Frobenius norm; watch for residual blow-up)
            telemetry[f"layer_{i}_attn_out_wnorm"] = block.attn.output.weight.detach().float().norm().item()

            # ------------------------------------------------------------


        # final norm gain before the LM head
        fn_tensor = self.model.final_norm.weight.detach().cpu()
        telemetry["final_norm_gain_mean"] = fn_tensor.mean().item()
        telemetry["final_norm_gain_std"] = fn_tensor.std().item()
        telemetry["final_norm_gain_hist"] = fn_tensor

        return telemetry



    # Forward pass
    def forward(self, input_ids, attention_mask, current_step = None, easiness_score = None):

        # Gather Context from the model
        # features: [b, seq_len, hidden_size]
        features, total_aux_loss, total_sparsity_loss = self.model(input_ids, attention_mask, current_step, easiness_score)

        # project features onto classifier (standard: no sz scaling)
        # [b, seq_len, hidden_size] * [hidden_size, vocab_size] = [b, seq_len, vocab_size]
        logits = cast_linear(features, self.classifier)

        # Return Logits
        return logits, total_aux_loss, total_sparsity_loss


