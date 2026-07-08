%%writefile model.py

##################################################
# Defines the HELM V1 architecture
# Inherited the PretrainedConfig and PreTrainedModel
# Utilizes many of the concepts found in Nvidia's 2024 nGPT architecture
# Initializes the Weights
# Uses 36 to approx. sqrt(1280)
# Uses 0.028 to approx. 1/sqrt(280)

# Note this version doesn't have sparsity_loss and the s_threshold shouldn't be learnable
# This version also will cause a recompilation post-topk warm-up
# Additionally, jitter_noise in the self attention is not accounted for


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
from transformers import AutoTokenizer
from transformers import PretrainedConfig, PreTrainedModel



# modified justnorm() function
# better than F.normalize(), max() causes micro walls during gradient descent
# better than nGPT's version, prevents division by 0 error
def justnorm(x, dim = -1, eps = 1e-12):
    res = x / (x.norm(p=2, dim=dim, keepdim=True) + eps)
    return res

# Hugging Face Core Method (for future deployment)
class HELMConfig(PretrainedConfig):

    model_type = "helm"

    def __init__(
        self,
        hidden_size = 1280,
        sqrt_hidden_size = 36,
        max_position_embeddings = 4096,
        initializer_range = 0.028,
        num_hidden_layers = 12,
        num_attention_heads = 20,
        rope_theta = 160000,
        intermediate_size = 3456,
        norm_eps = 1e-12,
        hidden_act = "swiglu",
        swiglu_s_init = 1.0,
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
        num_router_latents = 4,
        max_active_heads = 6,
        num_permanent_heads = 2,
        selection_threshold_init = 0.5,
        router_init_scale = 10.0,
        jitter_noise = 0.01,
        sparsity_lambda = 0.01, 
        router_aux_loss_coeff = 0.02,
        router_grad_clip = 0.05,    
        sparsity_warm_up_steps = 2000,
        use_sigmoid_scaling = False,
        ngpt_sqk_init_value = 1.0,  
        ngpt_sqk_init_scale = 0.028,
        ngpt_alpha_value_attn = 0.05,
        ngpt_alpha_scale_attn = 0.028,
        ngpt_alpha_value_mlp = 0.05,
        ngpt_alpha_scale_mlp = 0.028,
        ngpt_suv_value = 1.0,
        ngpt_suv_scale = 1.0,
        ngpt_sz_init_value = 1.00,
        ngpt_sz_init_scale = 0.028,
        bias = False,
        **kwargs 
    ):  
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
        self.num_router_latents = num_router_latents
        self.max_active_heads = max_active_heads
        self.num_permanent_heads = num_permanent_heads
        self.num_elastic_heads = max_active_heads - num_permanent_heads
        self.selection_threshold_init = selection_threshold_init
        self.router_init_scale = router_init_scale
        self.jitter_noise = jitter_noise
        self.sparsity_lambda = sparsity_lambda
        self.router_aux_loss_coeff = router_aux_loss_coeff
        self.router_grad_clip = router_grad_clip     
        self.sparsity_warm_up_steps = sparsity_warm_up_steps
        self.use_sigmoid_scaling = use_sigmoid_scaling
        self.ngpt_sqk_init_value = ngpt_sqk_init_value
        self.ngpt_sqk_init_scale = ngpt_sqk_init_scale
        self.ngpt_alpha_value_attn = ngpt_alpha_value_attn
        self.ngpt_alpha_scale_attn = ngpt_alpha_scale_attn
        self.ngpt_alpha_value_mlp = ngpt_alpha_value_mlp
        self.ngpt_alpha_scale_mlp = ngpt_alpha_scale_mlp
        self.ngpt_suv_value = ngpt_suv_value
        self.ngpt_suv_scale = ngpt_suv_scale
        self.ngpt_sz_init_value = ngpt_sz_init_value
        self.ngpt_sz_init_scale = ngpt_sz_init_scale
        self.bias = bias

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
        
        # Map input_ids from Word Embeddings
        word_embeds = self.word_embeddings(input_ids)

        # Normalize (an nGPT must to allow cos. sim. to work)
        embeddings = justnorm(word_embeds)

        # Return
        return embeddings



# NOVEL: Multi-Latent Summary Router to decide which heads to use
class HELMMultiViewRouter(nn.Module):
   
    # Initialize the following:
    #   - Summary Query Matrix (q_down_proj)
    #   - Latent Importance Weights (l_i_weights)
    #   - Router_Init_Scale (tau)
    #   - Linear Router Gate (q_down_proj)
    #   - Selection Threshold (s_threshold_init)
    def __init__(self, config):
        super().__init__()

        # Yoink some things from config
        self.config = config
        self.scale = config.sqrt_hidden_size
        self.num_elastic_heads = self.config.num_elastic_heads

        # Summary Query Matrix size() : [hidden_size, num_router_latents]
        self.q_down_proj = nn.Linear(
            config.hidden_size,
            config.num_router_latents,
            bias = config.bias
        )

        # Latent Important Weights
        # size() [num_router_latents]
        self.l_i_weights = nn.Parameter(
            torch.ones(config.num_router_latents)
        )

        # Router_Init_Scale size() : [1]
        self.tau = nn.Parameter(torch.tensor(config.router_init_scale))

        # s_threshold size() : [1]
        self.s_threshold = nn.Parameter(torch.tensor(config.selection_threshold_init))

        # Linear Router Gate size() : [hidden_size, num_attention_heads - num_permanent_heads]
        self.q_up_proj = nn.Linear(
            config.hidden_size,
            config.num_elastic_heads,
            bias = config.bias
        )

    # Pass in only Hidden States 
    # Don't pass in attention mask bc theres no attention here (duh)
    def forward(self, hidden_states, current_step = None):

        # Set step to infinity if not passed in (during inference)
        step = current_step if current_step is not None else float('inf')

        # Write vars for cleaner code
        q_down_proj = self.q_down_proj
        l_i_weights = self.l_i_weights
        tau = self.tau
        q_up_proj = self.q_up_proj 
        scale = self.scale
        num_elastic_heads = self.num_elastic_heads
        
        # Norm Query Matrix
        # Requires .weight since the matrix was defined before
        q_down_proj = justnorm(q_down_proj.weight, dim = 1)

        # Multiply the hidden_state by Down projection (q_down_proj)
        # Call it "scanner"
        # Size: [b, s, hidden_size] * [hidden_size * num_router_latents] = [b, s, num_router_latents]
        scanner = F.linear(hidden_states, q_down_proj)

        # Apply Softmax to entire sequences (sequence level routing)
        scanner_softmax = F.softmax(scale * scanner, dim = 1)

        # Apply Transpose to allow for dimension matching
        # [b, s, num_router_latents] -> [b, num_router_latents, s]
        scanner_softmax = scanner_softmax.transpose(1,2)

        # Create Latent Vectors (Summary of the sequence in 4 vectors)
        # Size: [b,num_router_latents,s] * [b, s, hidden_size] = [b, num_router_latents, hidden_size]
        # Use bmm since the softmax is dynamic
        latents = torch.bmm(scanner_softmax, hidden_states)

        # Normalize the latents
        latents = justnorm(latents)

        # Scale latents by Learnable important parameters (l_i_weights)
        # Softmax them first
        l_i_weights = F.softmax(l_i_weights, dim = 0)

        # Apply l_i_weights to latents
        # Sum the Latents together
        # Size: [b, 1, hidden_size]
        pooled_latents = (latents * l_i_weights.view(1, -1, 1)).sum(dim=1)

        # Normalize again (per nGPT requirements)
        pooled_latents = justnorm(pooled_latents)
        
        # Normalize q_up_proj      
        # Requires .weight since the matrix was defined before
        q_up_proj = justnorm(q_up_proj.weight, dim = 1)

        # Multiply the latents by the classifer (q_up_proj)
        # Call it "class_scores"
        # Size: [b, 1, hidden_size] * [hidden_size, num_elastic_heads] = [b, 1, num_elastic_heads]
        class_scores = F.linear(pooled_latents, q_up_proj)

        # Multiply this by Tau (router_init_scale)
        class_scores = class_scores * tau * scale

        # Sigmoid Scores
        sigmoid_scores = torch.sigmoid(class_scores)

        # Selection Strategy:
        # Format: [b, num_elastic_heads, 1, 1]
        # Use Top-K for when we are in warm-up phase
        if (step < self.config.sparsity_warm_up_steps):

            # Call top_k on the router attention heads
            _, indices = sigmoid_scores.topk(num_elastic_heads, dim = -1)
            
            # Create One-hot encoding mask for top_k
            # size(): [num_elastic_heads]
            flat_mask = F.one_hot(indices, num_classes=sigmoid_scores.size(-1)).sum(dim=1).float()

        else:
            # Standard Threshold Phase: Hard 0.0 or 1.0 decisions
            flat_mask = (sigmoid_scores > self.s_threshold).float()

        # use_sigmoid_scaling = True: router_mask = Sigmoid values and 0s (Accuracy)
        # use_sigmoid_scaling = False: router_mask = 1s and 0s (Efficiency)
        if self.config.use_sigmoid_scaling:
            # Scale back to sigmoid scores
            flat_mask = flat_mask * sigmoid_scores

        # Apply STE for Dead Router Heads during backprop
        flat_mask = flat_mask.detach() - sigmoid_scores.detach() + sigmoid_scores

        # Add attention dimensions: [b, num_elastic_heads, 1, 1]
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

        # STE math going on ________________________________________________________________________________________ LOOK INTO THIS
        if self.training:
            P = sigmoid_scores.mean(dim=0).squeeze() 
            f = (sigmoid_scores > self.s_threshold).float().mean(dim=0).squeeze() # Best approx of hard_mask here
            self.aux_loss = self.num_elastic_heads * torch.sum(f * P)
        else:
            self.aux_loss = 0.0

        # Return Mask
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
        self.d_head = config.hidden_size // config.num_attention_heads
        self.ngpt_sqk_init_value = config.ngpt_sqk_init_value
        self.ngpt_sqk_init_scale = config.ngpt_sqk_init_scale

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

        # SQK scalers right after RoPE
        self.sqk = nn.Parameter(self.ngpt_sqk_init_scale*torch.ones(self.hidden_size, dtype=torch.float32))

        # Output Matrix
        self.output = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias = config.bias
        )
    
    # Define Training
    def forward(self, hidden_states, attention_mask, router_mask):

        # Obtain projection from hidden_states onto QKV
        # size(): [b, seq_len, hidden_size * 3]
        qkv_proj = self.qkv(hidden_states)

        # Obtain Hidden Size
        batch_size, seq_len, hidden_size = hidden_states.size()

        # Split Projects
        # q, k, v size(): [b, seq_len, hidden_size]
        q, k, v = qkv_proj.split(hidden_size, dim=-1)

        # Define sqk for scaling q, k, and v
        # size(): [hidden_size]
        sqk = (self.sqk * (self.ngpt_sqk_init_value/self.ngpt_sqk_init_scale))
        # Resizing is required for when we element-wise multiply this by q and k matrice:s [1, num_attention_heads, 1, d_head] * [b, num_attention_heads, seq_len, hidden_size]
        # size(): [hidden_size]-> [1, num_attention_heads, 1, d_head]
        sqk = sqk.view(1, self.num_attention_heads, 1, self.d_head)

        # If Batch > 1 or model is in training mode: Cutting Losses and broadcast the mask
        if (batch_size > 1 or self.training):
            
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

            # Normalize q and k
            q = justnorm(q)
            k = justnorm(k)

            # Apply RoPE
            q = self.RoPE(q)
            k = self.RoPE(k)

            # Apply sqk scaling factor and justnorm to q and k (splice sqk too)
            q = sqk * q 
            k = sqk * k 

            # Apply Attention
            # Sclae by sqrt(dk)
            # A whole lot happens here. final size(): [b, num_attention_heads, seq_len, d_head]
            context_layer = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask,
                scale=math.sqrt(self.d_head)
            )

            # 🩹 THE ROBUST XLA BROADCAST FIX:
            if router_mask is not None:
                if router_mask.shape[1] != context_layer.shape[1]:
                    # We need to map 'V' views to 'H' heads (e.g., 18 views -> 20 heads)
                    # We treat the views like a 1D image and "stretch" it to the head count
                    # [B, V, 1, 1] -> [B, 1, V]
                    m = router_mask.squeeze(-1).squeeze(-1).unsqueeze(1)
                    # Interpolate to [B, 1, H]
                    m = F.interpolate(m, size=context_layer.shape[1], mode='nearest')
                    # Reshape back to [B, H, 1, 1]
                    router_mask = m.squeeze(1).unsqueeze(-1).unsqueeze(-1)
                
                # Explicit expansion for XLA safety
                # Apply Mask
                # size(): [b, num_attention_heads, seq_len, d_head]
                context_layer = context_layer * router_mask.expand_as(context_layer)

            # # Apply Mask
            # # size(): [b, num_attention_heads, seq_len, d_head]
            # context_layer = context_layer * router_mask

            # Reshape 
            # size(): [b, seq_len, num_attention_heads, d_head] 
            context_reshaped = context_layer.permute(0, 2, 1, 3).contiguous()
            
            # Flatten the last two dimensions: 
            # size(): [b, seq_len, num_attention_heads, d_head] 
            context_reshaped = context_reshaped.view(batch_size, seq_len, -1)

            # Project context onto the Output Matrix
            context_layer = self.output(context_reshaped)

        # Apply splicing logic and win on efficiency.
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

            # Normalize q and k
            q_sliced = justnorm(q_sliced)
            k_sliced = justnorm(k_sliced)

            # Apply RoPE
            q_sliced = self.RoPE(q_sliced)
            k_sliced = self.RoPE(k_sliced)

            # Apply sqk scaling factor and justnorm to q and k 
            sqk_sliced = sqk[:, active_indices, :, :]
            q_sliced = sqk_sliced * q_sliced  
            k_sliced = sqk_sliced * k_sliced  

            # Flash Attention
            # size(): [b, num_active_heads, seq_len, d_head]
            context_sliced = F.scaled_dot_product_attention(
                q_sliced, k_sliced, v_sliced, 
                attn_mask=attention_mask,
                scale=math.sqrt(self.d_head)
            )

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



# HELMMLP (FFN of nGPT architecture)
# All of this stays the same from the original nGPT paper
class HELMMLP(nn.Module):

    # Define the Following:
    #   - Constants from config (for convience?)
    #       * hidden_size
    #       * ngpt_alpha_value_attn
    #       * ngpt_alpha_scale_attn
    #       * ngpt_alpha_value_mlp
    #       * ngpt_alpha_scale_mlp
    #       * ngpt_suv_value
    #       * ngpt_suv_scale
    #   - Eigen learning rate after attention (attn_alpha)
    #   - Eigen learning rate after mlp (mlp_alpha)
    #   - MLP expansion layer (mlp_exp)
    #   - suv scaling vectors for SwiGLU (suv)
    #   - SiLU() activation (silu)
    #   - MLP projection layer (mlp_expand)
    def __init__(self, config):
        super().__init__()

        # Gather Config Values for convience
        self.hidden_size = config.hidden_size
        self.ngpt_alpha_value_attn = config.ngpt_alpha_value_attn
        self.ngpt_alpha_scale_attn = config.ngpt_alpha_scale_attn
        self.ngpt_alpha_value_mlp = config.ngpt_alpha_value_mlp
        self.ngpt_alpha_scale_mlp = config.ngpt_alpha_scale_mlp
        self.ngpt_suv_value = config.ngpt_suv_value
        self.ngpt_suv_scale = config.ngpt_suv_scale
        self.intermediate_size = config.intermediate_size

        # Alpha Eigen Update after Attention (1st Optimizer Step)
        self.attn_alpha = torch.nn.Parameter(self.ngpt_alpha_scale_attn*torch.ones(self.hidden_size, dtype=torch.float32))

        # Alpha Eigen Update after MLP (2nd Optimizer Step)
        self.mlp_alpha = torch.nn.Parameter(self.ngpt_alpha_scale_mlp*torch.ones(self.hidden_size, dtype=torch.float32))

        # MLP expansion layer
        self.mlp_exp = nn.Linear(
            self.hidden_size, 
            2 * self.intermediate_size, 
            bias = config.bias
        )

        # suv scaling vectors during SwiGLU 
        self.suv = torch.nn.Parameter(self.ngpt_suv_scale*torch.ones(2 * self.intermediate_size, dtype=torch.float32))

        # Define SiLU()
        self.silu = nn.SiLU()

        # MLP projection layer (shrink)
        self.mlp_proj  = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.bias
        )
    
    # Peform MLP from the output of the output matrix to the end of the transformer block
    def forward(self, hidden_states, hidden_states_attention):

        # Even more convience
        hidden_size = self.hidden_size
        ngpt_alpha_value_attn = self.ngpt_alpha_value_attn
        ngpt_alpha_scale_attn = self.ngpt_alpha_scale_attn
        ngpt_alpha_value_mlp = self.ngpt_alpha_value_mlp
        ngpt_alpha_scale_mlp = self.ngpt_alpha_scale_mlp
        ngpt_suv_value = self.ngpt_suv_value
        ngpt_suv_scale = self.ngpt_suv_scale

        # Mostly Lifted from the nGPT model.py

        # Define the eigen learning rate 
        # alpha >=0
        # size(): [hidden_size]
        lr = self.attn_alpha * (ngpt_alpha_value_attn / ngpt_alpha_scale_attn)
        lr = torch.abs(lr)
        
        # Apply Normalization to hidden states before and after attention
        # both size(): [b, seq_len, hidden_size]
        A_norm = justnorm(hidden_states)
        B_norm = justnorm(hidden_states_attention)
            
        # h = Norm(h + alpha_a * (h_a - h)) (element-wise)
        # size(): [b, seq_len, hidden_size]
        hidden_states_opt1 = A_norm + lr * (B_norm - A_norm)
        hidden_states_opt1 = justnorm(hidden_states_opt1)

        # Get u and v matrices by multiplying by mlp_exp
        # size(): [b, seq_len, hidden_size] * [hidden_size, 2 * intermediate_size] = [b, seq_len, 2 * intermediate_size]
        uv = self.mlp_exp(hidden_states_opt1)

        # prepare scaling vector suv
        # size(): [intermediate_size * 2] (remember, they are concatenated)
        suv = self.suv * (ngpt_suv_value/ngpt_suv_scale) * (hidden_size ** 0.5)
        
        # element-wise uv by scaling vector suv
        # size(): [b, seq_len, 2 * intermediate_size]
        uv = suv * uv  

        # Chunk uv into u and v
        # both size(): [b, seq_len, intermediate_size]
        u, v = torch.chunk(uv, 2, dim=-1)

        # Apply u * silu(v), the whole point of SwiGLU (element-wise)
        # size(): [b, seq_len, intermediate_size]
        x_mlp = u * self.silu(v)

        # Project x_mlp to the mlp_proj layer (shrink)
        # size(): [b, seq_len, intermediate_size] * [intermediate_size, hidden_size] = [b, seq_len, hidden_size]
        h_mlp = self.mlp_proj(x_mlp)

        # Define the eigen learning rate 
        # alpha >=0
        # size(): [hidden_size]
        lr = self.mlp_alpha * (ngpt_alpha_value_mlp / ngpt_alpha_scale_mlp) 
        lr = torch.abs(lr)

        # Apply Normalization to hidden states after attention and after mlp
        # both size(): [b, seq_len, hidden_size]
        A_norm = justnorm(hidden_states_opt1)
        B_norm = justnorm(h_mlp)

        # h = Norm(h + alpha_m * (h_a - h)) (element-wise)
        # size(): [b, seq_len, hidden_size]
        hidden_states_opt2 = A_norm + lr * (B_norm - A_norm)
        hidden_states_opt2 = justnorm(hidden_states_opt2)

        # Return new hidden_state
        return hidden_states_opt2



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
    
    # Define the forward pass
    # Extra: Return the aux_loss from the router
    def forward(self, hidden_states, attention_mask, current_step = None):
        router_mask = self.mlt_vw_rtr(hidden_states, current_step)
        aux_loss = self.mlt_vw_rtr.aux_loss
        attn_output = self.attn(hidden_states, attention_mask, router_mask)
        layer_output = self.mlp(hidden_states, attn_output)
        return layer_output, aux_loss



# HELMModel - HELM without the head 
class HELMModel(nn.Module):

    # Define the following:
    #   - HELMEmbedding
    #   - HELMBlock
    def __init__(self, config):
        super().__init__()

        # Embedding layer
        self.embedding = HELMEmbedding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [HELMBlock(config) for _ in range(config.num_hidden_layers)]
        )
    
    # Forward Pass
    def forward(self, input_ids, attention_mask, current_step = None):


        # Reshape Attention Mask to be 4D for SDPA [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()

        # Then make attend = 0 and mask = -inf to allow for Flash Attention compatitbility
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)

        # Pass input_ids through the input
        embeddings = self.embedding(input_ids)

        # Set Embeddings to be hidden_states
        hidden_states = embeddings 

        # Accumulate total_aux_loss

        total_aux_loss = 0

        # Run Tranformer Blocks
        # Remember to pass in current_step #
        for block in self.blocks:
            hidden_states, aux_loss = block(hidden_states, attention_mask, current_step)
            total_aux_loss += aux_loss

        # Return hidden state (feature extraction / context location prediction)
        return hidden_states, total_aux_loss



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

        # Define from Config
        self.ngpt_sz_init_value = config.ngpt_sz_init_value
        self.ngpt_sz_init_scale = config.ngpt_sz_init_scale

        # Define the Model
        self.model = HELMModel(config)

        # Define the head Layer
        self.classifier = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias = config.bias
        )

        # Define the head layer scaling vetor 
        self.sz = nn.Parameter(torch.ones(config.vocab_size, dtype=torch.float32))

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

    
    # Forward pass
    def forward(self, input_ids, attention_mask, current_step = None):

        # Gather Context from the model
        features, total_aux_loss = self.model(input_ids, attention_mask, current_step)

        # Scale / prepare sz
        sz = self.sz * (self.ngpt_sz_init_value / self.ngpt_sz_init_scale)

        # project features onto classifer
        unscaled_logits = self.classifier(features)

        # Scale the logits with sz
        logits = sz * unscaled_logits

        # Return Logits
        return logits, total_aux_loss

    
