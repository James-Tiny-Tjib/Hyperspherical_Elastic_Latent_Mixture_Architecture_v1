%%writefile model.py

##################################################
# Defines the HELM V1 architecture
# Inherited the PretrainedConfig and PreTrainedModel
# Utilizes many of the concepts found in Nvidia's 2024 nGPT architecture
# Initializes the Weights
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
        max_active_heads = 5,
        num_permanent_heads = 2,
        selection_threshold = 0.5,
        router_init_scale = 10.0,
        jitter_noise = 0.01,
        sparsity_lambda = 0.01, 
        router_aux_loss_coeff = 0.02,
        router_grad_clip = 0.05,    
        sparsity_warm_up_steps = 2000,
        use_sigmoid_scaling = True,
        ngpt_sqk_init_value = 1.0,  
        ngpt_sqk_init_scale = 0.03125,
        ngpt_alpha_value_attn = 0.05,
        ngpt_alpha_scale_attn = 0.03125,
        ngpt_alpha_value_mlp = 0.05,
        ngpt_alpha_scale_mlp = 0.03125,
        ngpt_suv_value = 1.0,
        ngpt_suv_scale = 1.0,
        ngpt_sz_init_value = 1.00,
        ngpt_sz_init_scale = 0.03125,
        bias = False,
        use_ckpt = False,
        lr = 15e-4,
        use_exclusive_attention = False,
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
        self.selection_threshold = selection_threshold
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
        self.use_ckpt = use_ckpt
        self.lr = lr
        self.use_exclusive_attention = use_exclusive_attention

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

    # Pass in only Hidden States 
    # Don't pass in attention mask bc theres no attention here (duh)
    def forward(self, hidden_states, step_tensor):


        # Write vars for cleaner code
        q_down_proj = self.q_down_proj
        l_i_weights = self.l_i_weights
        tau = self.tau
        q_up_proj = self.q_up_proj 
        scale = self.scale
        num_elastic_heads = self.num_elastic_heads
        self.selection_threshold = self.config.selection_threshold


        
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
        # Use bmm (batch matrix matric product) b/c [b, n_r_l, s] * [b, s, h_s] (dims don't match up normally)
        # Could've transposed, but this is more memory efficient
        latents = torch.bmm(scanner_softmax, hidden_states)

        # # Normalize the latents (or should we???)
        # latents = justnorm(latents)

        # Scale latents by Learnable important parameters (l_i_weights)
        # Softmax them first
        l_i_weights = F.softmax(l_i_weights, dim = 0)

        # Apply l_i_weights to latents
        # Sum the Latents together
        # Size: ([b, num_router_latents, hidden_size] * broadcast [1, num_router_latents, 1]) and sum the latents = [b, 1 (size of pooled_latents when we added them together), hidden_size]
        pooled_latents = (latents * l_i_weights.view(1, -1, 1)).sum(dim=1, keepdim = True)

        # # Normalize again (per nGPT requirements)
        # # Or should we? We need the weight of these latents so that sigmoid actually does its job
        # # Technically tau tries to help it, but maybe commenting this out makes more sense
        # pooled_latents = justnorm(pooled_latents)
        
        # Normalize q_up_proj      
        # Requires .weight since the matrix was defined before
        # size [num_permanent_heads - num_elastic_heads, hidden_size]
        q_up_proj = justnorm(q_up_proj.weight, dim = 1)

        # Multiply the latents by the classifer (q_up_proj)
        # Call it "class_scores"
        # Size: [b, 1, hidden_size] * [hidden_size, num_permanent_heads - num_elastic_heads] = [b, 1, num_permanent_heads - num_elastic_heads]
        class_scores = F.linear(pooled_latents, q_up_proj)

        # Multiply this by Tau (router_init_scale) and ngpt scaler sqrt(hidden_size), or should we???
        class_scores = class_scores * tau  # * scale

        # Sigmoid Scores
        # Size: still [b, 1, num_permanent_heads - num_elastic_heads], but with sigmoid scores
        sigmoid_scores = torch.sigmoid(class_scores)

        # Apply Router Gradient Clip (to prevent violent gradient updates if a specific head was wrong)
        if self.training and self.config.router_grad_clip > 0.0:
            
            def clip_router_gradients(raw_gradient):
                return torch.clamp(
                    raw_gradient,
                    min = -self.config.router_grad_clip,
                    max = self.config.router_grad_clip
            )

            sigmoid_scores.register_hook(clip_router_gradients)
            


        # NEW SELECTION STRATEGY: #########################
        # To prevent recompiling during top-k to threshold, we will move onto a slightly more refined approach:


        # Apply top_k logic:
        # indices dimension: [batch_size, 1, num_elastic_heads]
        _, indices = sigmoid_scores.topk(num_elastic_heads, dim = -1)
        # topk_mask size sum([b, 1, num_elastic_heads,num_permanent_heads - num_elastic_heads]) of the 3rd dimension = [b, 1 ,num_permanent_heads - num_elastic_heads]
        # Look at how one_hot works: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
        topk_mask = F.one_hot(indices, num_classes=sigmoid_scores.size(-1)).sum(dim=2).float()

        # Also calculate threshold mask:
        # indices dimension: [batch_size, 1, num_elastic_heads] (we just 0'd the sigmoid_scores values that are less then selection_threshold)
        threshold_mask = (sigmoid_scores > self.selection_threshold).float()

        # determine which phase of training this is in
        is_warmup = step_tensor < self.config.sparsity_warm_up_steps

        # Based on is_warmup, either use topk or threshold
        # flat_mask size [b, 1, num_permanent_heads - num_elastic_heads] (choose topk or threshold):
        # This is slightly bad for GPUs (since its a waste to calculation both), but at least the TPU doesn't have to recompile
        flat_mask = torch.where(is_warmup, topk_mask, threshold_mask)
    
        # # ###############################################



        # use_sigmoid_scaling = True: router_mask = Sigmoid values and 0s (Accuracy)
        # use_sigmoid_scaling = False: router_mask = 1s and 0s (Efficiency)
        if self.config.use_sigmoid_scaling:
            # Scale back to sigmoid scores
            flat_mask = flat_mask * sigmoid_scores


        # Apply STE for Dead Router Heads during backprop
        # Must happen after sigmoid_scaling or else torch could believe the sigmoid scaling are dynamically linked
        # .detach() Ignored during Backprop (Autograd doesn't see anything with.detach(), so when backprop happens, they disappear)
        # Forward pass: flat_mask- sigmoid_scores + sigmoid_scores = flat_mask
        # Backward pass: sigmoid_scores
        flat_mask = flat_mask.detach() - sigmoid_scores.detach() + sigmoid_scores

        # Add attention dimensions: [b, 1, num_permanent_heads - num_elastic_heads] -> [b, num_elastic_heads, 1, 1]
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

        # Calculate Training Losses Here:
        if self.training:
            # aux_loss: ensure that the router doesn't route the same head everytime (ignoring permanent heads)
            # Take each sequence in the batch, and average the sigmoid score within each head
            # squeeze = remove all dims = 1
            # size: [b, 1, num_permanent_heads - num_elastic_heads] -> [num_permanent_heads - num_elastic_heads]
            P = sigmoid_scores.mean(dim=0).squeeze()
            # Take each sequence in the batch, and average the sigmoid score only for the ones passing the threshold
            # If for example they all routed
            # size: [num_permanent_heads - num_elastic_heads]
            f = (sigmoid_scores > self.selection_threshold).float().mean(dim=0).squeeze() # Best approx of hard_mask here
            # Dot P*f and multiply by the number of elastic heads. This is the aux_loss
            raw_aux_loss = self.num_elastic_heads * torch.sum(f * P)
            self.aux_loss = self.config.router_aux_loss_coeff * raw_aux_loss
        
            # sparsity_loss: ensure that the model doesn't just turn on all the heads

            # Apply sparsity annealing for warm_up
            sparsity_warm_up_scale = torch.clamp(step_tensor / self.config.sparsity_warm_up_steps, max = 1.0)

            # Multiply set scaling factor (sparsity_lamdba) by the mean of the sigmoid_scores (remember sigmoid scores don't add up to 1)
            self.sparsity_loss = sparsity_warm_up_scale * self.config.sparsity_lambda * sigmoid_scores.mean()

        else:
            self.aux_loss = torch.tensor(0.0, device=hidden_states.device)
            self.sparsity_loss = torch.tensor(0.0, device=hidden_states.device)

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
        self.num_elastic_heads = config.num_elastic_heads
        self.d_head = config.hidden_size // config.num_attention_heads
        self.ngpt_sqk_init_value = config.ngpt_sqk_init_value
        self.ngpt_sqk_init_scale = config.ngpt_sqk_init_scale
        self.config = config

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
        self.sqk = nn.Parameter(self.ngpt_sqk_init_scale*torch.ones(self.hidden_size))

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
            q = sqk.to(q.dtype) * q 
            k = sqk.to(k.dtype) * k 

            # Apply Attention
            # Sclae by sqrt(dk)
            # A whole lot happens here. final size(): [b, num_attention_heads, seq_len, d_head]
            context_layer = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask,
                scale=math.sqrt(self.d_head)
            )

            # Add Exclusive Attention (better results?)
            if (self.config.use_exclusive_attention):
                Vn = torch.nn.functional.normalize(v, dim=-1)
                context_layer = context_layer - (context_layer * Vn).sum(dim=-1, keepdim=True) * Vn

            # # If broadcasting work, apply mask like this
            # # size(): [b, num_attention_heads, seq_len, d_head]
            # context_layer = context_layer * router_mask

            # if router_mask is not None:
            #     if router_mask.shape[1] != context_layer.shape[1]:
            #         # We need to map 'V' views to 'H' heads (e.g., 18 views -> 20 heads)
            #         # We treat the views like a 1D image and "stretch" it to the head count
            #         # [B, V, 1, 1] -> [B, 1, V]
            #         m = router_mask.squeeze(-1).squeeze(-1).unsqueeze(1)
            #         # Interpolate to [B, 1, H]
            #         m = F.interpolate(m, size=context_layer.shape[1], mode='nearest')
            #         # Reshape back to [B, H, 1, 1]
            #         router_mask = m.squeeze(1).unsqueeze(-1).unsqueeze(-1)

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
            q_sliced = sqk_sliced.to(q_sliced.dtype) * q_sliced  
            k_sliced = sqk_sliced.to(k_sliced.dtype) * k_sliced  

            # Flash Attention
            # size(): [b, num_active_heads, seq_len, d_head]
            context_sliced = F.scaled_dot_product_attention(
                q_sliced, k_sliced, v_sliced, 
                attn_mask=attention_mask,
                scale=math.sqrt(self.d_head)
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
        self.attn_alpha = torch.nn.Parameter(self.ngpt_alpha_scale_attn*torch.ones(self.hidden_size))

        # Alpha Eigen Update after MLP (2nd Optimizer Step)
        self.mlp_alpha = torch.nn.Parameter(self.ngpt_alpha_scale_mlp*torch.ones(self.hidden_size))

        # MLP expansion layer
        self.mlp_exp = nn.Linear(
            self.hidden_size, 
            2 * self.intermediate_size, 
            bias = config.bias
        )

        # suv scaling vectors during SwiGLU 
        self.suv = torch.nn.Parameter(self.ngpt_suv_scale*torch.ones(2 * self.intermediate_size))

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
        
        # Apply Normalization to hidden states before and after attention
        # both size(): [b, seq_len, hidden_size]
        A_norm = justnorm(hidden_states)
        B_norm = justnorm(hidden_states_attention)

        # Define the eigen learning rate 
        # alpha >=0
        # size(): [hidden_size]
        lr = self.attn_alpha * (ngpt_alpha_value_attn / ngpt_alpha_scale_attn)
        lr = torch.abs(lr).to(A_norm.dtype)
            
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

        # Apply Normalization to hidden states after attention and after mlp
        # both size(): [b, seq_len, hidden_size]
        A_norm = justnorm(hidden_states_opt1)
        B_norm = justnorm(h_mlp)

        # Define the eigen learning rate 
        # alpha >=0
        # size(): [hidden_size]
        lr = self.mlp_alpha * (ngpt_alpha_value_mlp / ngpt_alpha_scale_mlp) 
        lr = torch.abs(lr).to(A_norm.dtype)

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
    def forward(self, hidden_states, attention_mask, step_tensor):
        router_mask = self.mlt_vw_rtr(hidden_states, step_tensor)
        aux_loss = self.mlt_vw_rtr.aux_loss
        sparsity_loss = self.mlt_vw_rtr.sparsity_loss
        attn_output = self.attn(hidden_states, attention_mask, router_mask)
        layer_output = self.mlp(hidden_states, attn_output)
        return layer_output, aux_loss, sparsity_loss



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
        
    
    # Forward Pass
    def forward(self, input_ids, attention_mask, current_step = None):


        # Reshape Attention Mask to be 4D for SDPA [batch_size, 1, 1, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()

        # Then make attend = 0 and mask = -inf to allow for Flash Attention compatitbility
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
        hidden_states = embeddings 

        # Accumulate aux_loss and sparsity_loss
        total_aux_loss = 0
        total_sparsity_loss = 0

        # Run Tranformer Blocks
        for block in self.blocks:
            # Use Gradient Checkpointing
            if self.use_ckpt and self.training:
                hidden_states, aux_loss, sparsity_loss = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states, 
                    attention_mask, 
                    step_tensor,
                    use_reentrant=False
                )
            # Or Standard Forward Pass
            else:
                hidden_states, aux_loss, sparsity_loss = block(hidden_states, attention_mask, step_tensor)
                
            total_aux_loss += aux_loss
            total_sparsity_loss += sparsity_loss

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
        self.sz = nn.Parameter(torch.ones(config.vocab_size))

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
    def normalize_ngpt_matrices(self):
        
        # Define all the projection matrices to normalize
        keys_to_normalize = (
            "word_embeddings.weight",
            "classifier.weight",
            "attn.qkv.weight",
            "attn.output.weight",
            "mlp.mlp_exp.weight",
            "mlp.mlp_proj.weight",
            "mlt_vw_rtr.q_down_proj.weight"
            # ,"mlt_vw_rtr.q_up_proj.weight" # Remove b/c messing up sigmoid scores
        )

        # Normalize every one of those mats along their dim = 1 (embedding)
        # The model's weights are transposed (for backprop) so instead of dim = 0, we do dim = 1
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith(keys_to_normalize):
                    param.data = justnorm(param.data, dim = 1, eps = 1e-12)

    
    # Forward pass
    def forward(self, input_ids, attention_mask, current_step = None):

        # Gather Context from the model
        # features: [b, seq_len, hidden_size]
        features, total_aux_loss, total_sparsity_loss = self.model(input_ids, attention_mask, current_step)

        # Scale / prepare sz
        sz = self.sz * (self.ngpt_sz_init_value / self.ngpt_sz_init_scale)

        # project features onto classifer
        # [b, seq_len, hidden_size] * [hidden_size, vocab_size] = [b, seq_len, vocab_size]
        unscaled_logits = self.classifier(features)

        # Scale the logits with sz
        logits = sz.to(unscaled_logits.dtype) * unscaled_logits

        # Return Logits
        return logits, total_aux_loss, total_sparsity_loss