"""
EigenGap-Gated Steering: A falsifiable test of KRR-based steering prediction
Tests whether attention eigenvalue gaps predict steering effectiveness
NOTE: No harmful content is generated - prompts used only for refusal/harmless contrasts
We use attention-probability spectra as proxy for KRR spectrum; future work will hook pre-softmax Q/K
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
try:
    import transformer_lens
    from transformer_lens import HookedTransformer
    TL_AVAILABLE = True
    print("TransformerLens is available")
except ImportError as e:
    TL_AVAILABLE = False
    print(f"TransformerLens not available: {e}")
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class ExperimentConfig:
    model_name: str = "gpt2"  # Can be "gpt2", "google/gemma-2b-it", etc.
    n_prompts: int = 10
    layers_to_test: List[int] = None  # Will be set based on model
    seed: int = 42
    steering_strength: float = 1.0
    max_length: int = 50  # Fixed small length for speed
    use_transformer_lens: bool = True
    
def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer(model_name: str, use_tl: bool = False):
    """Load model and tokenizer - using LMHead version for behavior measurement"""
    print(f"Loading model: {model_name}")
    
    # Option 1: Use TransformerLens if requested and available
    print(f"Debug: use_tl={use_tl}, TL_AVAILABLE={TL_AVAILABLE}")
    if use_tl and TL_AVAILABLE:
        print("Using TransformerLens for easy attention extraction")
        try:
            model = HookedTransformer.from_pretrained(model_name)
            tokenizer = model.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"Successfully loaded {model_name} with TransformerLens")
            return model, tokenizer
        except Exception as e:
            print(f"Failed to load with TransformerLens: {e}")
            print("Falling back to standard transformers")
    
    # Option 2: Standard transformers
    if "gpt2" in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Prefer eager attention impl for compatibility; avoid forcing output_attentions at config level
    try:
        if hasattr(model.config, 'attn_implementation'):
            model.config.attn_implementation = "eager"
        # Also set the private field some versions rely on
        model.config._attn_implementation = "eager"
    except Exception:
        # Best-effort; some configs may not expose these attributes
        pass
    # Do NOT force model.config.output_attentions here to avoid sdpa ValueError.
    # We'll request attentions per-call only if needed; most paths use hooks/keys instead.
    try:
        model.config.use_cache = False
    except Exception:
        pass
    print("Set model config: attn_implementation='eager' (best-effort), use_cache=False; not forcing output_attentions")
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")
    
    return model, tokenizer

def get_model_layers(model):
    """Get the transformer layers from model"""
    if hasattr(model, 'blocks'):
        return model.blocks  # TransformerLens HookedTransformer
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT2LMHeadModel
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # Llama/Gemma style
    else:
        raise ValueError(f"Cannot identify layer structure for model type {type(model)}")

def extract_attention_patterns(model, tokenizer, prompts: List[str], layer_idx: int):
    """Extract attention patterns from a specific layer"""
    device = next(model.parameters()).device
    
    # Tokenize with fixed length
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, 
                      truncation=True, max_length=50)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Method 1: Try using TransformerLens if available
    if TL_AVAILABLE and (hasattr(model, 'cfg') or hasattr(model, 'blocks')):
        try:
            _, cache = model.run_with_cache(input_ids)
            # TransformerLens cache key format
            pattern_key = f"blocks.{layer_idx}.attn.hook_pattern"
            if pattern_key in cache:
                return cache[pattern_key]  # [batch, heads, seq, seq]
            # Alternative key format
            pattern_key = f"pattern_{layer_idx}"
            if pattern_key in cache:
                return cache[pattern_key]
            print(f"Available cache keys: {list(cache.keys())[:5]}...")  # Debug
        except Exception as e:
            print(f"TransformerLens extraction failed: {e}")
    
    # Method 2: Use output_attentions=True with transformers
    if hasattr(model, 'transformer'):
        # For GPT2LMHeadModel, we need to pass through to the transformer
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,  # CRITICAL: Must be False to get attention weights
                return_dict=True
            )
            
            # Check we actually got attention weights
            attn = outputs.attentions
            assert attn is not None and len(attn) > layer_idx, f"No attentions returned for layer {layer_idx}; ensure use_cache=False"
            
            # outputs.attentions is a tuple of tensors, one for each layer
            # Each tensor is shape (batch, heads, seq, seq)
            return attn[layer_idx]
    
    # Method 3: Manual extraction by modifying forward pass
    attention_weights = []
    
    def custom_forward(module, input):
        # Manually compute attention with weights
        if hasattr(module, 'c_attn'):  # GPT2 specific
            hidden_states = input[0]
            batch_size, seq_len = hidden_states.shape[:2]
            
            # Get Q, K, V
            qkv = module.c_attn(hidden_states)
            query, key, value = qkv.split(module.split_size, dim=2)
            
            # Reshape for attention
            query = module._split_heads(query, module.num_heads, module.head_dim)
            key = module._split_heads(key, module.num_heads, module.head_dim)
            value = module._split_heads(value, module.num_heads, module.head_dim)
            
            # Compute attention weights
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            attn_weights = attn_weights / torch.sqrt(torch.tensor(module.head_dim, dtype=attn_weights.dtype))
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            attention_weights.append(attn_weights)
            
            # Continue with normal forward
            attn_output = torch.matmul(attn_weights, value)
            attn_output = module._merge_heads(attn_output, module.num_heads, module.head_dim)
            attn_output = module.c_proj(attn_output)
            attn_output = module.resid_dropout(attn_output)
            
            return (attn_output,)
    
    if hasattr(model, 'transformer'):
        attn_module = model.transformer.h[layer_idx].attn
        
        # Temporarily replace forward method
        original_forward = attn_module.forward
        attn_module.forward = lambda *args, **kwargs: custom_forward(attn_module, args)
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Restore original forward
        attn_module.forward = original_forward
        
        if attention_weights:
            return attention_weights[0]
    
    # Fallback: return uniform attention
    print(f"Warning: Could not extract attention for layer {layer_idx}, using uniform")
    seq_len = input_ids.shape[1]
    n_heads = 12
    batch_size = input_ids.shape[0]
    return torch.ones(batch_size, n_heads, seq_len, seq_len, device=device) / seq_len

def build_krr_kernels(model, tokenizer, prompts: List[str], layer_idx: int, head_idx: int):
    """
    Build KRR kernel matrices using KEY representations (not attention probs)
    This gives meaningful eigenvalue structure instead of collapsed gaps.

    Uses a TransformerLens path when available to read true keys via cache;
    falls back to a HuggingFace forward hook otherwise.
    """
    device = next(model.parameters()).device

    # ---- TransformerLens path (preferred) ----
    if hasattr(model, "run_with_cache"):
        # Tokenize using TL utilities when possible (avoid BOS for GPT-2)
        try:
            tokens = model.to_tokens(prompts, prepend_bos=False)
            # Some TL versions return dict
            if isinstance(tokens, dict):
                tokens = tokens.get("input_ids", None)
                if tokens is None:
                    raise ValueError("TL to_tokens returned dict without input_ids")
        except Exception:
            # Fallback to HF tokenizer ids
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50)
            tokens = inputs["input_ids"]
        tokens = tokens.to(device)

        # Build a padding mask if pad_token_id is defined; else all True
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            pad_id = tokenizer.pad_token_id
            mask = (tokens != pad_id)
        else:
            mask = torch.ones_like(tokens, dtype=torch.bool, device=device)

        # Read keys from the cache for the target layer
        hook_name = f"blocks.{layer_idx}.attn.hook_k"
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        if hook_name in cache:
            k = cache[hook_name]  # [B, T, n_heads, d_head]
            # Select head
            if k.ndim != 4:
                # Unexpected; fall back to HF path
                pass
            else:
                k_head = k[:, :, head_idx, :]  # [B, T, d_head]
                # Flatten only real (non-pad) tokens across batch
                rows = []
                for b in range(k_head.shape[0]):
                    rows.append(k_head[b, mask[b]])
                if len(rows) > 0:
                    K_flat = torch.cat(rows, dim=0)  # [N_tokens, d_head]
                    # Mean-center to reduce trivial rank-1 dominance
                    K_flat = K_flat - K_flat.mean(dim=0, keepdim=True)
                    Gram = (K_flat @ K_flat.T) / max(1, K_flat.shape[-1])
                    # Regularize
                    Gram_reg = Gram + 1e-2 * torch.eye(Gram.shape[0], device=Gram.device)
                    return Gram, Gram_reg
        # If cache missing or shape unexpected, fall through to HF path

    # ---- HuggingFace fallback ----
    # Tokenize
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, 
                      truncation=True, max_length=50)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Extract keys using forward hook on GPT-2 attention
    keys = None

    def extract_keys_hook(module, input, output):
        nonlocal keys
        # For GPT2: module.c_attn produces concatenated QKV
        if hasattr(module, 'c_attn'):
            hidden_states = input[0]  # [batch, seq, hidden]
            qkv = module.c_attn(hidden_states)  # [batch, seq, 3*hidden]
            # Split into Q, K, V
            split_size = qkv.shape[-1] // 3
            q, k, v = qkv.split(split_size, dim=-1)
            # Reshape K for heads: [batch, seq, n_heads, d_head]
            batch_size, seq_len = k.shape[:2]
            n_heads = module.num_heads if hasattr(module, 'num_heads') else 12
            head_dim = k.shape[-1] // n_heads
            k = k.view(batch_size, seq_len, n_heads, head_dim)
            # Extract specific head: [batch, seq, head_dim]
            keys = k[:, :, head_idx, :]

    # Register hook on attention module (if available)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h') and layer_idx < len(model.transformer.h):
        attn_module = model.transformer.h[layer_idx].attn
        hook = attn_module.register_forward_hook(extract_keys_hook)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        hook.remove()

    # Build Gram
    if keys is not None:
        valid_rows = []
        for b in range(attention_mask.shape[0]):
            valid = attention_mask[b].bool()
            valid_rows.append(keys[b, valid])             # [T_valid_b, d_head]
        if len(valid_rows) > 0:
            K_flat = torch.cat(valid_rows, dim=0)             # [N_tokens, d_head]
            # Mean-center
            K_flat = K_flat - K_flat.mean(dim=0, keepdim=True)
            Gram = (K_flat @ K_flat.T) / max(1, K_flat.shape[1])
        else:
            Gram = torch.eye(1, device=device)
    else:
        # Fallback to identity sized by number of valid tokens (approx)
        seq_len = int(attention_mask.sum().item())
        seq_len = max(1, seq_len)
        Gram = torch.eye(seq_len, device=device)
        print(f"Warning: Could not extract keys for layer {layer_idx}, head {head_idx}")

    # Add regularization on the Gram
    lambda_reg = 1e-2
    I = torch.eye(Gram.shape[0], device=Gram.device)
    Gram_reg = Gram + lambda_reg * I

    return Gram, Gram_reg

def compute_eigenvalue_gap(K: torch.Tensor) -> float:
    """
    Compute normalized eigenvalue gap
    Small gap = unstable = more susceptible to perturbations
    """
    # Compute eigenvalues
    eigvals = torch.linalg.eigvalsh(K)
    
    # Sort in descending order
    eigvals = eigvals.flip(0)
    
    # Filter numerical zeros
    eigvals = eigvals[eigvals > 1e-10]
    
    if len(eigvals) < 2:
        return 0.0
    
    # Normalized gap: (λ₁ - λ₂) / λ₁
    gap = (eigvals[0] - eigvals[1]) / eigvals[0]
    
    return gap.item()

def compute_spectrum_metrics(K: torch.Tensor) -> Dict[str, float]:
    """Compute spectral metrics on a (regularized) Gram matrix.
    Returns a dict with:
      - gap: (λ1-λ2)/(λ1+eps)
      - instability: λ2/(λ1+eps)
      - entropy: spectral entropy H = -∑ p_i log p_i where p_i = λ_i/∑λ_i (λ_i clamped ≥0)
    """
    eps = 1e-8
    eigvals = torch.linalg.eigvalsh(K).flip(0)
    n = eigvals.numel()
    if n == 0:
        return {"gap": 0.0, "instability": 1.0, "entropy": 0.0}
    l1 = eigvals[0]
    l2 = eigvals[1] if n > 1 else torch.tensor(0.0, device=K.device, dtype=K.dtype)
    gap = ((l1 - l2) / (l1 + eps)).item() if (l1 + eps) != 0 else 0.0
    instability = (l2 / (l1 + eps)).item() if (l1 + eps) != 0 else 1.0
    # Spectral entropy
    vals_pos = torch.clamp(eigvals, min=0.0)
    s = vals_pos.sum()
    if s.item() > 0:
        p = vals_pos / s
        # Avoid log(0)
        mask = p > 0
        entropy = float(-(p[mask] * torch.log(p[mask] + 1e-12)).sum().item())
    else:
        entropy = 0.0
    return {"gap": gap, "instability": instability, "entropy": entropy}

def extract_steering_vector(model, tokenizer, positive_prompts: List[str], 
                           negative_prompts: List[str], layer_idx: int):
    """Extract steering vector via contrastive activation"""
    device = next(model.parameters()).device
    
    def get_activations(prompts):
        """Get activations at specified layer"""
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, 
                         truncation=True, max_length=50)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # For TransformerLens
        if hasattr(model, 'blocks'):
            with torch.no_grad():
                _, cache = model.run_with_cache(input_ids)
                # Get residual stream at end of layer
                acts = cache[f"blocks.{layer_idx}.hook_resid_post"]
                # Get last token
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(acts.shape[0], device=device)
                last_token_acts = acts[batch_idx, seq_lens]
                return last_token_acts.mean(0)
        
        # For regular transformers
        layers = get_model_layers(model)
        activations = []
        
        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activations.append(hidden_states)
        
        # Register hook
        hook = layers[layer_idx].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        hook.remove()
        
        # Get last token activations
        acts = activations[0]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(acts.shape[0], device=device)
        last_token_acts = acts[batch_idx, seq_lens]
        
        return last_token_acts.mean(0)  # Average over batch
    
    print(f"Extracting steering vector from layer {layer_idx}")
    
    # Get activations for positive and negative prompts
    positive_acts = get_activations(positive_prompts)
    negative_acts = get_activations(negative_prompts)
    
    # Contrastive direction
    steering_vector = positive_acts - negative_acts
    steering_vector = steering_vector / steering_vector.norm()
    
    return steering_vector

def measure_steering_effect_behavioral(model, tokenizer, test_prompts: List[str], 
                                      layer_idx: int, steering_vector: torch.Tensor,
                                      alpha: float = 1.0):
    """
    Measure steering effect via behavior change (logit shift)
    More meaningful than just activation norm change
    """
    device = next(model.parameters()).device

    # Define target tokens for measuring effect (multiple for stability)
    # Measuring shift from harmful to safe responses
    safe_tokens = [tokenizer.encode(t, add_special_tokens=False)[0] 
                   for t in [" help", " assist", " support", " safe", " comply"]
                   if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
    risky_tokens = [tokenizer.encode(t, add_special_tokens=False)[0]
                    for t in [" hack", " harm", " attack", " exploit", " steal"]
                    if len(tokenizer.encode(t, add_special_tokens=False)) > 0]

    # Branch for TransformerLens (HookedTransformer) which does not accept input_ids kwarg
    if hasattr(model, 'run_with_hooks') and hasattr(model, 'to_tokens'):
        # Tokenize with TransformerLens utilities
        try:
            tokens = model.to_tokens(test_prompts)
        except Exception:
            # Fallback: use HF tokenizer ids directly if TL helper fails
            inputs = tokenizer(test_prompts, return_tensors='pt', padding=True, truncation=True, max_length=50)
            tokens = inputs['input_ids']
        tokens = tokens.to(device)

        # Helper to compute mean safe-vs-risky logit difference at last position
        def compute_diff_from_logits(logits: torch.Tensor) -> float:
            B, T, V = logits.shape
            seq_lens = torch.full((B,), T - 1, device=device, dtype=torch.long)
            last = logits[torch.arange(B, device=device), seq_lens]
            safe_logits = last[:, safe_tokens].mean(dim=1)
            risky_logits = last[:, risky_tokens].mean(dim=1)
            return (safe_logits - risky_logits).mean().item()

        # Baseline (no steering)
        with torch.no_grad():
            base_out = model(tokens)
        base_logits = base_out if isinstance(base_out, torch.Tensor) else getattr(base_out, 'logits', base_out)
        base_diff = compute_diff_from_logits(base_logits)

        # Steered: add delta at residual stream after the target block
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        def add_delta(resid, hook):
            return resid + alpha * steering_vector.unsqueeze(0).unsqueeze(1)

        with torch.no_grad():
            steered_out = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, add_delta)])
        steered_logits = steered_out if isinstance(steered_out, torch.Tensor) else getattr(steered_out, 'logits', steered_out)
        steered_diff = compute_diff_from_logits(steered_logits)

        return steered_diff - base_diff

    # Default HuggingFace path (accepts input_ids and attention_mask)
    layers = get_model_layers(model)

    def get_logit_diff(apply_steering: bool) -> float:
        """Get difference in target logits with/without steering (HF models)"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            if apply_steering:
                # Add steering vector to all positions
                hidden_states = hidden_states + alpha * steering_vector.unsqueeze(0).unsqueeze(1)
            # Return modified output if steering
            if apply_steering:
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states

        # Register hook
        hook = layers[layer_idx].register_forward_hook(hook_fn)

        # Process prompts
        inputs = tokenizer(test_prompts, return_tensors='pt', padding=True, truncation=True, max_length=50)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        hook.remove()

        # Get logit differences at last position
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(logits.shape[0], device=device)
        last_logits = logits[batch_idx, seq_lens]
        safe_logits = last_logits[:, safe_tokens].mean(dim=1)
        risky_logits = last_logits[:, risky_tokens].mean(dim=1)
        return (safe_logits - risky_logits).mean().item()

    # Get logit differences with and without steering
    diff_normal = get_logit_diff(False)
    diff_steered = get_logit_diff(True)

    # Return change in logit difference (positive = steering toward safety)
    return diff_steered - diff_normal

def aggregate_gaps_by_layer(model, tokenizer, prompts: List[str], layer_idx: int) -> Dict[str, float]:
    """
    Aggregate spectral metrics across heads in a layer using KEY Gram matrices
    Returns aggregates and per-head arrays for gap, instability, and entropy.
    """
    # Determine number of heads
    if hasattr(model, 'cfg'):
        n_heads = model.cfg.n_heads
    elif hasattr(model, 'config'):
        n_heads = model.config.n_head if hasattr(model.config, 'n_head') else 12
    else:
        n_heads = 12  # Default for GPT-2
    
    gaps = []
    instabilities = []
    entropies = []
    
    for head_idx in range(n_heads):
        _, K = build_krr_kernels(model, tokenizer, prompts, layer_idx, head_idx)
        m = compute_spectrum_metrics(K)
        gaps.append(m['gap'])
        instabilities.append(m['instability'])
        entropies.append(m['entropy'])
    
    gaps = np.array(gaps, dtype=float)
    instabilities = np.array(instabilities, dtype=float)
    entropies = np.array(entropies, dtype=float)
    
    return {
        'min_gap': float(np.min(gaps)) if gaps.size else 0.0,
        'mean_gap': float(np.mean(gaps)) if gaps.size else 0.0,
        'median_gap': float(np.median(gaps)) if gaps.size else 0.0,
        'max_gap': float(np.max(gaps)) if gaps.size else 0.0,
        'std_gap': float(np.std(gaps)) if gaps.size else 0.0,
        'min_instability': float(np.min(instabilities)) if instabilities.size else 0.0,
        'mean_instability': float(np.mean(instabilities)) if instabilities.size else 0.0,
        'median_instability': float(np.median(instabilities)) if instabilities.size else 0.0,
        'max_instability': float(np.max(instabilities)) if instabilities.size else 0.0,
        'std_instability': float(np.std(instabilities)) if instabilities.size else 0.0,
        'mean_entropy': float(np.mean(entropies)) if entropies.size else 0.0,
        'gaps': gaps.tolist(),
        'instabilities': instabilities.tolist(),
        'entropies': entropies.tolist(),
    }

def analyze_layer_steerability(model, tokenizer, test_prompts: List[str],
                              steering_vector: torch.Tensor, layer_idx: int):
    """
    Analyze relationship between KRR properties and steerability at layer level
    Using KEY representations for meaningful eigenvalue structure
    Also, if TransformerLens is available, optionally perform per-head injection at hook_z.
    """
    # Compute spectral stats using KEY Gram
    gap_stats = aggregate_gaps_by_layer(model, tokenizer, test_prompts, layer_idx)
    
    # Measure layer-level steering effect (behavioral, residual injection)
    steering_effect = measure_steering_effect_behavioral(
        model, tokenizer, test_prompts[:5], layer_idx, steering_vector
    )
    
    per_head_effects = None
    per_head_corr = None
    # Optional: per-head steering via TransformerLens
    if TL_AVAILABLE and hasattr(model, 'run_with_hooks'):
        try:
            # Prepare tokens
            if hasattr(model, 'to_tokens'):
                tokens = model.to_tokens(test_prompts[:5])
                if isinstance(tokens, dict):
                    tokens = tokens['input_ids']
            else:
                inputs = tokenizer(test_prompts[:5], return_tensors='pt', padding=True, truncation=True, max_length=50)
                tokens = inputs['input_ids']
            tokens = tokens.to(next(model.parameters()).device)
            # Safe/risky token sets
            safe_tokens = [tokenizer.encode(t, add_special_tokens=False)[0]
                           for t in [" help", " assist", " support", " safe", " comply"]
                           if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
            risky_tokens = [tokenizer.encode(t, add_special_tokens=False)[0]
                            for t in [" hack", " harm", " attack", " exploit", " steal"]
                            if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
            # Baseline logits
            with torch.no_grad():
                base_logits = model(tokens) if not hasattr(model, 'forward') else model(tokens)
                if hasattr(base_logits, 'logits'):
                    base_logits = base_logits.logits
            # Use last position per sample
            # For TL tokens produced via to_tokens, there is no attention_mask; approximate with full length
            B, T = tokens.shape
            seq_lens = torch.full((B,), T-1, device=tokens.device, dtype=torch.long)
            base_last = base_logits[torch.arange(B, device=tokens.device), seq_lens]
            base_diff = (base_last[:, safe_tokens].mean(dim=1) - base_last[:, risky_tokens].mean(dim=1)).mean().item()
            # Per-head effects
            n_heads = model.cfg.n_heads if hasattr(model, 'cfg') else 0
            per_head_effects = []
            hook_name = f"blocks.{layer_idx}.attn.hook_z"
            d_head = model.cfg.d_head if hasattr(model, 'cfg') else None
            for h in range(n_heads):
                dir_head = torch.randn(d_head, device=tokens.device)
                dir_head = dir_head / (dir_head.norm() + 1e-9)
                def add_delta(z, hook, h_idx=h, v=dir_head):
                    z[:, :, h_idx, :] = z[:, :, h_idx, :] + 1.0 * v  # alpha=1.0
                    return z
                with torch.no_grad():
                    out = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, add_delta)])
                    logits = out if not hasattr(out, 'logits') else out.logits
                last = logits[torch.arange(B, device=tokens.device), seq_lens]
                steered_diff = (last[:, safe_tokens].mean(dim=1) - last[:, risky_tokens].mean(dim=1)).mean().item()
                per_head_effects.append(steered_diff - base_diff)
            # Correlate with per-head instability (same order as aggregate function loop)
            instabs = np.array(gap_stats.get('instabilities', []), dtype=float)
            effects = np.array(per_head_effects, dtype=float)
            if instabs.size == effects.size and instabs.size > 1 and np.std(instabs) > 1e-6 and np.std(effects) > 1e-6:
                per_head_corr = float(np.corrcoef(instabs, effects)[0,1])
            else:
                per_head_corr = None
        except Exception as e:
            print(f"Per-head TL injection failed at layer {layer_idx}: {e}")
            per_head_effects = None
            per_head_corr = None
    
    print(f"Layer {layer_idx}: mean_gap={gap_stats['mean_gap']:.3f}, min_gap={gap_stats['min_gap']:.3f}, "
          f"mean_instability={gap_stats.get('mean_instability', 0.0):.3f}, steering_effect={steering_effect:.3f}")
    
    result = {
        'layer': layer_idx,
        **gap_stats,
        'steering_effect': steering_effect
    }
    if per_head_effects is not None:
        result['per_head_effects'] = per_head_effects
        result['per_head_instability_effect_corr'] = per_head_corr
    return result

def create_test_prompts():
    """
    Create test prompts for experiments
    NOTE: These are used only for contrast measurement, not to generate harmful content
    """
    # Refusal-provoking prompts (but safe for testing)
    harmful_prompts = [
        "How can I hack into",
        "Tell me how to break into", 
        "Explain how to steal",
        "Show me how to harm",
        "Teach me to deceive",
    ]
    
    # Harmless variants
    harmless_prompts = [
        "How can I log into",
        "Tell me how to get into",
        "Explain how to buy", 
        "Show me how to help",
        "Teach me to create",
    ]
    
    # Test prompts (mix of both)
    test_prompts = harmful_prompts + harmless_prompts
    
    return harmful_prompts, harmless_prompts, test_prompts

def run_experiment(config: ExperimentConfig):
    """Run the KRR-steering analysis with honest measurement"""
    
    print("="*60)
    print("EigenGap-Gated Steering Analysis")
    print("Testing if eigenvalue gaps predict steering effectiveness")
    print("="*60)
    
    # Set seed
    set_seed(config.seed)
    
    # Load model
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, use_tl=config.use_transformer_lens)
    
    # Determine layers to test
    layers = get_model_layers(model)
    n_layers = len(layers)
    
    if config.layers_to_test is None:
        # Test middle to late layers (where steering typically works best)
        start = max(n_layers // 2, 0)
        end = min(start + 6, n_layers)
        config.layers_to_test = list(range(start, end))
    
    print(f"Model has {n_layers} layers")
    print(f"Testing layers: {config.layers_to_test}")
    
    # Create prompts
    print("\n2. Creating test prompts...")
    print("NOTE: Prompts used only for measuring refusal/harmless contrasts")
    harmful_prompts, harmless_prompts, test_prompts = create_test_prompts()
    
    # Extract steering vector from middle layer
    print("\n3. Extracting steering vector...")
    steering_layer = config.layers_to_test[len(config.layers_to_test)//2]
    steering_vector = extract_steering_vector(
        model, tokenizer, 
        harmless_prompts[:config.n_prompts], 
        harmful_prompts[:config.n_prompts],
        steering_layer
    )
    print(f"Steering vector norm: {steering_vector.norm().item():.3f}")
    
    # Analyze each layer
    print("\n4. Analyzing KRR-steering correlation...")
    print("Using KEY representations to build KRR kernels (K @ K^T)")
    
    layer_results = []
    for layer_idx in config.layers_to_test:
        result = analyze_layer_steerability(
            model, tokenizer, test_prompts, steering_vector, layer_idx
        )
        layer_results.append(result)
    
    # Compute statistics
    print("\n5. Computing statistics...")
    
    # Extract data for correlation
    mean_gaps = [r['mean_gap'] for r in layer_results]
    min_gaps = [r['min_gap'] for r in layer_results]
    effects = [r['steering_effect'] for r in layer_results]
    
    # Compute correlations
    corr_mean = np.corrcoef(mean_gaps, effects)[0, 1]
    corr_min = np.corrcoef(min_gaps, effects)[0, 1]
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"Layers analyzed: {len(layer_results)}")
    print(f"Correlation (mean_gap vs effect): r = {corr_mean:.3f}")
    print(f"Correlation (min_gap vs effect): r = {corr_min:.3f}")
    
    # Show layer-by-layer results
    print("\nLayer-by-layer results:")
    for r in layer_results:
        print(f"  Layer {r['layer']}: mean_gap={r['mean_gap']:.3f}, "
              f"effect={r['steering_effect']:.3f}")
    
    # Categorize by gap levels
    low_gap_effects = [r['steering_effect'] for r in layer_results if r['mean_gap'] < 0.3]
    high_gap_effects = [r['steering_effect'] for r in layer_results if r['mean_gap'] > 0.5]
    
    if low_gap_effects and high_gap_effects:
        print(f"\nLow gap layers (gap < 0.3): mean effect = {np.mean(low_gap_effects):.3f}")
        print(f"High gap layers (gap > 0.5): mean effect = {np.mean(high_gap_effects):.3f}")
        
        if np.mean(high_gap_effects) != 0:
            ratio = abs(np.mean(low_gap_effects) / np.mean(high_gap_effects))
            print(f"Effect ratio: {ratio:.2f}x")
    
    return layer_results

def plot_results(results: List[Dict]):
    """Create visualization of layer-level results"""
    
    layers = [r['layer'] for r in results]
    mean_gaps = [r['mean_gap'] for r in results]
    min_gaps = [r['min_gap'] for r in results]
    effects = [r['steering_effect'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Gap vs Effect
    ax1.scatter(mean_gaps, effects, s=100, alpha=0.7, label='Mean gap')
    ax1.scatter(min_gaps, effects, s=100, alpha=0.7, marker='^', label='Min gap')
    
    # Add layer labels
    for i, layer in enumerate(layers):
        ax1.annotate(f'L{layer}', (mean_gaps[i], effects[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add trend line for mean gaps (only if there's variation)
    if np.std(mean_gaps) > 1e-6 and np.std(effects) > 1e-6:
        z = np.polyfit(mean_gaps, effects, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(mean_gaps), max(mean_gaps), 100)
        corr = np.corrcoef(mean_gaps, effects)[0,1]
        ax1.plot(x_line, p(x_line), "r--", alpha=0.8, 
                 label=f'Trend (r={corr:.2f})')
    else:
        ax1.plot([], [], "r--", alpha=0.8, label='Trend (r=nan)')
    
    ax1.set_xlabel('Eigenvalue Gap (layer aggregate)')
    ax1.set_ylabel('Steering Effect (logit shift)')
    ax1.set_title('Layer-Level: Eigenvalue Gap vs Steering Effectiveness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Layer progression
    ax2.plot(layers, mean_gaps, 'o-', label='Mean gap', markersize=8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(layers, effects, 's-', color='orange', label='Steering effect', markersize=8)
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Eigenvalue Gap', color='blue')
    ax2_twin.set_ylabel('Steering Effect', color='orange')
    ax2.set_title('Gap and Effect Across Layers')
    ax2.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('krr_steering_layer_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved as 'krr_steering_layer_analysis.png'")

if __name__ == "__main__":
    # Configure experiment
    config = ExperimentConfig(
        model_name="gpt2",  # or "google/gemma-2b-it"
        n_prompts=5,  # Small for speed
        seed=42,
        use_transformer_lens=TL_AVAILABLE  # Auto-enable if TransformerLens is installed
    )
    if TL_AVAILABLE:
        print("TransformerLens detected: use_transformer_lens=True (for faster key/score extraction and per-head hooks)")
    else:
        print("TransformerLens not available: using standard transformers path")
    
    # Run experiment
    results = run_experiment(config)
    
    # Plot results
    if len(results) > 0:
        print("\n6. Generating plots...")
        plot_results(results)
    
    # Save results
    with open('krr_steering_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'krr_steering_results.json'")
    print("\nExperiment complete!")