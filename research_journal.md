# Research Journal — autoresearch/mar17-arch

## Architecture Canvas

| Component | Current | Tested Variants (result) | Untested Ideas |
|-----------|---------|-------------------------|----------------|
| Attention | PoPE + all-global (LLLL) | sliding window SSSL (worse), HEAD_DIM=64 4-heads (worse) | differential attention, multi-head latent attention, shared QK |
| MLP/FFN | SwiGLU 8/3x (hidden=704) | squared ReLU (marginal loss), parallel attn+MLP (worse) | MoE, wider MLP, deeper MLP |
| Normalization | RMSNorm (custom `norm()`) | sandwich norm (marginal) | QK-norm, deep norm scaling |
| Optimizer | NorMuon+Nesterov+AdamW | Polar Express coeffs (tiny win), cautious WD (win), momentum ramp (worse) | SOAP, schedule-free AdamW, Shampoo |
| Residual | lambda scaling + x0 residual | — | pre-norm variants, fixup init, deeper residual |
| Embedding | token embed + value embeds (alternating) | VE all layers (worse), weight tying (crash) | factored embeddings, shared VE |
| Positional | PoPE frequencies | HEAD_DIM=64 (hurts PoPE) | ALiBi, NoPE, RoPE, learned positions |
| Depth/Width | 4 layers, 256d, 11.3M params | n_embd=384 (worse), depth=6 AR=42 (worse - VE overhead) | depth=5 no-VE, recycling layers |
| Capacity tricks | — | z-loss (redundant), batch 2^13 (worse) | weight sharing/recycling layers, distillation |

## Constraint Analysis

- **Params**: 11.3M of 15M budget (3.7M headroom)
- **Steps**: 300 fixed, ~400ms/step → ~2min training + compile/eval overhead
- **Memory**: 5.6GB of ~24GB available (huge headroom)
- **Primary bottleneck**: Capacity — 11.3M params, 4 layers, 256d is tiny. Architecture efficiency matters more than raw size.
- **Secondary bottleneck**: Convergence — only 300 steps means we need fast-converging architectures

## Experiment-Type Bandit

| Category | Tried | Kept | Success Rate | Notes |
|----------|-------|------|-------------|-------|
| architecture | 9 | 4 | 44% | all-global attn, cautious WD, Polar Express, depth=6 failed |
| optimizer | 4 | 0 | 0% | momentum ramp, no warmup, matrix LR, emb LR combo |
| hyperparameter | 7 | 0 | 0% | batch size, warmdown, LR tweaks |

## Experiment Log

### Experiment 20: depth=6 AR=42 (DISCARD)
**Hypothesis:** More layers (6 vs 4) at same model_dim=256 would improve quality.
**Result:** val_bpb=1.578 (vs 1.379 best). 14.9M params — nearly all budget used.
**Interpretation:** Adding depth also adds VE layers (3 vs 2), each costing ~2.1M params. The extra VE overhead + slow convergence at 300 steps made this much worse. Deeper models need either: no VE on extra layers, or more steps.
**Failure chain:** QUALITY_LOSS → try depth=5 without VE, or recycling existing 4 layers

### Pattern Summary (after experiment 20)
- Architecture changes that reduce overhead while maintaining quality are winning
- The embedding/VE bottleneck is the biggest untapped opportunity — 74% of params in lookup tables
- Depth increases are killed by VE overhead (each new VE layer costs ~2.1M params)
- Next experiments should focus on: removing VE to free params for depth, or recycling layers
