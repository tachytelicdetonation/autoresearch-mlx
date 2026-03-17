"""Generate charts for the mar17 autoresearch run."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data ---
# Kept experiments (in order)
kept = [
    ("baseline", 1.923455, "baseline"),
    ("Muon optimizer", 1.724442, "architecture"),
    ("Batch 2^15", 1.671829, "hyperparameter"),
    ("PoPE", 1.658866, "architecture"),
    ("Batch 2^14", 1.642328, "hyperparameter"),
    ("5% warmup", 1.633649, "hyperparameter"),
    ("SwiGLU", 1.629301, "architecture"),
    ("NorMuon", 1.429509, "architecture"),
    ("Warmdown 0.67", 1.425042, "hyperparameter"),
    ("Embed LR 1.0", 1.416348, "hyperparameter"),
    ("Nesterov", 1.406336, "architecture"),
    ("Momentum 0.90", 1.394128, "hyperparameter"),
    ("Momentum 0.85", 1.392171, "hyperparameter"),
]

# All experiments (kept + discarded, in chronological order)
all_experiments = [
    ("baseline", 1.923455, "keep", "baseline"),
    ("Muon optimizer", 1.724442, "keep", "architecture"),
    ("Batch 2^15", 1.671829, "keep", "hyperparameter"),
    ("PoPE", 1.658866, "keep", "architecture"),
    ("Batch 2^14", 1.642328, "keep", "hyperparameter"),
    ("Depth=8", 2.078417, "discard", "hyperparameter"),
    ("Depth=6", 1.890840, "discard", "hyperparameter"),
    ("5% warmup", 1.633649, "keep", "hyperparameter"),
    ("SwiGLU", 1.629301, "keep", "architecture"),
    ("MLP 6x", 1.675638, "discard", "architecture"),
    ("Batch 2^13", 1.664235, "discard", "hyperparameter"),
    ("Matrix LR 0.08", 1.706638, "discard", "hyperparameter"),
    ("Matrix LR 0.06", 1.681841, "discard", "hyperparameter"),
    ("NorMuon", 1.429509, "keep", "architecture"),
    ("Warmdown 0.67", 1.425042, "keep", "hyperparameter"),
    ("Embed LR 1.0", 1.416348, "keep", "hyperparameter"),
    ("All-long attn", 1.420523, "discard", "hyperparameter"),
    ("Weight decay 0.1", 1.423305, "discard", "hyperparameter"),
    ("Aspect ratio 96", 1.504477, "discard", "hyperparameter"),
    ("HEAD_DIM 64", 1.510516, "discard", "hyperparameter"),
    ("Beta1 0.9", 1.431180, "discard", "hyperparameter"),
    ("Unembed LR 0.01", 1.418677, "discard", "hyperparameter"),
    ("Nesterov", 1.406336, "keep", "architecture"),
    ("Momentum 0.90", 1.394128, "keep", "hyperparameter"),
    ("Momentum 0.85", 1.392171, "keep", "hyperparameter"),
    ("NorMuon β2 0.99", 1.392696, "discard", "hyperparameter"),
    ("AttnRes", 1.684061, "discard", "architecture"),
    ("No VE", 1.454418, "discard", "architecture"),
]

plt.style.use('seaborn-v0_8-whitegrid')
colors = {'architecture': '#2196F3', 'hyperparameter': '#FF9800', 'baseline': '#9E9E9E'}

# ============================================================
# Chart 1: val_bpb progression (kept experiments only)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
names = [k[0] for k in kept]
vals = [k[1] for k in kept]
cats = [k[2] for k in kept]
bar_colors = [colors[c] for c in cats]

bars = ax.bar(range(len(names)), vals, color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('val_bpb (lower is better)', fontsize=11)
ax.set_title('autoresearch/mar17 — Kept Experiments Progression', fontsize=14, fontweight='bold')
ax.set_ylim(1.3, 2.0)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, vals)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add improvement arrows
for i in range(1, len(vals)):
    delta = vals[i] - vals[i-1]
    ax.annotate(f'{delta:+.3f}', xy=(i, vals[i] - 0.02),
                fontsize=7, ha='center', va='top', color='#333333')

legend_patches = [
    mpatches.Patch(color=colors['architecture'], label='Architecture/Optimizer'),
    mpatches.Patch(color=colors['hyperparameter'], label='Hyperparameter'),
    mpatches.Patch(color=colors['baseline'], label='Baseline'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('charts/progression.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 2: All experiments — kept vs discarded
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Track the "current best" line
current_best = all_experiments[0][1]
best_line_x = []
best_line_y = []

for i, (name, val, status, cat) in enumerate(all_experiments):
    if status == 'keep':
        current_best = val
        marker = 'o'
        alpha = 1.0
        size = 80
        edgecolor = '#1B5E20'
        linewidth = 2
    else:
        marker = 'X'
        alpha = 0.5
        size = 60
        edgecolor = '#B71C1C'
        linewidth = 1.5

    ax.scatter(i, val, c=colors[cat], marker=marker, s=size, alpha=alpha,
               edgecolors=edgecolor, linewidths=linewidth, zorder=3)
    best_line_x.append(i)
    best_line_y.append(current_best)

# Draw the "current best" step line
ax.step(best_line_x, best_line_y, where='post', color='#4CAF50', linewidth=2,
        alpha=0.7, linestyle='--', label='Current best', zorder=2)

ax.set_xticks(range(len(all_experiments)))
ax.set_xticklabels([e[0] for e in all_experiments], rotation=60, ha='right', fontsize=7)
ax.set_ylabel('val_bpb (lower is better)', fontsize=11)
ax.set_title('autoresearch/mar17 — All Experiments (Kept vs Discarded)', fontsize=14, fontweight='bold')

# Custom legend
legend_items = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888', markeredgecolor='#1B5E20',
               markersize=10, markeredgewidth=2, label='Kept'),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#888', markeredgecolor='#B71C1C',
               markersize=10, markeredgewidth=1.5, label='Discarded'),
    mpatches.Patch(color=colors['architecture'], label='Architecture/Optimizer'),
    mpatches.Patch(color=colors['hyperparameter'], label='Hyperparameter'),
    plt.Line2D([0], [0], color='#4CAF50', linewidth=2, linestyle='--', label='Current best'),
]
ax.legend(handles=legend_items, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('charts/all_experiments.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 3: Architecture vs Hyperparameter contribution
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
arch_gain = abs(sum(kept[i][1] - kept[i-1][1] for i in range(1, len(kept)) if kept[i][2] == 'architecture'))
hyper_gain = abs(sum(kept[i][1] - kept[i-1][1] for i in range(1, len(kept)) if kept[i][2] == 'hyperparameter'))
total = arch_gain + hyper_gain

ax1.pie([arch_gain, hyper_gain],
        labels=[f'Architecture\n({arch_gain:.3f} bpb)', f'Hyperparameter\n({hyper_gain:.3f} bpb)'],
        colors=[colors['architecture'], colors['hyperparameter']],
        autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax1.set_title('Share of Total Improvement', fontsize=13, fontweight='bold')

# Waterfall chart
improvements = []
for i in range(1, len(kept)):
    delta = kept[i-1][1] - kept[i][1]  # positive = improvement
    improvements.append((kept[i][0], delta, kept[i][2]))

improvements.sort(key=lambda x: x[1], reverse=True)
names_wf = [x[0] for x in improvements]
deltas = [x[1] for x in improvements]
cats_wf = [colors[x[2]] for x in improvements]

bars = ax2.barh(range(len(names_wf)), deltas, color=cats_wf, edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(names_wf)))
ax2.set_yticklabels(names_wf, fontsize=9)
ax2.set_xlabel('val_bpb improvement (higher = better)', fontsize=10)
ax2.set_title('Individual Experiment Impact', fontsize=13, fontweight='bold')
ax2.invert_yaxis()

for bar, delta in zip(bars, deltas):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'-{delta:.3f}', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/arch_vs_hyper.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Chart 4: Accept/Reject rate
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

kept_arch = sum(1 for e in all_experiments if e[2] == 'keep' and e[3] == 'architecture')
disc_arch = sum(1 for e in all_experiments if e[2] == 'discard' and e[3] == 'architecture')
kept_hyp = sum(1 for e in all_experiments if e[2] == 'keep' and e[3] == 'hyperparameter')
disc_hyp = sum(1 for e in all_experiments if e[2] == 'discard' and e[3] == 'hyperparameter')

x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [kept_arch, kept_hyp], width, label='Kept',
               color=['#2196F3', '#FF9800'], edgecolor='#1B5E20', linewidth=2)
bars2 = ax.bar(x + width/2, [disc_arch, disc_hyp], width, label='Discarded',
               color=['#90CAF9', '#FFE0B2'], edgecolor='#B71C1C', linewidth=2)

ax.set_xticks(x)
ax.set_xticklabels(['Architecture/Optimizer', 'Hyperparameter'], fontsize=11)
ax.set_ylabel('Number of experiments', fontsize=11)
ax.set_title('Experiment Accept/Reject Rate by Category', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(int(bar.get_height())), ha='center', fontweight='bold')

# Add accept rates
arch_rate = kept_arch / (kept_arch + disc_arch) * 100
hyp_rate = kept_hyp / (kept_hyp + disc_hyp) * 100
ax.text(0, max(kept_arch, disc_arch) + 1, f'{arch_rate:.0f}% accept', ha='center', fontsize=10, color='#1B5E20')
ax.text(1, max(kept_hyp, disc_hyp) + 1.5, f'{hyp_rate:.0f}% accept', ha='center', fontsize=10, color='#1B5E20')

plt.tight_layout()
plt.savefig('charts/accept_reject.png', dpi=150, bbox_inches='tight')
plt.close()

print("Charts generated in charts/")
