import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & Encode ───────────────────────────────────────────────────────────
df = pd.read_csv('bank_marketing.csv')

cat_cols = ['job','marital','education','default','housing',
            'loan','contact','month','poutcome']
num_cols = ['age','balance','day','duration','campaign','pdays','previous']
feature_names = cat_cols + num_cols

df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df_enc[feature_names]
y = df_enc['y'].map({'no': 0, 'yes': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. Train Decision Tree ─────────────────────────────────────────────────────
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]
acc  = round((y_pred == y_test).mean() * 100, 2)
auc  = round(roc_auc_score(y_test, y_prob) * 100, 2)

print("=" * 55)
print("        BANK MARKETING — DECISION TREE")
print("=" * 55)
print(f"  Accuracy : {acc}%")
print(f"  ROC-AUC  : {auc}%")
print(f"  Depth    : {dt.get_depth()}   Leaves: {dt.get_n_leaves()}")
print()
print(classification_report(y_test, y_pred, target_names=['No','Yes']))

# ── 3. Text Rules (depth 3) ────────────────────────────────────────────────────
dt3 = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
dt3.fit(X_train, y_train)
print("\n── Decision Rules (depth 3) ─────────────────────────────")
print(export_text(dt3, feature_names=feature_names))

# ── 4. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 22), facecolor='#F8F9FA')
fig.suptitle('Bank Marketing — Decision Tree Classifier',
             fontsize=20, fontweight='bold', y=0.98,
             color='#1a1a2e')

gs = fig.add_gridspec(3, 3,
                      height_ratios=[2.5, 1, 1],
                      hspace=0.45, wspace=0.35,
                      left=0.04, right=0.97,
                      top=0.95, bottom=0.04)

TEAL   = '#0F6E56'
RED    = '#A32D2D'
BLUE   = '#185FA5'
AMBER  = '#854F0B'
LBLUE  = '#B5D4F4'
LTEAL  = '#9FE1CB'
LGRAY  = '#E8E8E8'

# ── 4a. Main tree plot (depth 4 for readability) ──────────────────────────────
ax_tree = fig.add_subplot(gs[0, :])
ax_tree.set_facecolor('#FFFFFF')

dt4 = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
dt4.fit(X_train, y_train)

plot_tree(
    dt4,
    feature_names=feature_names,
    class_names=['No ✗', 'Yes ✓'],
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    precision=2,
    fontsize=8,
    ax=ax_tree,
    node_ids=False,
)

# Re-colour nodes: internal=blue tint, leaf No=red tint, leaf Yes=teal tint
for artist in ax_tree.get_children():
    if hasattr(artist, 'get_facecolor'):
        fc = artist.get_facecolor()
        if fc is not None:
            # matplotlib colours nodes using orange→blue gradient by default;
            # we rely on filled=True defaults and just add a title
            pass

ax_tree.set_title('Decision Tree Structure (max depth = 4)',
                  fontsize=13, fontweight='bold', pad=8, color='#1a1a2e')

leg_patches = [
    mpatches.Patch(facecolor='#AED6F1', label='Internal node (split)'),
    mpatches.Patch(facecolor='#A9DFBF', label='Leaf → Yes (subscribe)'),
    mpatches.Patch(facecolor='#F1948A', label='Leaf → No (no subscribe)'),
]
ax_tree.legend(handles=leg_patches, loc='upper right',
               fontsize=9, framealpha=0.85)

# ── 4b. Feature Importance bar ────────────────────────────────────────────────
ax_fi = fig.add_subplot(gs[1, 0])
ax_fi.set_facecolor('#FFFFFF')

fi   = dt.feature_importances_
fi_s = sorted(zip(feature_names, fi), key=lambda x: x[1])[-10:]
names, vals = zip(*fi_s)
bar_colors = [TEAL if v > 0.05 else BLUE if v > 0.01 else LBLUE for v in vals]

bars = ax_fi.barh(names, [v * 100 for v in vals],
                  color=bar_colors, edgecolor='none', height=0.6)
for bar, val in zip(bars, vals):
    ax_fi.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
               f'{val*100:.1f}%', va='center', fontsize=8, color='#333')

ax_fi.set_xlabel('Importance (%)', fontsize=9)
ax_fi.set_title('Feature Importance', fontsize=11,
                fontweight='bold', color='#1a1a2e')
ax_fi.spines[['top','right']].set_visible(False)
ax_fi.tick_params(labelsize=8)
ax_fi.set_xlim(0, max(vals) * 120)

# ── 4c. Confusion Matrix ──────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1, 1])
ax_cm.set_facecolor('#FFFFFF')

cm = confusion_matrix(y_test, y_pred)
labels = [['TN', 'FP'], ['FN', 'TP']]
cell_colors = [[LTEAL, '#F7C1C1'], ['#FAC775', LTEAL]]
cell_text_colors = [['#0F6E56', '#A32D2D'], ['#854F0B', '#0F6E56']]

ax_cm.set_xlim(0, 2); ax_cm.set_ylim(0, 2)
for i in range(2):
    for j in range(2):
        ax_cm.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                       facecolor=cell_colors[i][j],
                                       edgecolor='white', linewidth=2))
        ax_cm.text(j + 0.5, 1.5 - i,
                   f'{labels[i][j]}\n{cm[i, j]:,}',
                   ha='center', va='center', fontsize=11,
                   fontweight='bold', color=cell_text_colors[i][j])

ax_cm.set_xticks([0.5, 1.5])
ax_cm.set_xticklabels(['Pred: No', 'Pred: Yes'], fontsize=9)
ax_cm.set_yticks([0.5, 1.5])
ax_cm.set_yticklabels(['Actual: Yes', 'Actual: No'], fontsize=9)
ax_cm.set_title('Confusion Matrix', fontsize=11,
                fontweight='bold', color='#1a1a2e')
ax_cm.tick_params(length=0)
for sp in ax_cm.spines.values():
    sp.set_visible(False)

# ── 4d. ROC Curve ─────────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 2])
ax_roc.set_facecolor('#FFFFFF')

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax_roc.plot(fpr, tpr, color=BLUE, lw=2,
            label=f'Decision Tree (AUC = {auc}%)')
ax_roc.plot([0, 1], [0, 1], color='#B4B2A9', lw=1,
            linestyle='--', label='Random classifier')
ax_roc.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax_roc.set_xlabel('False Positive Rate', fontsize=9)
ax_roc.set_ylabel('True Positive Rate', fontsize=9)
ax_roc.set_title('ROC Curve', fontsize=11,
                 fontweight='bold', color='#1a1a2e')
ax_roc.legend(fontsize=8, loc='lower right')
ax_roc.spines[['top','right']].set_visible(False)
ax_roc.tick_params(labelsize=8)

# ── 4e. Metrics bar chart ─────────────────────────────────────────────────────
ax_met = fig.add_subplot(gs[2, 0])
ax_met.set_facecolor('#FFFFFF')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
metrics = {
    'Accuracy':  accuracy_score(y_test, y_pred) * 100,
    'Precision': precision_score(y_test, y_pred) * 100,
    'Recall':    recall_score(y_test, y_pred) * 100,
    'F1 Score':  f1_score(y_test, y_pred) * 100,
    'AUC':       auc,
}
m_names = list(metrics.keys())
m_vals  = list(metrics.values())
m_colors = [TEAL, BLUE, AMBER, '#993C1D', '#3C3489']

bars2 = ax_met.bar(m_names, m_vals, color=m_colors,
                   edgecolor='none', width=0.55)
for bar, val in zip(bars2, m_vals):
    ax_met.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='#333')

ax_met.set_ylim(0, 115)
ax_met.set_ylabel('Score (%)', fontsize=9)
ax_met.set_title('Performance Metrics', fontsize=11,
                 fontweight='bold', color='#1a1a2e')
ax_met.spines[['top','right']].set_visible(False)
ax_met.tick_params(axis='x', labelsize=8.5, rotation=10)
ax_met.tick_params(axis='y', labelsize=8)
ax_met.axhline(100, color=LGRAY, linewidth=0.8, linestyle='--')

# ── 4f. Depth vs Accuracy ─────────────────────────────────────────────────────
ax_depth = fig.add_subplot(gs[2, 1])
ax_depth.set_facecolor('#FFFFFF')

depths = list(range(1, 16))
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_accs.append(m.score(X_train, y_train) * 100)
    test_accs.append(m.score(X_test, y_test) * 100)

ax_depth.plot(depths, train_accs, 'o-', color=TEAL,
              lw=2, ms=5, label='Train accuracy')
ax_depth.plot(depths, test_accs,  's--', color=RED,
              lw=2, ms=5, label='Test accuracy')
ax_depth.axvline(5, color=AMBER, linestyle=':', lw=1.5,
                 label='Selected depth=5')
ax_depth.fill_between(depths, train_accs, test_accs,
                      alpha=0.07, color=AMBER, label='Overfit gap')
ax_depth.set_xlabel('Max Depth', fontsize=9)
ax_depth.set_ylabel('Accuracy (%)', fontsize=9)
ax_depth.set_title('Depth vs Accuracy', fontsize=11,
                   fontweight='bold', color='#1a1a2e')
ax_depth.legend(fontsize=7.5)
ax_depth.spines[['top','right']].set_visible(False)
ax_depth.tick_params(labelsize=8)

# ── 4g. Class distribution bar ────────────────────────────────────────────────
ax_dist = fig.add_subplot(gs[2, 2])
ax_dist.set_facecolor('#FFFFFF')

vc = df['y'].value_counts()
bar_d = ax_dist.bar(['No (88.3%)', 'Yes (11.7%)'],
                    [vc['no'], vc['yes']],
                    color=[RED, TEAL], edgecolor='none', width=0.5)
for bar, val in zip(bar_d, [vc['no'], vc['yes']]):
    ax_dist.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 200,
                 f'{val:,}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color='#333')

ax_dist.set_ylabel('Number of Clients', fontsize=9)
ax_dist.set_title('Class Distribution', fontsize=11,
                  fontweight='bold', color='#1a1a2e')
ax_dist.spines[['top','right']].set_visible(False)
ax_dist.tick_params(labelsize=9)

# ── 5. Save ───────────────────────────────────────────────────────────────────
out = 'decision_tree_visualization.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\n✓ Saved → {out}")
