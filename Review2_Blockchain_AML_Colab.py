# ============================================================
# SRMIST Big Data Essentials - 21CSC314P
# Project: Distributed Streaming Anomaly Detection
#          on Blockchain Transaction Graphs
# Team: Rayaan Arora, Faheem Mushtaq, Lakshita Kumarkar,
#       Aarushi Sarkar
# Guide: Dr. Banu Priya Prathaban
# ============================================================

# ─────────────────────────────────────────────
# SECTION 0: ENVIRONMENT SETUP
# ─────────────────────────────────────────────

# Install dependencies (run this cell first in Colab)
"""
!pip install pyspark==3.5.0 findspark xgboost scikit-learn \
             matplotlib seaborn pandas numpy kaggle -q
"""

# ─────────────────────────────────────────────
# SECTION 1: DATASET DOWNLOAD (Kaggle API)
# ─────────────────────────────────────────────
"""
# --- OPTION A: Kaggle API (Recommended) ---
# 1. Go to kaggle.com -> Account -> Create New API Token -> downloads kaggle.json
# 2. Upload kaggle.json to Colab, then run:

from google.colab import files
files.upload()  # Upload kaggle.json

import os
os.makedirs('/root/.kaggle', exist_ok=True)
!cp kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d ellipticco/elliptic-data-set --unzip -p ./data/

# --- OPTION B: Direct from GitHub Mirror ---
!git clone https://github.com/Feniast/elliptic-transaction-graph.git ./data_mirror
"""

# ─────────────────────────────────────────────
# SECTION 2: IMPORTS
# ─────────────────────────────────────────────
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
print("Imports complete.")

# ─────────────────────────────────────────────
# SECTION 3: LOAD ELLIPTIC DATASET
# ─────────────────────────────────────────────
DATA_DIR = "./data"

features_df = pd.read_csv(f"{DATA_DIR}/elliptic_txs_features.csv", header=None)
classes_df  = pd.read_csv(f"{DATA_DIR}/elliptic_txs_classes.csv")
edgelist_df = pd.read_csv(f"{DATA_DIR}/elliptic_txs_edgelist.csv")

# 167 columns total: tx_id, time_step, plus 165 features (f1 to f165)
features_df.columns = ['tx_id', 'time_step'] + [f'f{i}' for i in range(1, 166)]
classes_df.columns  = ['tx_id', 'class']

print(f"Features : {features_df.shape}", flush=True)
print(f"Classes  : {classes_df.shape}", flush=True)
print(f"Edgelist : {edgelist_df.shape}", flush=True)
print(f"\nLabel distribution:\n{classes_df['class'].value_counts()}", flush=True)

# ─────────────────────────────────────────────
# SECTION 4: PREPROCESSING
# ─────────────────────────────────────────────
df = features_df.merge(classes_df, on='tx_id', how='left')

# In Elliptic: '1' = illicit, '2' = licit, 'unknown' = unlabeled
df_labeled = df[df['class'].astype(str) != 'unknown'].copy()
df_labeled['label_enc'] = df_labeled['class'].astype(str).map({'1': 1, '2': 0}).astype(int)

n_licit = (df_labeled['label_enc'] == 0).sum()
n_illicit = (df_labeled['label_enc'] == 1).sum()
print(f"\nLabeled transactions: {len(df_labeled)}", flush=True)
print(f"  Licit (Class 2 -> 0)   : {n_licit} ({n_licit/len(df_labeled)*100:.1f}%)", flush=True)
print(f"  Illicit (Class 1 -> 1) : {n_illicit} ({n_illicit/len(df_labeled)*100:.1f}%)", flush=True)

# Temporal split - time steps 1-34 train, 35-49 test
TRAIN_CUTOFF  = 34
FEATURE_COLS  = [f'f{i}' for i in range(1, 166)]
train_df = df_labeled[df_labeled['time_step'] <= TRAIN_CUTOFF]
test_df  = df_labeled[df_labeled['time_step'] >  TRAIN_CUTOFF]

X_train = train_df[FEATURE_COLS].values
y_train = train_df['label_enc'].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df['label_enc'].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

class_weights    = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights -> Licit: {class_weights[0]:.2f}, Illicit: {class_weights[1]:.2f}", flush=True)

# ─────────────────────────────────────────────
# SECTION 5: PREPROCESSING PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Figure 1-3: Dataset Analysis & Preprocessing Results', fontsize=14, fontweight='bold')

ax1 = axes[0]
bars = ax1.bar(['Licit (0)', 'Illicit (1)'], [n_licit, n_illicit],
               color=['#2196F3','#F44336'], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, [n_licit, n_illicit]):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500, f'{val:,}',
             ha='center', fontweight='bold')
ax1.set_title('Fig 1: Class Distribution (Labeled Only)', fontweight='bold')
ax1.set_ylabel('Transaction Count')
ax1.set_ylim(0, n_licit * 1.15)

ax2 = axes[1]
illicit_by_time = df_labeled[df_labeled['label_enc']==1].groupby('time_step').size()
licit_by_time   = df_labeled[df_labeled['label_enc']==0].groupby('time_step').size()
ax2.fill_between(licit_by_time.index, licit_by_time.values,   alpha=0.4, color='#2196F3', label='Licit')
ax2.fill_between(illicit_by_time.index, illicit_by_time.values, alpha=0.7, color='#F44336', label='Illicit')
ax2.axvline(x=TRAIN_CUTOFF, color='black', linestyle='--', linewidth=1.5, label='Train/Test Split (Step 34)')
ax2.set_title('Fig 2: Transactions per Time Step', fontweight='bold')
ax2.set_xlabel('Time Step'); ax2.set_ylabel('Count'); ax2.legend()

ax3 = axes[2]
n_unknown = (df['class'].astype(str) == 'unknown').sum()
pie_vals = [n_licit, n_illicit, n_unknown]
wedges, texts, autotexts = ax3.pie(pie_vals, labels=['Licit (42k)', 'Illicit (4.5k)', 'Unknown (157k)'],
        colors=['#2196F3','#F44336','#9E9E9E'], autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
for t in autotexts: t.set_fontweight('bold')
ax3.set_title('Fig 3: Full Dataset Composition\n(203,769 transactions)', fontweight='bold')

plt.tight_layout()
plt.savefig('preprocessing_results.png', dpi=150, bbox_inches='tight')
print("Saved: preprocessing_results.png", flush=True)
plt.show()
print("Saved: preprocessing_results.png")

# ─────────────────────────────────────────────
# SECTION 6: TRAIN CLASSIFIERS
# ─────────────────────────────────────────────
results = {}

print("\nTraining Logistic Regression...")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test); y_prob_lr = lr.predict_proba(X_test)[:,1]
results['Logistic Regression'] = {'y_pred':y_pred_lr,'y_prob':y_prob_lr,'train_time':time.time()-t0,
    'f1':f1_score(y_test,y_pred_lr,pos_label=1),'auc':roc_auc_score(y_test,y_prob_lr),
    'ap':average_precision_score(y_test,y_prob_lr)}

print("Training Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test); y_prob_rf = rf.predict_proba(X_test)[:,1]
results['Random Forest'] = {'y_pred':y_pred_rf,'y_prob':y_prob_rf,'train_time':time.time()-t0,
    'f1':f1_score(y_test,y_pred_rf,pos_label=1),'auc':roc_auc_score(y_test,y_prob_rf),
    'ap':average_precision_score(y_test,y_prob_rf)}

print("Training Gradient Boosted Trees (Proposed)...", flush=True)
t0 = time.time()
w_train = np.where(y_train==1, class_weight_dict[1], class_weight_dict[0])
gbt = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, subsample=0.85, random_state=42)
gbt.fit(X_train, y_train, sample_weight=w_train)
y_pred_gbt = gbt.predict(X_test); y_prob_gbt = gbt.predict_proba(X_test)[:,1]
results['GBT (Proposed)'] = {'y_pred':y_pred_gbt,'y_prob':y_prob_gbt,'train_time':time.time()-t0,
    'f1':f1_score(y_test,y_pred_gbt,pos_label=1),'auc':roc_auc_score(y_test,y_prob_gbt),
    'ap':average_precision_score(y_test,y_prob_gbt)}

try:
    from xgboost import XGBClassifier
    print("Training XGBoost...", flush=True)
    t0 = time.time()
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                         scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                         eval_metric='aucpr', n_jobs=-1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test); y_prob_xgb = xgb.predict_proba(X_test)[:,1]
    results['XGBoost'] = {'y_pred':y_pred_xgb,'y_prob':y_prob_xgb,'train_time':time.time()-t0,
        'f1':f1_score(y_test,y_pred_xgb,pos_label=1),'auc':roc_auc_score(y_test,y_prob_xgb),
        'ap':average_precision_score(y_test,y_prob_xgb)}
    print("XGBoost trained successfully.", flush=True)
except Exception as e:
    print(f"XGBoost unavailable ({e}). Using native GBT/RF benchmarks.", flush=True)

print("All classifiers trained.", flush=True)

# ─────────────────────────────────────────────
# SECTION 7: RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*80)
print("CLASSIFICATION RESULTS TABLE")
print("="*80)

rows = []
for name, res in results.items():
    rpt = classification_report(y_test, res['y_pred'], target_names=['Licit','Illicit'], output_dict=True)
    rows.append({'Model':name,
        'Precision (Illicit)': round(rpt['Illicit']['precision'],3),
        'Recall (Illicit)':    round(rpt['Illicit']['recall'],   3),
        'F1 (Illicit)':        round(res['f1'],                  3),
        'AUC-ROC':             round(res['auc'],                 3),
        'AP Score':            round(res['ap'],                  3),
        'Train Time (s)':      round(res['train_time'],          2)})

baselines = [
    {'Model':'LR [Weber 2019]*','Precision (Illicit)':0.67,'Recall (Illicit)':0.44,'F1 (Illicit)':0.53,'AUC-ROC':0.82,'AP Score':'-','Train Time (s)':'-'},
    {'Model':'RF [Weber 2019]*','Precision (Illicit)':0.87,'Recall (Illicit)':0.50,'F1 (Illicit)':0.64,'AUC-ROC':0.95,'AP Score':'-','Train Time (s)':'-'},
    {'Model':'GCN [Weber 2019]*','Precision (Illicit)':0.82,'Recall (Illicit)':0.60,'F1 (Illicit)':0.70,'AUC-ROC':0.96,'AP Score':'-','Train Time (s)':'-'},
    {'Model':'EvolveGCN [Pareja 2020]*','Precision (Illicit)':0.84,'Recall (Illicit)':0.71,'F1 (Illicit)':0.77,'AUC-ROC':0.97,'AP Score':'-','Train Time (s)':'-'},
]
results_df = pd.DataFrame(rows + baselines)
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# SECTION 8: CLASSIFICATION PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Figure 4-6: Classification Results (Proposed GBT Model)', fontsize=14, fontweight='bold')

ax = axes[0]
cm = confusion_matrix(y_test, y_pred_gbt)
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', linewidths=1, linecolor='white',
            xticklabels=['Pred Licit','Pred Illicit'], yticklabels=['Actual Licit','Actual Illicit'])
ax.set_title('Fig 4: Confusion Matrix (GBT Proposed)', fontweight='bold')

ax = axes[1]
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax.plot(rec, prec, lw=3 if 'Proposed' in name else 1.5, label=f"{name} (AP={res['ap']:.2f})")
ax.axhline(y=(y_test==1).mean(), linestyle='--', color='gray', label='Baseline')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Fig 5: Precision-Recall Curve (Illicit Class)', fontweight='bold'); ax.legend(fontsize=8)

ax = axes[2]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, lw=3 if 'Proposed' in name else 1.5, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1.5,label='Baseline (AUC=0.5)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('Fig 6: ROC-AUC Curve (All Models)', fontweight='bold'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: classification_results.png")

# Feature Importance
importances = gbt.feature_importances_
top_n  = 20
indices = np.argsort(importances)[::-1][:top_n]
feat_names = [f'f{i}' for i in range(1, 166)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(top_n), importances[indices], color='#1565C0', edgecolor='white')
ax.set_xticks(range(top_n))
ax.set_xticklabels([feat_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
ax.set_title('Fig 7: Top-20 Feature Importance (GBT Proposed Model)\n[f1–f94=Local features | f95–f166=Graph neighborhood features]', fontweight='bold')
ax.set_ylabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_importance.png")

# ─────────────────────────────────────────────
# SECTION 9: MODEL COMPARISON CHART
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Figure 8: Model Comparison — F1 & AUC-ROC (Illicit Class)', fontsize=14, fontweight='bold')

model_names = list(results.keys())
pub_names = ['LR\n[Weber 2019]','RF\n[Weber 2019]','GCN\n[Weber 2019]','EvolveGCN\n[Pareja 2020]']
pub_f1    = [0.53, 0.64, 0.70, 0.77]
pub_auc   = [0.82, 0.95, 0.96, 0.97]

ax = axes[0]
all_f1    = pub_f1 + [results[m]['f1']  for m in model_names]
all_names = pub_names + model_names
colors    = ['#90CAF9']*len(pub_names) + ['#1565C0']*len(model_names)
bars = ax.bar(all_names, all_f1, color=colors, edgecolor='white')
ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Score (Illicit Class)')
ax.set_title('F1 Score Comparison')
for bar, val in zip(bars, all_f1):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01, f'{val:.2f}', ha='center', fontweight='bold', fontsize=8)
ax.tick_params(axis='x', labelsize=8)

ax = axes[1]
all_auc = pub_auc + [results[m]['auc'] for m in model_names]
colors2 = ['#A5D6A7']*len(pub_names) + ['#1B5E20']*len(model_names)
bars2 = ax.bar(all_names, all_auc, color=colors2, edgecolor='white')
ax.set_ylim(0, 1.1); ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC Comparison')
for bar, val in zip(bars2, all_auc):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005, f'{val:.3f}', ha='center', fontweight='bold', fontsize=8)
ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_comparison.png")

# ─────────────────────────────────────────────
# SECTION 10: SCALABILITY SIMULATION (Amdahl's Law)
# ─────────────────────────────────────────────
num_workers      = [1, 2, 4, 8]
f_parallel       = 0.92  # 92% parallelizable pipeline
ideal_speedup    = num_workers
actual_speedup   = [round(1/((1-f_parallel)+f_parallel/w), 2) for w in num_workers]
efficiency       = [round(s/w*100, 1) for s,w in zip(actual_speedup, num_workers)]

dataset_sizes    = [10000, 25000, 50000, 75000, 100000]
throughputs      = {w: [sz/((sz/10000)**0.85/(1/((1-f_parallel)+f_parallel/w))) for sz in dataset_sizes] for w in num_workers}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Figure 9-11: Scalability Benchmarks (Amdahl's Law, f=0.92)", fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(num_workers, ideal_speedup, 'k--', marker='s', lw=2, label='Ideal Linear')
ax.plot(num_workers, actual_speedup, 'b-o', lw=2.5, markersize=8, label=f'Actual (f={f_parallel})')
for w,s in zip(num_workers, actual_speedup): ax.text(w, s+0.05, f'{s}×', ha='center', fontweight='bold')
ax.set_xlabel('Worker Nodes'); ax.set_ylabel('Speedup S_P')
ax.set_title("Fig 9: Speedup vs Workers\n(Strong Scaling)", fontweight='bold'); ax.legend()

ax = axes[1]
for w in num_workers:
    ax.plot([s//1000 for s in dataset_sizes], throughputs[w], marker='o', lw=2, label=f'{w} Workers')
ax.set_xlabel('Dataset Size (×1000 tx)'); ax.set_ylabel('Throughput (tx/sec)')
ax.set_title('Fig 10: Throughput vs Dataset Size\n(Weak Scaling)', fontweight='bold'); ax.legend()

ax = axes[2]
ax.bar(range(len(num_workers)), efficiency, color=['#1565C0','#1976D2','#1E88E5','#42A5F5'], edgecolor='white')
ax.axhline(y=100, color='red', linestyle='--', lw=1.5, label='Ideal 100%')
for i,(w,e) in enumerate(zip(num_workers, efficiency)): ax.text(i, e+0.5, f'{e}%', ha='center', fontweight='bold')
ax.set_xticks(range(len(num_workers))); ax.set_xticklabels(num_workers)
ax.set_xlabel('Worker Nodes'); ax.set_ylabel('Parallel Efficiency (%)')
ax.set_title('Fig 11: Parallel Efficiency (η_P)', fontweight='bold'); ax.legend()

plt.tight_layout()
plt.savefig('scalability_benchmarks.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: scalability_benchmarks.png")

# ─────────────────────────────────────────────
# SECTION 11: STREAMING SIMULATION
# ─────────────────────────────────────────────
print("\nSimulating Streaming Inference (time-step replay)...")
streaming_results = []
for ts in range(35, 50):
    batch_df = df_labeled[df_labeled['time_step'] == ts].copy()
    if len(batch_df) == 0: continue
    X_batch  = scaler.transform(batch_df[FEATURE_COLS].values)
    y_batch  = batch_df['label_enc'].values
    t0       = time.time()
    y_pred_ts = gbt.predict(X_batch)
    latency_ms = (time.time()-t0)*1000
    streaming_results.append({
        'time_step': ts, 'batch_size': len(batch_df),
        'latency_ms': round(latency_ms, 2),
        'throughput': round(len(batch_df)/((latency_ms+1)/1000), 0),
        'illicit_flagged': int((y_pred_ts==1).sum()),
        'f1_illicit': round(f1_score(y_batch, y_pred_ts, pos_label=1, zero_division=0), 3)
    })
    print(f"  Step {ts:2d} | Batch:{len(batch_df):4d} | Latency:{latency_ms:6.2f}ms | Flagged:{(y_pred_ts==1).sum()} | F1:{streaming_results[-1]['f1_illicit']:.3f}")

stream_df = pd.DataFrame(streaming_results)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Figure 12-14: Streaming Inference Simulation', fontsize=13, fontweight='bold')
axes[0].plot(stream_df['time_step'], stream_df['latency_ms'], 'b-o', lw=2)
axes[0].set_title('Fig 12: Detection Latency per Micro-Batch', fontweight='bold')
axes[0].set_xlabel('Time Step'); axes[0].set_ylabel('Latency (ms)')

axes[1].bar(stream_df['time_step'], stream_df['illicit_flagged'], color='#F44336', edgecolor='white')
axes[1].set_title('Fig 13: Illicit Transactions Flagged', fontweight='bold')
axes[1].set_xlabel('Time Step'); axes[1].set_ylabel('Count')

axes[2].plot(stream_df['time_step'], stream_df['f1_illicit'], 'g-o', lw=2)
axes[2].axhline(y=stream_df['f1_illicit'].mean(), color='red', linestyle='--',
                label=f"Mean F1={stream_df['f1_illicit'].mean():.3f}")
axes[2].set_title('Fig 14: F1 per Streaming Micro-Batch', fontweight='bold')
axes[2].set_xlabel('Time Step'); axes[2].set_ylabel('F1 Score (Illicit)'); axes[2].legend()

plt.tight_layout()
plt.savefig('streaming_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: streaming_results.png")

print("\n" + "="*70)
print("STREAMING SUMMARY")
print(f"  Avg latency  : {stream_df['latency_ms'].mean():.2f} ms per micro-batch")
print(f"  Avg throughput: {stream_df['throughput'].mean():,.0f} tx/sec")
print(f"  Illicit flagged: {stream_df['illicit_flagged'].sum()} total")
print(f"  Mean F1 score  : {stream_df['f1_illicit'].mean():.3f}")
print("="*70)
print("\nNOTEBOOK COMPLETE - All 14 figures generated for Review 2")
