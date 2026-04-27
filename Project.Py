# =====================================================
# CRICKET MATCH PREDICTION USING MACHINE LEARNING
# Dataset : IPL Ball-by-Ball Data
# =====================================================


# =====================================================
# 1  IMPORT LIBRARIES
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =====================================================
# 2  LOAD DATA
# =====================================================
df = pd.read_csv(
    "IPL.csv",
    low_memory=False
)

print("=" * 55)
print("  RAW DATA INFO")
print("=" * 55)
print(f"  Total Rows    : {df.shape[0]:,}")
print(f"  Total Columns : {df.shape[1]}")
print(f"  Unique Matches: {df['match_id'].nunique()}")
print(f"  Seasons       : {sorted(df['season'].astype(str).unique())}")
print("=" * 55)


# =====================================================
# 3  CLEAN & AGGREGATE TO MATCH LEVEL
#    (one row per match)
# =====================================================

# Keep only first ball of each match to get match-level info
match_df = (
    df
    .sort_values(['match_id', 'innings', 'over', 'ball'])
    .groupby('match_id')
    .first()
    .reset_index()
)

# Keep relevant columns
cols = [
    'match_id', 'batting_team', 'bowling_team',
    'toss_winner', 'toss_decision',
    'match_won_by', 'venue', 'season'
]
match_df = match_df[cols].copy()

# Drop rows where winner is 'Unknown' or missing
match_df = match_df[match_df['match_won_by'] != 'Unknown']
match_df = match_df.dropna(subset=['match_won_by', 'toss_winner', 'toss_decision'])

# Clean season to numeric
def clean_season(s):
    s = str(s)
    if '/' in s:
        return int(s.split('/')[0])
    return int(float(s))

match_df['season'] = match_df['season'].apply(clean_season)

print(f"\n  Match-Level Rows (after cleaning) : {len(match_df)}")


# =====================================================
# 4  FEATURE ENGINEERING
# =====================================================

# Binary target: 1 = toss winner won the match, 0 = toss loser won
match_df['toss_winner_won'] = (
    match_df['toss_winner'] == match_df['match_won_by']
).astype(int)

# Team 1 is batting first
match_df['team1_won'] = (
    match_df['batting_team'] == match_df['match_won_by']
).astype(int)

# Encode categorical features
le_team = LabelEncoder()
le_venue = LabelEncoder()
le_toss = LabelEncoder()

all_teams = pd.concat([match_df['batting_team'], match_df['bowling_team']]).unique()
le_team.fit(all_teams)

match_df['batting_team_enc']  = le_team.transform(match_df['batting_team'])
match_df['bowling_team_enc']  = le_team.transform(match_df['bowling_team'])
match_df['venue_enc']         = le_venue.fit_transform(match_df['venue'])
match_df['toss_decision_enc'] = le_toss.fit_transform(match_df['toss_decision'])
match_df['toss_winner_enc']   = le_team.transform(match_df['toss_winner'])

# Features & Target
features = [
    'batting_team_enc',
    'bowling_team_enc',
    'toss_winner_enc',
    'toss_decision_enc',
    'venue_enc',
    'season'
]

X = match_df[features]
y = match_df['team1_won']   # 1 = batting-first team won

print(f"\n  Features Used : {features}")
print(f"  Target        : team1_won (1=Bat first team won, 0=Bowl first team won)")
print(f"  Class Balance : {y.value_counts().to_dict()}")


# =====================================================
# 5  TRAIN / VALIDATION / TEST SPLIT  (70 / 20 / 10)
# =====================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42
)

print(f"\n  Train  : {len(X_train)} samples")
print(f"  Val    : {len(X_val)}   samples")
print(f"  Test   : {len(X_test)}  samples")


# =====================================================
# 6  FEATURE SCALING
# =====================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# =====================================================
# 7  MODELS
# =====================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN'                : KNeighborsClassifier(n_neighbors=5),
    'SVM'                : SVC(kernel='rbf', probability=True),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42)
}


# =====================================================
# 8  VALIDATION RESULTS
# =====================================================
print("\n" + "=" * 55)
print("  VALIDATION RESULTS")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(
        f"  {name:<22} "
        f"Acc={accuracy_score(y_val, pred):.3f}  "
        f"F1={f1_score(y_val, pred):.3f}"
    )


# =====================================================
# 9  FINAL TEST EVALUATION TABLE
# =====================================================
results = []

for name, model in models.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)

    acc  = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec  = recall_score(y_test, pred)
    f1   = f1_score(y_test, pred)
    rauc = auc(fpr, tpr)

    results.append([name, acc, prec, rec, f1, rauc])

result_df = pd.DataFrame(
    results,
    columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
)

print("\n" + "=" * 55)
print("  MODEL COMPARISON (TEST SET)")
print("=" * 55)
print(result_df.to_string(index=False))

best = result_df.loc[result_df['F1 Score'].idxmax()]
print(f"\n  Best Model : {best['Model']}")
print(f"  F1 Score   : {best['F1 Score']:.3f}")
print(f"  Accuracy   : {best['Accuracy']:.3f}")
print(f"  ROC-AUC    : {best['ROC-AUC']:.3f}")


# =====================================================
# 10  DATASET STATISTICS (for poster facts)
# =====================================================
print("\n" + "=" * 55)
print("  DATASET STATISTICS")
print("=" * 55)
total = len(match_df)
bat_first_wins  = int(match_df['team1_won'].sum())
bowl_first_wins = total - bat_first_wins
toss_win_pct    = match_df['toss_winner_won'].mean() * 100

print(f"  Total Matches         : {total}")
print(f"  Bat-First Wins        : {bat_first_wins}  ({bat_first_wins/total*100:.1f}%)")
print(f"  Bowl-First Wins       : {bowl_first_wins} ({bowl_first_wins/total*100:.1f}%)")
print(f"  Toss Winner Win %     : {toss_win_pct:.1f}%")
print(f"  Unique Venues         : {match_df['venue'].nunique()}")
print(f"  Unique Teams          : {len(all_teams)}")
print(f"  Season Range          : {match_df['season'].min()} - {match_df['season'].max()}")


# =====================================================
# 11  GRAPH 1 — WIN DISTRIBUTION PIE CHART
# =====================================================
labels  = ['Bat First Wins', 'Bowl First Wins']
sizes   = [bat_first_wins, bowl_first_wins]
colors  = ['#1a3a6e', '#c0392b']

plt.figure(figsize=(5, 5))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
plt.title('Bat First vs Bowl First Wins', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('win_distribution.png', dpi=150)
plt.show()
print("\n  [Saved] win_distribution.png")


# =====================================================
# 12  GRAPH 2 — WINS PER SEASON
# =====================================================
season_wins = match_df.groupby('season')['team1_won'].agg(['sum','count'])
season_wins['bat_pct'] = season_wins['sum'] / season_wins['count'] * 100

plt.figure(figsize=(10, 4))
plt.bar(
    season_wins.index.astype(str),
    season_wins['bat_pct'],
    color='#1a3a6e',
    edgecolor='white'
)
plt.axhline(50, color='red', linestyle='--', linewidth=1, label='50% line')
plt.xlabel('Season')
plt.ylabel('Bat-First Win %')
plt.title('Bat-First Win % Per Season', fontsize=13, fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('season_wins.png', dpi=150)
plt.show()
print("  [Saved] season_wins.png")


# =====================================================
# 13  GRAPH 3 — CORRELATION HEATMAP
# =====================================================
corr_df = pd.DataFrame(X_train, columns=features)
corr_df['team1_won'] = y_train.values

plt.figure(figsize=(7, 5))
sns.heatmap(
    corr_df.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdBu',
    center=0,
    linewidths=0.5
)
plt.title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("  [Saved] correlation_heatmap.png")


# =====================================================
# 14  GRAPH 4 — COMBINED ROC CURVE
# =====================================================
plt.figure(figsize=(7, 5))

for name, model in models.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name}  AUC={roc_auc:.3f}", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curve — All Models', fontsize=13, fontweight='bold')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()
print("  [Saved] roc_curves.png")


# =====================================================
# 15  GRAPH 5 — CONFUSION MATRIX (Best Model)
# =====================================================
best_model_name = best['Model']
best_model = models[best_model_name]
best_pred  = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(5, 4))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Bowl-First Win', 'Bat-First Win']
)
disp.plot(
    cmap='Blues',
    colorbar=False,
    ax=plt.gca()
)
plt.title(
    f'Confusion Matrix — {best_model_name}',
    fontsize=11,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("  [Saved] confusion_matrix.png")


# =====================================================
# 16  GRAPH 6 — MODEL PERFORMANCE BAR CHART
# =====================================================
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
x = np.arange(len(result_df['Model']))
width = 0.15

fig, ax = plt.subplots(figsize=(10, 5))
colors_bar = ['#1a3a6e', '#c0392b', '#27ae60', '#f39c12', '#8e44ad']

for i, metric in enumerate(metrics):
    ax.bar(
        x + i * width,
        result_df[metric],
        width,
        label=metric,
        color=colors_bar[i],
        edgecolor='white'
    )

ax.set_xticks(x + width * 2)
ax.set_xticklabels(result_df['Model'], fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("  [Saved] model_comparison.png")


# =====================================================
# 17  SUMMARY
# =====================================================
print("\n" + "=" * 55)
print("  PROJECT SUMMARY — CRICKET MATCH PREDICTION")
print("=" * 55)
print(f"  Dataset       : IPL Ball-by-Ball (ball-level aggregated)")
print(f"  Total Matches : {total}")
print(f"  Features      : {len(features)}")
print(f"  Target        : Bat-First Team Win (Binary)")
print(f"  Best Model    : {best['Model']}")
print(f"  Accuracy      : {best['Accuracy']:.3f}")
print(f"  F1 Score      : {best['F1 Score']:.3f}")
print(f"  ROC-AUC       : {best['ROC-AUC']:.3f}")
print("=" * 55)
print("\n  All 6 graphs saved. Use them for your poster!")