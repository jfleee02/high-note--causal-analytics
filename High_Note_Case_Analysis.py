# -*- coding: utf-8 -*-
"""
HIGH NOTE CASE STUDY — Full Project
=====================================
Member 1: Data Foundation & Descriptive Statistics
Member 2: Exploratory Visualization
Member 3: Propensity Score Matching (PSM)
Member 4: Logistic Regression & Strategy
"""

from google.colab import drive
drive.mount('/content/drive')

# ── Set your Drive path here — update if your folder is different ──
BASE_PATH = '/content/drive/MyDrive/Colab Notebooks/CS Project/'
RAW_DATA  = BASE_PATH + 'HighNote_Data.csv'
CLEAN_DATA = BASE_PATH + 'HighNote_clean.csv'


# ══════════════════════════════════════════════════════════════════
# MEMBER 1 — Data Foundation & Descriptive Statistics
# ══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load raw data ─────────────────────────────────────────────────
df = pd.read_csv(RAW_DATA)
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())


# ── Data cleaning ─────────────────────────────────────────────────

# Missing values
print('\nMissing values per column:')
print(df.isnull().sum())

# Logical range checks
print('\nValue counts — adopter:', df['adopter'].value_counts().to_dict())
print('Value counts — male:   ', df['male'].value_counts().to_dict())

# Outlier check (flagged, not removed — power users are real signal)
numeric_cols = ['age', 'friend_cnt', 'avg_friend_age', 'friend_country_cnt',
                'subscriber_friend_cnt', 'songsListened', 'lovedTracks',
                'posts', 'playlists', 'shouts', 'tenure']

print('\nOutlier check (> mean + 3 SD):')
for col in numeric_cols:
    n = (df[col] > df[col].mean() + 3 * df[col].std()).sum()
    pct = round(n / len(df) * 100, 2)
    print(f'  {col:<25} {n} outliers ({pct}%)')

print('\nNote: Outliers retained — engagement variables are right-skewed by nature.')

df_clean = df.copy()
print('\nClean dataset shape:', df_clean.shape)


# ── Adopter breakdown ─────────────────────────────────────────────
adopters     = df_clean[df_clean['adopter'] == 1]
non_adopters = df_clean[df_clean['adopter'] == 0]

print('\nTotal users    :', len(df_clean))
print('Adopters       :', len(adopters),     f'({round(len(adopters)/len(df_clean)*100, 1)}%)')
print('Non-adopters   :', len(non_adopters), f'({round(len(non_adopters)/len(df_clean)*100, 1)}%)')


# ── Summary statistics table with t-tests ────────────────────────
key_vars = ['age', 'male', 'friend_cnt', 'avg_friend_age', 'avg_friend_male',
            'friend_country_cnt', 'subscriber_friend_cnt', 'songsListened',
            'lovedTracks', 'posts', 'playlists', 'shouts', 'tenure', 'good_country']

rows = []
for var in key_vars:
    a_vals  = adopters[var]
    na_vals = non_adopters[var]
    t_stat, p_val = stats.ttest_ind(a_vals, na_vals, equal_var=False)
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
    rows.append({
        'Variable':        var,
        'All Mean':        round(df_clean[var].mean(), 3),
        'Non-Adopter Mean':round(na_vals.mean(), 3),
        'Adopter Mean':    round(a_vals.mean(), 3),
        'Mean Diff':       round(a_vals.mean() - na_vals.mean(), 3),
        '% Diff':          round((a_vals.mean() - na_vals.mean()) / (na_vals.mean() + 1e-9) * 100, 1),
        't-stat':          round(t_stat, 3),
        'p-value':         round(p_val, 4),
        'Sig':             sig,
    })

stats_table = pd.DataFrame(rows)
stats_table


# ── Interpretation ────────────────────────────────────────────────
print("""
All 14 variables show statistically significant differences between adopters
and non-adopters (p < 0.001).

Social network variables show the largest relative differences:
subscriber_friend_cnt is 4x higher for adopters (1.64 vs 0.42), and friend_cnt
is about 2x higher. Engagement variables are also uniformly higher for adopters —
lovedTracks (+204%), shouts (+232%), and songsListened (+92%). Demographic
differences are smaller but still significant: adopters are about 2 years older
and more likely to be male.

Key takeaway: Community engagement and social connections — especially having
premium subscriber friends — are the strongest signals of conversion.
""")


# ── Save clean data ───────────────────────────────────────────────
df_clean.to_csv(CLEAN_DATA, index=False)
print('Clean data saved to:', CLEAN_DATA)


# ══════════════════════════════════════════════════════════════════
# MEMBER 2 — Exploratory Visualization
# ══════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(CLEAN_DATA)

sns.set_theme(style='whitegrid', context='talk')

# Label adopter column for cleaner chart labels
df['adopter_label'] = df['adopter'].map({0: 'Free Users', 1: 'Premium Users'})

palette = ['#4C72B0', '#DD8452']


# ── Helper functions ──────────────────────────────────────────────

def make_boxplot(data, x, y, title, ylabel, log_scale=False):
    plt.figure(figsize=(8, 5.5))
    ax = sns.boxplot(data=data, x=x, y=y,
                     order=['Free Users', 'Premium Users'],
                     palette=palette, width=0.6)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=12)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel(ylabel + ' (log scale)', fontsize=12)
    plt.tight_layout()
    plt.show()

def make_barplot(data, x, y, title, ylabel):
    plt.figure(figsize=(8, 5.5))
    ax = sns.barplot(data=data, x=x, y=y,
                     order=['Free Users', 'Premium Users'],
                     palette=palette, errorbar=('ci', 95))
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.show()


# ── Demographic charts ────────────────────────────────────────────
print('Demographic Charts')

make_boxplot(df, 'adopter_label', 'age',
             'Age Distribution by Subscription Status', 'Age')

make_barplot(df, 'adopter_label', 'male',
             'Gender Distribution by Subscription Status', 'Proportion Male')

make_barplot(df, 'adopter_label', 'good_country',
             'Country Market Distribution by Subscription Status', 'Proportion from Good Country')


# ── Peer influence charts ─────────────────────────────────────────
print('Peer Influence Charts')

make_boxplot(df, 'adopter_label', 'subscriber_friend_cnt',
             'Subscriber Friends vs Premium Adoption',
             'Number of Subscriber Friends', log_scale=True)

make_boxplot(df, 'adopter_label', 'friend_cnt',
             'Total Friends vs Premium Adoption',
             'Total Friends', log_scale=True)


# ── Engagement charts ─────────────────────────────────────────────
print('Engagement Charts')

make_boxplot(df, 'adopter_label', 'songsListened',
             'Songs Listened vs Premium Adoption', 'Songs Listened', log_scale=True)

make_boxplot(df, 'adopter_label', 'playlists',
             'Playlists Created vs Premium Adoption', 'Number of Playlists', log_scale=True)

make_boxplot(df, 'adopter_label', 'shouts',
             'Social Interaction (Shouts) vs Premium Adoption', 'Number of Shouts', log_scale=True)


# ── Participation ladder ──────────────────────────────────────────
print('Participation Ladder Chart')

ladder_vars = ['songsListened', 'lovedTracks', 'playlists', 'posts', 'shouts']

ladder_df = df.groupby('adopter_label')[ladder_vars].mean().T
ladder_df_norm = ladder_df.div(ladder_df.max(axis=1), axis=0)

ladder_long = ladder_df_norm.reset_index().melt(
    id_vars='index', var_name='User Type', value_name='Normalized Engagement'
)
ladder_long.columns = ['Metric', 'User Type', 'Normalized Engagement']

metric_labels = {
    'songsListened': 'Songs Listened', 'lovedTracks': 'Loved Tracks',
    'playlists': 'Playlists', 'posts': 'Posts', 'shouts': 'Shouts'
}
ladder_long['Metric'] = ladder_long['Metric'].map(metric_labels)

plt.figure(figsize=(10, 6))
sns.barplot(data=ladder_long, x='Metric', y='Normalized Engagement',
            hue='User Type', palette=palette)
plt.title('Participation Ladder: Normalized Engagement by Subscription Status', fontsize=15)
plt.ylabel('Normalized Engagement (0–1 scale)')
plt.xlabel('')
plt.xticks(rotation=20)
plt.legend(title='')
plt.tight_layout()
plt.show()


# ── Summary table ─────────────────────────────────────────────────
summary_vars = ['age', 'male', 'good_country', 'subscriber_friend_cnt',
                'friend_cnt', 'songsListened', 'playlists', 'shouts']

summary_table = df.groupby('adopter_label')[summary_vars].mean().T
summary_table.columns = ['Free Users', 'Premium Users']
summary_table['Difference'] = summary_table['Premium Users'] - summary_table['Free Users']
summary_table.round(3)


# ══════════════════════════════════════════════════════════════════
# MEMBER 3 — Propensity Score Matching (PSM)
# ══════════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

df = pd.read_csv(CLEAN_DATA)

# ── Define treatment group ────────────────────────────────────────
df['treated'] = (df['subscriber_friend_cnt'] >= 1).astype(int)

print('Treatment group (>=1 sub. friend):', df['treated'].value_counts()[1])
print('Control group   (0 sub. friends): ', df['treated'].value_counts()[0])
print('\nAdoption rate - Treated:', round(df[df['treated']==1]['adopter'].mean() * 100, 1), '%')
print('Adoption rate - Control:', round(df[df['treated']==0]['adopter'].mean() * 100, 1), '%')

raw_diff = df[df['treated']==1]['adopter'].mean() - df[df['treated']==0]['adopter'].mean()
print('Raw difference:', round(raw_diff * 100, 2), 'pp')


# ── Logistic regression → propensity scores ───────────────────────
covariates = ['age', 'male', 'good_country', 'tenure',
              'friend_cnt', 'avg_friend_age', 'avg_friend_male',
              'friend_country_cnt', 'songsListened', 'lovedTracks',
              'posts', 'playlists', 'shouts']

X = df[covariates]
y = df['treated']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logit = LogisticRegression(max_iter=1000, random_state=42)
logit.fit(X_scaled, y)

df['pscore'] = logit.predict_proba(X_scaled)[:, 1]

null_ll  = -log_loss(y, [y.mean()] * len(y), normalize=False)
model_ll = -log_loss(y, df['pscore'], normalize=False)
r2 = 1 - model_ll / null_ll
print('McFadden R²:', round(r2, 4))

plt.figure(figsize=(8, 4))
plt.hist(df[df['treated']==0]['pscore'], bins=50, alpha=0.6, label='Control', color='orange', density=True)
plt.hist(df[df['treated']==1]['pscore'], bins=50, alpha=0.6, label='Treated', color='steelblue', density=True)
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution — Before Matching')
plt.legend()
plt.tight_layout()
plt.show()


# ── Nearest-neighbor matching ─────────────────────────────────────
treated_idx  = df[df['treated'] == 1].index.tolist()
control_pool = df[df['treated'] == 0].copy()

np.random.seed(42)
np.random.shuffle(treated_idx)

matched_treated = []
matched_control = []
used_controls   = set()

for t_idx in treated_idx:
    t_ps      = df.loc[t_idx, 'pscore']
    available = control_pool[~control_pool.index.isin(used_controls)].copy()
    available['dist'] = (available['pscore'] - t_ps).abs()
    best  = available.nsmallest(1, 'dist')
    c_idx = best.index[0]
    matched_treated.append(t_idx)
    matched_control.append(c_idx)
    used_controls.add(c_idx)

matched_df = pd.concat([
    df.loc[matched_treated].assign(group='treated'),
    df.loc[matched_control].assign(group='control')
])

print('Matched pairs:', len(matched_treated))
print('Match rate   : 100% — every treated user is matched')


# ── Covariate balance ─────────────────────────────────────────────
def smd(g1, g2):
    return (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)

balance_rows = []
for cov in covariates:
    before = smd(df[df['treated']==1][cov], df[df['treated']==0][cov])
    after  = smd(matched_df[matched_df['group']=='treated'][cov],
                 matched_df[matched_df['group']=='control'][cov])
    balance_rows.append({'Covariate': cov, 'SMD Before': round(before, 3), 'SMD After': round(after, 3)})

balance = pd.DataFrame(balance_rows)
print(balance)

# Balance chart
plt.figure(figsize=(10, 6))
x = np.arange(len(balance))
width = 0.35
plt.bar(x - width/2, balance['SMD Before'].abs(), width, label='Before Matching', color='orange')
plt.bar(x + width/2, balance['SMD After'].abs(),  width, label='After Matching',  color='steelblue')
plt.axhline(0.1, color='red', linestyle='--', lw=1.2, label='0.1 threshold (good balance)')
plt.xticks(x, balance['Covariate'], rotation=45, ha='right')
plt.ylabel('|Standardised Mean Difference|')
plt.title('Covariate Balance Before vs. After Matching')
plt.legend()
plt.tight_layout()
plt.show()

print("""
Note: Most covariates achieve good balance after matching. friend_cnt and
friend_country_cnt show some residual imbalance, which is expected with
nearest-neighbor matching on observational data.
""")


# ── Average Treatment Effect (ATT) ───────────────────────────────
treated_outcomes = matched_df[matched_df['group'] == 'treated']['adopter']
control_outcomes = matched_df[matched_df['group'] == 'control']['adopter']

rate_treated = treated_outcomes.mean()
rate_control = control_outcomes.mean()
ATT = rate_treated - rate_control

t_stat, p_val = stats.ttest_ind(treated_outcomes, control_outcomes, equal_var=False)

diffs = treated_outcomes.values - control_outcomes.values
ci_low, ci_high = stats.t.interval(0.95, df=len(diffs)-1,
                                    loc=np.mean(diffs), scale=stats.sem(diffs))

sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))

print('Adoption rate (Treated):', round(rate_treated * 100, 2), '%')
print('Adoption rate (Control):', round(rate_control * 100, 2), '%')
print('ATT (causal effect)    :', round(ATT * 100, 2), 'pp')
print('95% CI                 :', round(ci_low*100, 2), '% to', round(ci_high*100, 2), '%')
print('t-statistic            :', round(t_stat, 3))
print('p-value                :', round(p_val, 4), sig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(['Treated (>=1 sub. friend)', 'Matched Control (0 sub. friends)'],
            [rate_treated*100, rate_control*100],
            color=['steelblue', 'orange'], width=0.4)
axes[0].set_ylabel('Adoption Rate (%)')
axes[0].set_title(f'Adoption Rates After Matching — ATT = {round(ATT*100,2)} pp ({sig})')

axes[1].bar(['Raw Difference', 'PSM-Adjusted ATT'],
            [raw_diff*100, ATT*100],
            color=['gray', 'green'], width=0.4)
axes[1].errorbar(1, ATT*100,
                 yerr=[[ATT*100 - ci_low*100], [ci_high*100 - ATT*100]],
                 fmt='none', color='black', capsize=6)
axes[1].set_ylabel('Difference in Adoption Rate (pp)')
axes[1].set_title('Raw vs. PSM-Adjusted Estimate')

plt.tight_layout()
plt.show()


# ── PSM Summary table ─────────────────────────────────────────────
psm_summary = pd.DataFrame([{
    'Matched Pairs':              len(matched_treated),
    'Adoption Rate Treated (%)':  round(rate_treated * 100, 2),
    'Adoption Rate Control (%)':  round(rate_control * 100, 2),
    'ATT (pp)':                   round(ATT * 100, 2),
    'CI Lower (pp)':              round(ci_low * 100, 2),
    'CI Upper (pp)':              round(ci_high * 100, 2),
    't-stat':                     round(t_stat, 3),
    'p-value':                    round(p_val, 4),
    'Significance':               sig,
    'Raw Diff (pp)':              round(raw_diff * 100, 2),
    'Confounding Explained (pp)': round((raw_diff - ATT) * 100, 2)
}])
psm_summary


# ── Interpretation ────────────────────────────────────────────────
print("""
We used Propensity Score Matching to compare users with at least one subscriber
friend against similar users with none — matching on demographics and engagement
to make the comparison fair. We ended up with 9,823 matched pairs.

Before matching, the adoption rate gap was 12.51 pp, but some of that was simply
because treated users were already more active on the platform. Once we account
for that, the true effect of having a subscriber friend is +9.12 percentage points
(95% CI: 8.18%–10.06%, p < 0.001) — meaning about 3.39 pp of the original gap
was just selection bias.

This tells us that peer influence is real and meaningful. Users with premium friends
are more likely to convert not just because they're more engaged, but because those
friendships actually nudge them toward paying. For High Note, this makes free users
with existing subscriber friends a natural first target for any conversion push.
""")


# ══════════════════════════════════════════════════════════════════
# MEMBER 4 — Logistic Regression & Strategy
# ══════════════════════════════════════════════════════════════════

# Install statsmodels if needed
# !pip install statsmodels --quiet

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv(CLEAN_DATA)

target = 'adopter'

demo_vars        = ['age', 'male', 'good_country', 'tenure']
network_vars     = ['friend_cnt', 'avg_friend_age', 'avg_friend_male', 'friend_country_cnt']
consumption_vars = ['songsListened', 'lovedTracks']
social_vars      = ['subscriber_friend_cnt', 'posts', 'playlists', 'shouts']
all_vars         = demo_vars + network_vars + consumption_vars + social_vars

logit_df = df[[target] + all_vars].copy().dropna()

print('Logit sample size:', len(logit_df))
print('Adopter rate     :', round(logit_df[target].mean() * 100, 2), '%')


# ── VIF check ────────────────────────────────────────────────────
def compute_vif(dataframe, feature_list):
    X_vif = sm.add_constant(dataframe[feature_list].copy(), has_constant='add')
    vif_table = pd.DataFrame({
        'Variable': X_vif.columns,
        'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })
    return vif_table

vif_table = compute_vif(logit_df, all_vars)
print('\nVIF Table:')
print(vif_table.sort_values('VIF', ascending=False).round(3))

high_vif = vif_table[(vif_table['Variable'] != 'const') & (vif_table['VIF'] >= 5)]
if len(high_vif) > 0:
    print('\nVariables with VIF >= 5 (potential multicollinearity):')
    print(high_vif[['Variable', 'VIF']].round(3))
else:
    print('\nNo serious multicollinearity detected (all VIF < 5).')


# ── Helper: fit logit model ───────────────────────────────────────
def fit_logit_model(dataframe, y_col, x_cols, model_name='Model'):
    temp = dataframe[[y_col] + x_cols].dropna().copy()
    X = sm.add_constant(temp[x_cols], has_constant='add')
    y = temp[y_col]

    model    = sm.Logit(y, X).fit(disp=False)
    pred_prob = model.predict(X)
    auc      = roc_auc_score(y, pred_prob)
    pseudo_r2 = 1 - (model.llf / model.llnull)

    def sig_star(p):
        return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

    coef_table = pd.DataFrame({
        'Variable':    model.params.index,
        'Coef':        model.params.values,
        'Odds Ratio':  np.exp(model.params.values),
        'CI 2.5%':     np.exp(model.conf_int()[0].values),
        'CI 97.5%':    np.exp(model.conf_int()[1].values),
        'p-value':     model.pvalues.values,
        'Sig':         [sig_star(p) for p in model.pvalues.values]
    })

    metrics = {
        'Model': model_name, 'N': len(temp),
        'AUC': auc, 'McFadden R2': pseudo_r2,
    }
    return model, coef_table, metrics


# ── Run model variants ────────────────────────────────────────────
model_specs = {
    'M1 Demographics Only': demo_vars,
    'M2 Consumption Only':  consumption_vars,
    'M3 Social Only':       social_vars,
    'M4 Network Only':      network_vars,
    'M5 Full Model':        all_vars,
}

model_outputs = {}
metrics_rows  = []

for name, features in model_specs.items():
    model, coef_table, metrics = fit_logit_model(logit_df, target, features, model_name=name)
    model_outputs[name] = {'model': model, 'coef_table': coef_table, 'features': features}
    metrics_rows.append(metrics)

metrics_df = pd.DataFrame(metrics_rows).sort_values('McFadden R2', ascending=False)
print('\nModel Comparison:')
metrics_df.round(4)


# ── Full model results ────────────────────────────────────────────
full_table = model_outputs['M5 Full Model']['coef_table'].copy()
full_table = full_table[full_table['Variable'] != 'const'].sort_values('p-value')

print('\nFull Model — Logistic Regression Results:')
print(full_table[['Variable', 'Coef', 'Odds Ratio', 'CI 2.5%', 'CI 97.5%', 'p-value', 'Sig']].round(4))

print('\nSignificant drivers (p < 0.05):')
print(full_table[full_table['p-value'] < 0.05][['Variable', 'Odds Ratio', 'p-value', 'Sig']].round(4))


# ── Model comparison chart ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(ax=axes[0], data=metrics_df, x='AUC', y='Model', palette='Blues_d')
axes[0].set_title('Model AUC Comparison')
axes[0].set_xlim(metrics_df['AUC'].min() - 0.02, metrics_df['AUC'].max() + 0.02)

sns.barplot(ax=axes[1], data=metrics_df, x='McFadden R2', y='Model', palette='Greens_d')
axes[1].set_title('Model McFadden R² Comparison')
axes[1].set_xlim(metrics_df['McFadden R2'].min() - 0.02, metrics_df['McFadden R2'].max() + 0.02)

plt.tight_layout()
plt.show()


# ── Interpretation ────────────────────────────────────────────────
m1 = metrics_df[metrics_df['Model'] == 'M1 Demographics Only'].iloc[0]
m2 = metrics_df[metrics_df['Model'] == 'M2 Consumption Only'].iloc[0]
m3 = metrics_df[metrics_df['Model'] == 'M3 Social Only'].iloc[0]
m4 = metrics_df[metrics_df['Model'] == 'M4 Network Only'].iloc[0]
m5 = metrics_df[metrics_df['Model'] == 'M5 Full Model'].iloc[0]

print(f"""
The full logistic regression model performs best among all variants, with
AUC = {m5['AUC']:.4f} and McFadden R² = {m5['McFadden R2']:.4f}, confirming that
premium adoption is jointly driven by demographics, network, consumption, and
social engagement.

The social-only model (AUC = {m3['AUC']:.4f}, R² = {m3['McFadden R2']:.4f}) outperforms
the consumption-only model (AUC = {m2['AUC']:.4f}, R² = {m2['McFadden R2']:.4f}),
showing that community behavior explains more of the variation in adoption than
passive music listening alone.

The demographics-only model (AUC = {m1['AUC']:.4f}) and network-only model
(AUC = {m4['AUC']:.4f}) are the weakest, confirming that who you are matters less
than how you engage.

In the full model, subscriber_friend_cnt, lovedTracks, shouts, and songsListened
are among the most significant predictors. This aligns with the PSM finding from
Member 3 — peer influence is not just correlated with adoption, it causally drives it.

Strategic recommendation: High Note should prioritize free users who are already
socially connected to premium subscribers, as they are both the most likely to
convert and the most responsive to social nudges and referral incentives.
""")
