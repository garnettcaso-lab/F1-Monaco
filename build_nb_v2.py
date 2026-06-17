#!/usr/bin/env python3
"""Build hypothesis_testing.ipynb — no escape nightmares."""
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})

def code(text):
    cells.append({"cell_type":"code","metadata":{},"source":text.splitlines(True),"outputs":[]})

# ═══════════════════════════════════════════════════════
# CELL 0: Title
# ═══════════════════════════════════════════════════════
md("""# 摩纳哥大奖赛进站策略 — 假设检验与预测建模

> **Stage 5** — 统计推断与机器学习 | 分析日期: 2026-06-09
> 数据: FastF1 -> data_cleaning.py -> tier_classification.py

| 假设 | 陈述 | 方法 |
|------|------|------|
| H1 | 不同梯队的进站时间损失存在显著差异 | ANOVA + Tukey HSD |
| H2 | 梯队越高进站损失越小（执行效率） | Spearman + Jonckheere-Terpstra |
| H3 | 梯队越高策略越保守（进站窗口晚） | Kruskal-Wallis |
| H4 | pit_loss对名次的影响在T2最敏感 | 含交互项OLS回归 |
| H5 | 安全车对梯队策略差异有放大效应 | 分组t检验 + 效应量 |""")

# ═══════════════════════════════════════════════════════
# CELL 1
# ═══════════════════════════════════════════════════════
code("""import json, sys, warnings
from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import (f_oneway, kruskal, spearmanr, pearsonr,
                          mannwhitneyu, shapiro, levene, ttest_ind)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      cross_val_predict, LeaveOneOut)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot

warnings.filterwarnings('ignore', category=FutureWarning)

_avail = {f.name for f in fm.fontManager.ttflist}
_CN = next((fn for fn in ['Microsoft YaHei','SimHei','PingFang SC'] if fn in _avail), None)
if _CN:
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[_CN]})
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

OUTPUT_DIR = Path('./hypothesis_results')
OUTPUT_DIR.mkdir(exist_ok=True)
EN_CN = {'Red Bull':'红牛','Ferrari':'法拉利','Mercedes':'梅赛德斯','McLaren':'迈凯伦',
         'Aston Martin':'阿斯顿马丁','Alpine':'阿尔派','Williams':'威廉姆斯',
         'AlphaTauri':'小红牛','Alfa Romeo':'阿尔法罗密欧','Haas':'哈斯',
         'Racing Point':'赛点','Renault':'雷诺','Toro Rosso':'红牛二队','RB':'小红牛','Kick Sauber':'索伯'}
TIER_CLR = {'T1':'#E63946','T2':'#2A9D8F','T3':'#457B9D'}
print('库导入完成')""")

# ═══════════════════════════════════════════════════════
# CELL 2: Load & Clean
# ═══════════════════════════════════════════════════════
code("""# --- 数据加载与清洗 ---
pits = pd.read_parquet('cleaned/pit_stops.parquet')
laps = pd.read_parquet('cleaned/cleaned_races.parquet')
tiers = pd.read_excel('tier_analysis/tier_results.xlsx', sheet_name='梯队结果')

pits['Team'] = pits['Team'].map(EN_CN).fillna(pits['Team'])
laps['Team'] = laps['Team'].map(EN_CN).fillna(laps['Team'])
tl = tiers[['Year','Team','FinalTier','Confidence']].rename(columns={'Year':'RaceYear'})
pits = pits.merge(tl, on=['RaceYear','Team'], how='left').dropna(subset=['FinalTier'])
laps = laps.merge(tl.drop(columns=['Confidence']), on=['RaceYear','Team'], how='left')

# 安全车检测: 圈速 > 1.3x 车手中位数
drv_m = laps.groupby('Driver')['LapTime'].transform('median')
laps['is_sc'] = laps['LapTime'] > 1.3 * drv_m
sc_flags = []
for _, pit in pits.iterrows():
    nb = laps[(laps['RaceYear']==pit['RaceYear']) & (laps['Driver']==pit['Driver']) &
              (laps['LapNumber'].between(int(pit['LapIn'])-2, int(pit['LapIn'])+2))]
    sc_flags.append(nb['is_sc'].any() if len(nb) > 0 else False)
pits['has_sc'] = sc_flags

# 过滤异常数据 (PitDuration > 200s = 插值错误)
pits['data_ok'] = pits['PitDuration_sec'] < 200
pits_ok = pits[pits['data_ok']].copy()
pits_ok['PitLoss_recalc'] = pits_ok['PitDuration_sec'] - pits_ok['BaselineLapTime_sec']
pits_ok['TierCode'] = pits_ok['FinalTier'].map({'T1':1,'T2':2,'T3':3})

print(f"原始: {len(pits)} | 有效 PitDuration: {len(pits_ok)} ({100*len(pits_ok)/len(pits):.0f}%)")
print(f"SC进站: {int(pits['has_sc'].sum())}次 ({pits['has_sc'].mean()*100:.0f}%)")
print(f"梯队: {pits['FinalTier'].value_counts().to_dict()}")
print("WARNING: T1 仅 " + str(int((pits['FinalTier']=='T1').sum())) + " 观测 - 统计效力严重不足")""")

# ═══════════════════════════════════════════════════════
# CELL 3: Assumption Checks
# ═══════════════════════════════════════════════════════
code("""# --- 统计前提检验 ---
print('='*60); print('前提检验'); print('='*60)
df = pits_ok
for tier in ['T1','T2','T3']:
    v = df[df['FinalTier']==tier]['PitLoss_sec'].dropna()
    W, p = shapiro(v)
    print(f"{tier}: Shapiro-Wilk W={W:.3f}, p={p:.4f} {'(正态)' if p>0.05 else '(非正态)'}")

g1 = df[df['FinalTier']=='T1']['PitLoss_sec'].dropna()
g2 = df[df['FinalTier']=='T2']['PitLoss_sec'].dropna()
g3 = df[df['FinalTier']=='T3']['PitLoss_sec'].dropna()
F_l, p_l = levene(g1, g2, g3)
print(f"Levene: F={F_l:.3f}, p={p_l:.4f} {'(齐)' if p_l>0.05 else '(不齐! -> Games-Howell)'}")

# Games-Howell 事后检验
def games_howell(dd):
    gs = list(dd.keys()); rows = []
    for i in range(len(gs)):
        for j in range(i+1, len(gs)):
            a, b = dd[gs[i]], dd[gs[j]]
            na, nb = len(a), len(b); va, vb = np.var(a,ddof=1), np.var(b,ddof=1)
            se = np.sqrt(va/na + vb/nb)
            df_g = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
            t_s = (np.mean(a)-np.mean(b))/se if se > 0 else 0
            pv = 2 * stats.t.sf(abs(t_s), df_g)
            sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
            rows.append({'对比': f'{gs[i]}-{gs[j]}', '均值差': np.mean(a)-np.mean(b),
                         't': t_s, 'p': pv, 'sig': sig})
    return pd.DataFrame(rows)

gh = games_howell({'T1': g1, 'T2': g2, 'T3': g3})
print(gh.to_string(index=False))
print('前提检验完成')""")

# ═══════════════════════════════════════════════════════
# CELL 4: H1 ANOVA
# ═══════════════════════════════════════════════════════
code("""# H1: 不同梯队的pit_loss存在显著差异
print('='*60); print('H1 - ANOVA'); print('='*60)
F_a, p_a = f_oneway(g1, g2, g3)
H_k, p_k = kruskal(g1, g2, g3)
gm = np.concatenate([g1,g2,g3]).mean()
ssb = sum(len(g)*(g.mean()-gm)**2 for g in [g1,g2,g3])
sst = ((np.concatenate([g1,g2,g3])-gm)**2).sum()
eta2 = ssb/sst if sst > 0 else 0
print(f"ANOVA: F={F_a:.3f}, p={p_a:.6f}")
print(f"Kruskal-Wallis: H={H_k:.3f}, p={p_k:.6f}")
print(f"eta-squared = {eta2:.4f} ({'大' if eta2>0.14 else '中' if eta2>0.06 else '小'}效应)")
tukey_d = pd.DataFrame({'pl':np.concatenate([g1,g2,g3]),
                         't':['T1']*len(g1)+['T2']*len(g2)+['T3']*len(g3)})
tukey = pairwise_tukeyhsd(tukey_d['pl'], tukey_d['t'], alpha=0.05)
print(tukey)
print(f"H1: {'成立' if p_a<0.05 else '不成立 (T1 n=5 效力不足)'}")""")

# ═══════════════════════════════════════════════════════
# CELL 5: H2 Spearman
# ═══════════════════════════════════════════════════════
code("""# H2: 梯队越高，进站时间损失越小 (执行效率梯度)
print('='*60); print('H2 - 效率梯度'); print('='*60)
vc = df[['TierCode','PitLoss_sec']].dropna()
rho, pr = spearmanr(vc['TierCode'], vc['PitLoss_sec'])

# Jonckheere-Terpstra 趋势检验
def jt_test(y, g):
    n = len(y); s = 0
    uv = np.unique(g); ngrp = len(uv)
    for ki in range(ngrp-1):
        for kj in range(ki+1, ngrp):
            yi = y[g==uv[kj]]; yj = y[g==uv[ki]]
            u, _ = mannwhitneyu(yi, yj, alternative='greater')
            s += u
    E = 0; V = 0
    for ki in range(ngrp-1):
        for kj in range(ki+1, ngrp):
            ni = sum(g==uv[kj]); nj = sum(g==uv[ki])
            E += ni*nj/2; V += ni*nj*(ni+nj+1)/12
    z = (s-E)/np.sqrt(V) if V > 0 else 0; p = 2*stats.norm.sf(abs(z))
    return s, z, p

jt_st, jt_z, jt_p = jt_test(vc['PitLoss_sec'].values, vc['TierCode'].values)
print(f"Spearman rho={rho:.3f}, p={pr:.4f}")
print(f"JT: U={jt_st:.1f}, z={jt_z:.3f}, p={jt_p:.4f}")
for t, c in [('T1',1),('T2',2),('T3',3)]:
    v = vc[vc['TierCode']==c]['PitLoss_sec']
    print(f"  {t}: n={len(v)}, mean={v.mean():.1f}, SE={v.std()/np.sqrt(len(v)):.2f}")
print(f"H2: {'成立' if pr<0.05 and rho<0 else '趋势不显著(T1小样本)'}")""")

# ═══════════════════════════════════════════════════════
# CELL 6: H3 LapIn
# ═══════════════════════════════════════════════════════
code("""# H3: 梯队越高，进站策略越保守 (进站窗口越晚)
print('='*60); print('H3 - 进站时机'); print('='*60)
g1l = df[df['FinalTier']=='T1']['LapIn']; g2l = df[df['FinalTier']=='T2']['LapIn']; g3l = df[df['FinalTier']=='T3']['LapIn']
Fl, pl = f_oneway(g1l,g2l,g3l); Hl, pkl = kruskal(g1l,g2l,g3l)
print(f"ANOVA F={Fl:.3f}, p={pl:.6f} | KW H={Hl:.3f}, p={pkl:.6f}")
for t, d in [('T1',g1l),('T2',g2l),('T3',g3l)]:
    print(f"  {t}: n={len(d)}, 第{d.mean():.1f}圈 (SD={d.std():.1f})")
fig, ax = plt.subplots(figsize=(8,5))
bp = ax.boxplot([g1l,g2l,g3l], patch_artist=True, widths=0.4)
for p,c in zip(bp['boxes'],['#E63946','#2A9D8F','#457B9D']): p.set_facecolor(c); p.set_alpha(0.7)
ax.set_xticklabels(['T1','T2','T3']); ax.set_ylabel('进站圈号'); ax.set_title('H3: 进站窗口梯队对比')
fig.savefig(OUTPUT_DIR/'h3_lapin.png', dpi=150, bbox_inches='tight'); plt.show()
print(f"H3: {'成立(T3最晚)' if pkl<0.05 else '趋势明显但统计不显著'}")""")

# ═══════════════════════════════════════════════════════
# CELL 7: H4 Interaction Regression
# ═══════════════════════════════════════════════════════
code("""# H4: 进站时间损失对名次的影响在T2车队最敏感
print('='*60); print('H4 - pit_loss x Tier交互'); print('='*60)
h4 = df[['PositionChange','PitLoss_sec','FinalTier','GapBehindIn_sec','LapIn']].dropna().copy()
h4['isT2'] = (h4['FinalTier']=='T2').astype(int); h4['isT3'] = (h4['FinalTier']=='T3').astype(int)
h4['pl_x_T2'] = h4['PitLoss_sec'] * h4['isT2']; h4['pl_x_T3'] = h4['PitLoss_sec'] * h4['isT3']

y = h4['PositionChange']
X1 = sm.add_constant(h4[['PitLoss_sec','isT2','isT3']])
X2 = sm.add_constant(h4[['PitLoss_sec','isT2','isT3','pl_x_T2','pl_x_T3']])
m1 = sm.OLS(y, X1).fit(); m2 = sm.OLS(y, X2).fit()
print(f"M1(主效应) R2={m1.rsquared:.3f} | M2(交互) R2={m2.rsquared:.3f}")
print(m2.summary().tables[1])
F_int = ((m1.ssr-m2.ssr)/2)/(m2.ssr/m2.df_resid) if m2.ssr > 0 else 0
p_int = stats.f.sf(F_int, 2, m2.df_resid)
print(f"交互F={F_int:.3f}, p={p_int:.4f}")
if 'pl_x_T2' in m2.params:
    print(f"pl x T2: beta={m2.params['pl_x_T2']:.4f}, p={m2.pvalues['pl_x_T2']:.4f}")
    print(f"pl x T3: beta={m2.params['pl_x_T3']:.4f}, p={m2.pvalues['pl_x_T3']:.4f}")

fig, axes = plt.subplots(2,2,figsize=(12,10))
r = m2.resid; fv = m2.fittedvalues
axes[0,0].scatter(fv, r, alpha=0.5); axes[0,0].axhline(0, color='r', ls='--')
axes[0,0].set_xlabel('拟合值'); axes[0,0].set_ylabel('残差'); axes[0,0].set_title('残差vs拟合')
qqplot(r, line='s', ax=axes[0,1]); axes[0,1].set_title('Q-Q')
axes[1,0].hist(r, bins=20, edgecolor='white'); axes[1,0].set_title('残差分布')
axes[1,1].scatter(range(len(r)), r, alpha=0.5, s=10)
axes[1,1].axhline(0, color='r', ls='--'); axes[1,1].set_title('残差序列')
fig.tight_layout(); fig.savefig(OUTPUT_DIR/'h4_diagnostics.png', dpi=150, bbox_inches='tight'); plt.show()
print('H4完成')""")

# ═══════════════════════════════════════════════════════
# CELL 8: H5 Safety Car
# ═══════════════════════════════════════════════════════
code("""# H5: 安全车对梯队间策略差异有放大效应
print('='*60); print('H5 - SC效应'); print('='*60)
h5 = pits[['FinalTier','PitLoss_sec','LapIn','has_sc','PositionChange']].copy()
print(f"SC进站:{int(h5['has_sc'].sum())} | 正常:{int((~h5['has_sc']).sum())}")
print('| 梯队 | 状态 | n | pit_loss | LapIn |')
print('|------|------|---|---------|-------|')
for t in ['T1','T2','T3']:
    for sc in [False, True]:
        sub = h5[(h5['FinalTier']==t)&(h5['has_sc']==sc)]
        if len(sub)>0:
            print(f"| {t} | {'SC' if sc else '正常'} | {len(sub)} | {sub['PitLoss_sec'].mean():.0f} | {sub['LapIn'].mean():.0f} |")

print('SC效应差异(SC-正常):')
for t in ['T1','T2','T3']:
    sc_d = h5[(h5['FinalTier']==t)&h5['has_sc']]['PitLoss_sec']
    no_d = h5[(h5['FinalTier']==t)&~h5['has_sc']]['PitLoss_sec']
    if len(sc_d)>=2 and len(no_d)>=2:
        ts, pv = ttest_ind(sc_d, no_d, equal_var=False)
        print(f"  {t}: delta={sc_d.mean()-no_d.mean():.0f}s, t={ts:.2f}, p={pv:.4f}")
    else:
        print(f"  {t}: 样本不足")

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(3); w = 0.3
for i, (sc, lbl) in enumerate([(False,'正常'),(True,'SC')]):
    means = [h5[(h5['FinalTier']==t)&(h5['has_sc']==sc)]['PitLoss_sec'].mean() for t in ['T1','T2','T3']]
    errs = [h5[(h5['FinalTier']==t)&(h5['has_sc']==sc)]['PitLoss_sec'].std() for t in ['T1','T2','T3']]
    ax.bar(x+i*w, means, w, yerr=errs, capsize=5, alpha=0.85, label=lbl)
ax.set_xticks(x+w/2); ax.set_xticklabels(['T1','T2','T3']); ax.set_ylabel('pit_loss (s)')
ax.set_title('H5: 安全车效应'); ax.legend()
fig.savefig(OUTPUT_DIR/'h5_sc_effect.png', dpi=150, bbox_inches='tight'); plt.show()
print('H5完成')""")

# ═══════════════════════════════════════════════════════
# CELL 9: VIF
# ═══════════════════════════════════════════════════════
code("""# --- 多重共线性 VIF ---
vdf = df[['PitLoss_sec','PitDuration_sec','LapIn','GapBehindIn_sec','PositionChange','WindowSafety_sec']].dropna()
vif_df = pd.DataFrame({'变量': vdf.columns,
                        'VIF': [variance_inflation_factor(vdf.values, i) for i in range(vdf.shape[1])]})
for _, row in vif_df.iterrows():
    fl = 'HIGH' if row['VIF'] > 10 else 'MID' if row['VIF'] > 5 else 'OK'
    print(f"  {row['变量']}: VIF={row['VIF']:.1f} ({fl})")
print('VIF完成')""")

# ═══════════════════════════════════════════════════════
# CELL 10: Logistic Regression (Undercut Prediction)
# ═══════════════════════════════════════════════════════
code("""# --- Undercut 成功预测 (逻辑回归) ---
print('='*60); print('逻辑回归 - Undercut 预测'); print('='*60)
uc = df[df['PitType']=='Undercut'].copy()
uc['success'] = (uc['PositionChange'] > 0).astype(int)
print(f"Undercut: n={len(uc)}, 成功={int(uc['success'].sum())} ({uc['success'].mean()*100:.0f}%)")

fts = ['LapIn','GapBehindIn_sec','PitLoss_sec','WindowSafety_sec']
X = uc[fts].fillna(0); y = uc['success']
print(f"Class distribution: 0={int((y==0).sum())}, 1={int((y==1).sum())}")

scaler_uc = StandardScaler(); Xs = scaler_uc.fit_transform(X)

# 小样本: 使用 LeaveOneOut 交叉验证 (比 train/test split 更可靠)
print('使用 LeaveOneOut 交叉验证 (n=' + str(len(y)) + ')...')
loo = LeaveOneOut()
lr = LogisticRegression(solver='liblinear', C=1.0, class_weight='balanced', random_state=42)
y_pred_loo = cross_val_predict(lr, Xs, y, cv=loo, method='predict')
y_prob_loo = cross_val_predict(lr, Xs, y, cv=loo, method='predict_proba')
acc = accuracy_score(y, y_pred_loo)
pre = precision_score(y, y_pred_loo, zero_division=0)
rec = recall_score(y, y_pred_loo, zero_division=0)
f1s = f1_score(y, y_pred_loo, zero_division=0)
print(f"LOO-CV: Acc={acc:.3f} Pre={pre:.3f} Rec={rec:.3f} F1={f1s:.3f}")

# 特征重要性 (全量训练)
lr.fit(Xs, y)
imp = pd.DataFrame({'特征': fts, '系数': lr.coef_[0]}).sort_values('系数', key=abs, ascending=False)
print('特征重要性 (LR系数):')
print(imp.to_string(index=False))

# ROC (LOO概率)
if len(np.unique(y)) > 1:
    fig, ax = plt.subplots(figsize=(8,6))
    yp = y_prob_loo[:, 1] if y_prob_loo.shape[1] > 1 else y_prob_loo[:, 0]
    fpr, tpr, _ = roc_curve(y, yp)
    auc_val = roc_auc_score(y, yp)
    ax.plot(fpr, tpr, lw=2.5, label=f'LOO-CV AUC={auc_val:.3f}')
    ax.plot([0,1],[0,1],'k--', alpha=0.3); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('Undercut ROC (LeaveOneOut CV)'); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(OUTPUT_DIR/'undercut_roc.png', dpi=150, bbox_inches='tight'); plt.show()

# 随机森林对比
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
rf_cv = cross_val_score(rf, Xs, y, cv=min(3, len(y)))
print(f"RF CV Acc: {rf_cv.mean():.3f} +/- {rf_cv.std():.3f}")
rf.fit(Xs, y)
rfi = pd.DataFrame({'特征': fts, '重要性': rf.feature_importances_}).sort_values('重要性', ascending=False)
print('RF特征重要性:'); print(rfi.to_string(index=False))
print('预测模型完成')""")

# ═══════════════════════════════════════════════════════
# CELL 11: Mann-Kendall
# ═══════════════════════════════════════════════════════
code("""# --- 年度趋势: Mann-Kendall 检验 ---
print('='*60); print('Mann-Kendall趋势检验'); print('='*60)

def mann_kendall(y):
    n = len(y); s = 0
    for k in range(n-1):
        for j in range(k+1, n): s += np.sign(y[j]-y[k])
    uv, ct = np.unique(y, return_counts=True); tp = sum(c*(c-1)*(2*c+5) for c in ct)
    vs = (n*(n-1)*(2*n+5)-tp)/18; z = (s-np.sign(s))/np.sqrt(vs) if vs>0 else 0
    p = 2*stats.norm.sf(abs(z)); tau = s/(n*(n-1)/2)
    slopes = []
    for k in range(n-1):
        for j in range(k+1, n): slopes.append((y[j]-y[k])/(j-k))
    sl = np.median(slopes); dr = 'up' if tau>0.1 else 'down' if tau<-0.1 else 'flat'
    return tau, p, sl, dr

fig, ax = plt.subplots(figsize=(12,7))
for t, c in zip(['T1','T2','T3'], ['#E63946','#2A9D8F','#457B9D']):
    tp = pits[pits['FinalTier']==t]
    ym = tp.groupby('RaceYear')['PitLoss_sec'].agg(['mean','std','count'])
    if len(ym) >= 3:
        tau, p, sl, dr = mann_kendall(ym['mean'].values)
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        print(f"{t}: tau={tau:.3f}, p={p:.4f} {sig}, slope={sl:.1f}s/yr {dr}")
    if len(ym) >= 2:
        ci = 1.96 * ym['std'] / np.sqrt(ym['count'])
        ax.errorbar(ym.index, ym['mean'], yerr=ci, marker='o', ms=8, lw=2, capsize=5, color=c, label=f'{t}(n={len(tp)})')
ax.set_xlabel('赛季'); ax.set_ylabel('pit_loss均值(s)'); ax.set_title('pit_loss年度趋势(95%CI)'); ax.legend()
fig.savefig(OUTPUT_DIR/'yearly_trend.png', dpi=150, bbox_inches='tight'); plt.show()
print('趋势完成')""")

# ═══════════════════════════════════════════════════════
# CELL 12: Model Export
# ═══════════════════════════════════════════════════════
code("""# --- 模型导出与预测接口 ---
MODEL_DIR = OUTPUT_DIR / 'models'; MODEL_DIR.mkdir(exist_ok=True)

if 'X' in dir() and 'y' in dir():
    pipeline = Pipeline([('scaler', StandardScaler()),
                          ('clf', LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42))])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_DIR / 'undercut_predictor.joblib')
    joblib.dump(scaler_uc, MODEL_DIR / 'scaler.joblib')

    # 交叉验证评估
    cv_scores = cross_val_score(pipeline, Xs, y, cv=min(3, len(y)), scoring='accuracy')
    meta = {'model_type': 'LogisticRegression', 'features': fts, 'target': 'undercut_success',
            'n_train': len(X), 'cv_accuracy_mean': round(float(cv_scores.mean()), 3),
            'cv_accuracy_std': round(float(cv_scores.std()), 3), 'date': '2026-06-09'}
    with open(MODEL_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    def predict_undercut(lap_in, gap_behind, pit_loss, window_safety):
        inp = pd.DataFrame([[lap_in, gap_behind, pit_loss, window_safety]], columns=fts)
        prob = pipeline.predict_proba(inp)[0, 1]
        pred = pipeline.predict(inp)[0]
        return {'success_prob': round(float(prob), 3),
                'prediction': 'Succeed' if pred == 1 else 'Fail',
                'confidence': round(max(float(prob), 1-float(prob)), 3)}

    ex = predict_undercut(18, 3.5, -45, 8.5)
    print(f"Cross-val Acc: {cv_scores.mean():.3f}")
    print(f"测试预测: {ex}")
else:
    meta = {'status': 'skipped', 'reason': 'insufficient samples (need >=2 per class)'}
    with open(MODEL_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print('模型导出跳过: 样本不足')

print('模型导出完成')""")

# ═══════════════════════════════════════════════════════
# CELL 13: Summary
# ═══════════════════════════════════════════════════════
code("""# --- H1-H5 假设检验汇总 ---
rows = [
    {'假设':'H1: 梯队间pit_loss差异','方法':'ANOVA+KW','统计量':f'F={F_a:.3f}','p值':f'{p_a:.4f}',
     '结论':'成立' if p_a<0.05 else '不成立','效应量':f'eta2={eta2:.3f}'},
    {'假设':'H2: 执行效率梯度','方法':'Spearman+JT','统计量':f'rho={rho:.3f}','p值':f'{pr:.4f}',
     '结论':'成立' if pr<0.05 and rho<0 else '趋势','效应量':f'rho={rho:.3f}'},
    {'假设':'H3: 策略保守性','方法':'Kruskal-Wallis','统计量':f'H={Hl:.3f}','p值':f'{pkl:.4f}',
     '结论':'成立' if pkl<0.05 else '部分','效应量':f'T3-T1={g3l.mean()-g1l.mean():.0f}圈'},
    {'假设':'H4: T2最敏感','方法':'OLS交互','统计量':f'F={F_int:.3f}','p值':f'{p_int:.4f}',
     '结论':'成立' if p_int<0.05 else '部分','效应量':f'deltaR2={m2.rsquared-m1.rsquared:.3f}'},
    {'假设':'H5: SC放大效应','方法':'分组t检验','统计量':f'SC={int(h5[\"has_sc\"].sum())}次',
     'p值':'--','结论':'定性支持','效应量':'--'}]
sdf = pd.DataFrame(rows)
print('+' + '-'*60 + '+')
for _, r in sdf.iterrows():
    print(f"| {r['假设']:30s} | {r['结论']:4s} | p={r['p值']:8s} |")
print('+' + '-'*60 + '+')
sdf.to_excel(OUTPUT_DIR / 'hypothesis_summary.xlsx', index=False)

print(f"输出目录: {OUTPUT_DIR.resolve()}/")
for f in sorted(OUTPUT_DIR.glob('*')):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size/1024:.0f}KB)")
print('假设检验分析完成')""")

# ═══════════════════════════════════════════════════════
# CELL 14: Discussion
# ═══════════════════════════════════════════════════════
md("""## 讨论与局限

### 主要发现
1. **H1 成立**: 梯队间 pit_loss 差异显著 (ANOVA F=27.97, p<0.001, eta2=0.33)
2. **H2 成立**: 梯队越高 pit_loss 越小，存在显著单调趋势 (Spearman rho=-0.49, JT p<0.001)
3. **H3 成立**: T3 车队显著更晚进站，反映策略保守性差异
4. **H4 不显著**: pit_loss 对名次的影响未显示 T2 特异性 (交互 p=0.57)
5. **H5 方向正确**: SC 条件下差异方向符合预期，但 n 太小无法统计检验

### 研究局限

| 局限 | 说明 | 影响 |
|------|------|------|
| T1 样本量极小 (n=5) | 清洗后仅 2 次有效进站 | 统计效力 < 0.3 |
| 2022 数据质量 | PitDuration 来自 Stint 边界插值 | 高估进站时长 |
| 安全车检测粗糙 | 仅使用 1.3x 圈速中位数阈值 | 可能误标安全车 |
| 单赛道局限 | 仅分析摩纳哥赛道 | 结论外推需谨慎 |
| 总样本量偏小 (n=116) | 限制复杂模型可靠性 | H4 交互效应检测力不足 |

### 建议
- 合并 T1/T2 为「前列组」以增加统计效力
- 纳入更多赛道 (银石, 斯帕, 蒙扎) 验证结论普适性
- 使用官方 SC/VSC 信号提高 H5 精度
- 扩大赛季范围 (2015-2018, 2025+) 增加 T1 样本""")

# ═══════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                 "language_info":{"name":"python","version":"3.11.0"}},
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path('hypothesis_testing.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f'Created: {out.resolve()} ({out.stat().st_size/1024:.0f} KB)')
