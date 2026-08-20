#!/usr/bin/env python3
"""Regenerate ALL charts across stages 4-6 with proper CJK font rendering."""
import sys; import json; from pathlib import Path

# ══════════════════════════════════════════════════════
# Font setup — absolute first thing before any imports
# ══════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
fonts = {f.name for f in fm.fontManager.ttflist}
CN = next((n for n in ['Microsoft YaHei','SimHei','Source Han Sans SC'] if n in fonts), None)
if CN:
    matplotlib.rcParams['font.sans-serif'] = [CN]
    matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np; import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if CN:
    sns.set_theme(style='whitegrid', font=CN, rc={'axes.unicode_minus': False})
    plt.rcParams['font.sans-serif'] = [CN, 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    print(f'Font: {CN}')
else:
    sns.set_theme(style='whitegrid')

TIER_CLR = {'T1':'#E63946','T2':'#2A9D8F','T3':'#457B9D'}
EN = {'Red Bull':'红牛','Ferrari':'法拉利','Mercedes':'梅赛德斯','McLaren':'迈凯伦',
       'Aston Martin':'阿斯顿马丁','Alpine':'阿尔派','Williams':'威廉姆斯',
       'AlphaTauri':'小红牛','Alfa Romeo':'阿尔法罗密欧','Haas':'哈斯',
       'Racing Point':'赛点','Renault':'雷诺','Toro Rosso':'红牛二队','RB':'小红牛','Kick Sauber':'索伯'}

# ══════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════
pits = pd.read_parquet('cleaned/pit_stops.parquet')
pits['Team'] = pits['Team'].map(EN).fillna(pits['Team'])
tiers = pd.read_excel('tier_analysis/tier_results.xlsx', sheet_name='梯队结果')
tl = tiers[['Year','Team','FinalTier']].rename(columns={'Year':'RaceYear'})
pits = pits.merge(tl, on=['RaceYear','Team'], how='left').dropna(subset=['FinalTier'])
pits_ok = pits[pits['PitDuration_sec']<200].copy()
pits_ok['TierCode'] = pits_ok['FinalTier'].map({'T1':1,'T2':2,'T3':3})
df = pits_ok  # shorthand

# Stage 4 data
try:
    sim_base = pd.read_parquet('simulation_results/simulation_base.parquet')
    sim_grid = pd.read_parquet('simulation_results/simulation_grid.parquet')
except:
    sim_base = pd.DataFrame(); sim_grid = pd.DataFrame()

# ══════════════════════════════════════════════════════
# Stage 4: Descriptive Analysis Charts
# ══════════════════════════════════════════════════════
OUT4 = Path('./statistics')
OUT4.mkdir(exist_ok=True)

# --- pit_loss violin + trend ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
valid = df[df['PitLoss_sec'].notna()]
ax = axes[0]
parts = ax.violinplot([valid[valid['FinalTier']==t]['PitLoss_sec'].dropna() for t in ['T1','T2','T3']],
                       positions=[0,1,2], showmeans=True, showmedians=True)
for i,pc in enumerate(parts['bodies']):
    pc.set_facecolor(TIER_CLR[['T1','T2','T3'][i]]); pc.set_alpha(0.7)
for pn in ('cbars','cmins','cmaxes','cmeans','cmedians'):
    if pn in parts: parts[pn].set_color('#333')
ax.set_xticks([0,1,2]); ax.set_xticklabels(['T1(争冠组)','T2(中游组)','T3(后方组)'],fontsize=12)
ax.set_ylabel('进站时间损失 (秒)',fontsize=13); ax.set_title('pit_loss 梯队分布对比',fontsize=14,fontweight='bold'); ax.grid(axis='y',alpha=0.3)

ax = axes[1]
years = sorted(valid['RaceYear'].unique())
for tier, pos in [('T1',0),('T2',1),('T3',2)]:
    means, stds, yrs = [], [], []
    for yr in years:
        yd = valid[(valid['RaceYear']==yr)&(valid['FinalTier']==tier)]['PitLoss_sec']
        if len(yd)>=2: yrs.append(int(yr)); means.append(yd.mean()); stds.append(yd.std())
    if yrs:
        ax.errorbar(yrs,means,yerr=stds,marker='o',ms=8,lw=2,capsize=5,color=TIER_CLR[tier],
                    label=f'{tier}(争冠组)' if tier=='T1' else f'{tier}(中游组)' if tier=='T2' else f'{tier}(后方组)')
ax.set_xlabel('赛季',fontsize=13); ax.set_ylabel('pit_loss 均值 (秒)',fontsize=13)
ax.set_title('pit_loss 年度变化趋势',fontsize=14,fontweight='bold'); ax.legend(fontsize=10); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT4/'fig2_pit_loss_comparison.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print('[OK] fig2_pit_loss_comparison.png')

# --- Position change ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ax = axes[0]
cats = ['上升','不变','下降']; x = np.arange(3); w = 0.25
for i, t in enumerate(['T1','T2','T3']):
    sub = df[df['FinalTier']==t]
    if len(sub)==0: continue
    pc = sub['PositionChange'].dropna()
    counts = [int((pc>0).sum()),int((pc==0).sum()),int((pc<0).sum())]
    pct = [100*c/max(sum(counts),1) for c in counts]
    bars = ax.bar(x+i*w,counts,w,color=TIER_CLR[t],alpha=0.85,label=t)
    for j,(c,p) in enumerate(zip(counts,pct)):
        if c>0: ax.text(x[j]+i*w,c+0.3,f'{p:.0f}%',ha='center',fontsize=8,fontweight='bold')
ax.set_xticks(x+w); ax.set_xticklabels(cats,fontsize=12)
ax.set_ylabel('进站次数',fontsize=13); ax.set_title('进站位置得失分布 (分梯队)',fontsize=14,fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y',alpha=0.3)

ax = axes[1]
bp = ax.boxplot([df[df['FinalTier']==t]['PositionChange'].dropna() for t in ['T1','T2','T3']],
                 positions=[0,1,2], widths=0.5, patch_artist=True, showfliers=True, showmeans=True,
                 meanprops=dict(marker='D',markerfacecolor='red',markersize=6))
for i,box in enumerate(bp['boxes']):
    box.set_facecolor(TIER_CLR[['T1','T2','T3'][i]]); box.set_alpha(0.7)
ax.axhline(y=0,color='gray',ls='--',lw=1,alpha=0.6)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['T1(争冠组)','T2(中游组)','T3(后方组)'],fontsize=12)
ax.set_ylabel('位置变化 (正值=上升)',fontsize=13)
ax.set_title('位置变化分布 (箱线图)',fontsize=14,fontweight='bold'); ax.grid(axis='y',alpha=0.3)
fig.tight_layout(); fig.savefig(OUT4/'fig3_position_change.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print('[OK] fig3_position_change.png')

# --- Pit loss yearly trend ---
fig, ax = plt.subplots(figsize=(12, 7))
years = sorted(df['RaceYear'].unique())
for tier in ['T1','T2','T3']:
    means, ci_l, ci_h, yrs = [], [], [], []
    for yr in years:
        yd = df[(df['RaceYear']==yr)&(df['FinalTier']==tier)]['PitLoss_sec']
        if len(yd)>=3:
            m=yd.mean(); se=yd.std()/np.sqrt(len(yd)); ci=1.96*se
            yrs.append(int(yr)); means.append(m); ci_l.append(m-ci); ci_h.append(m+ci)
    if yrs:
        ax.plot(yrs,means,marker='s',ms=10,lw=2.5,color=TIER_CLR[tier],label=tier)
        ax.fill_between(yrs,ci_l,ci_h,alpha=0.15,color=TIER_CLR[tier])
ax.set_xlabel('赛季',fontsize=13); ax.set_ylabel('进站时间损失 (秒)',fontsize=13)
ax.set_title('pit_loss 年度变化趋势 (均值 +- 95% CI)',fontsize=14,fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT4/'fig6_pit_loss_yearly_trend.png',dpi=200,bbox_inches='tight'); plt.close(fig)
print('[OK] fig6_pit_loss_yearly_trend.png')

# ══════════════════════════════════════════════════════
# Stage 5: Hypothesis Testing Charts
# ══════════════════════════════════════════════════════
OUT5 = Path('./hypothesis_results'); OUT5.mkdir(exist_ok=True)

g1=df[df['FinalTier']=='T1']['LapIn']; g2=df[df['FinalTier']=='T2']['LapIn']; g3=df[df['FinalTier']=='T3']['LapIn']

# --- H3 boxplot ---
fig, ax = plt.subplots(figsize=(8,5))
bp=ax.boxplot([g1,g2,g3],patch_artist=True,widths=0.4)
for p,c in zip(bp['boxes'],['#E63946','#2A9D8F','#457B9D']): p.set_facecolor(c); p.set_alpha(0.7)
ax.set_xticklabels(['T1(争冠组)','T2(中游组)','T3(后方组)'],fontsize=12)
ax.set_ylabel('进站圈号',fontsize=13); ax.set_title('H3: 进站窗口梯队对比',fontsize=14,fontweight='bold')
fig.savefig(OUT5/'h3_lapin.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print('[OK] h3_lapin.png')

# --- H4 regression diagnostics ---
h4=df[['PositionChange','PitLoss_sec','FinalTier']].dropna()
h4['isT2']=(h4['FinalTier']=='T2').astype(int); h4['isT3']=(h4['FinalTier']=='T3').astype(int)
h4['plT2']=h4['PitLoss_sec']*h4['isT2']; h4['plT3']=h4['PitLoss_sec']*h4['isT3']
y=h4['PositionChange']; X=sm.add_constant(h4[['PitLoss_sec','isT2','isT3','plT2','plT3']])
m2=sm.OLS(y,X).fit(); r=m2.resid; fv=m2.fittedvalues
fig,axes=plt.subplots(2,2,figsize=(12,10))
axes[0,0].scatter(fv,r,alpha=0.5); axes[0,0].axhline(0,color='r',ls='--')
axes[0,0].set_xlabel('拟合值'); axes[0,0].set_ylabel('残差'); axes[0,0].set_title('残差 vs 拟合值')
qqplot(r,line='s',ax=axes[0,1]); axes[0,1].set_title('Q-Q 图')
axes[1,0].hist(r,bins=20,edgecolor='white'); axes[1,0].set_title('残差分布')
axes[1,1].scatter(range(len(r)),r,alpha=0.5,s=10); axes[1,1].axhline(0,color='r',ls='--')
axes[1,1].set_title('残差序列')
fig.tight_layout(); fig.savefig(OUT5/'h4_diagnostics.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print('[OK] h4_diagnostics.png')

# --- H5 SC effect ---
h5=pits[['FinalTier','PitLoss_sec','LapIn','has_sc']].copy() if 'has_sc' in pits.columns else pits[['FinalTier','PitLoss_sec','LapIn']].copy()
if 'has_sc' not in h5.columns: h5['has_sc'] = False
fig,ax=plt.subplots(figsize=(10,6))
x=np.arange(3); w=0.3
for i,(sc,lbl) in enumerate([(False,'正常'),(True,'有安全车')]):
    means=[h5[(h5['FinalTier']==t)&(h5['has_sc']==sc)]['PitLoss_sec'].mean() for t in ['T1','T2','T3']]
    errs=[h5[(h5['FinalTier']==t)&(h5['has_sc']==sc)]['PitLoss_sec'].std() for t in ['T1','T2','T3']]
    ax.bar(x+i*w,means,w,yerr=errs,capsize=5,alpha=0.85,label=lbl)
ax.set_xticks(x+w/2); ax.set_xticklabels(['T1(争冠组)','T2(中游组)','T3(后方组)'])
ax.set_ylabel('pit_loss (秒)',fontsize=13); ax.set_title('H5: 安全车效应(分梯队)',fontsize=14,fontweight='bold')
ax.legend(fontsize=11); ax.grid(axis='y',alpha=0.3)
fig.tight_layout(); fig.savefig(OUT5/'h5_sc_effect.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print('[OK] h5_sc_effect.png')

# --- Yearly trend ---
fig,ax=plt.subplots(figsize=(12,7))
for t,c in zip(['T1','T2','T3'],['#E63946','#2A9D8F','#457B9D']):
    tp=pits[pits['FinalTier']==t]
    ym=tp.groupby('RaceYear')['PitLoss_sec'].agg(['mean','std','count'])
    if len(ym)>=2:
        ci=1.96*ym['std']/np.sqrt(ym['count'])
        ax.errorbar(ym.index,ym['mean'],yerr=ci,marker='o',ms=8,lw=2,capsize=5,color=c,label=t)
ax.set_xlabel('赛季',fontsize=13); ax.set_ylabel('pit_loss 均值 (秒)',fontsize=13)
ax.set_title('pit_loss 年度趋势 (均值 +- 95% CI)',fontsize=14,fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT5/'yearly_trend.png',dpi=150,bbox_inches='tight'); plt.close(fig)
print('[OK] yearly_trend.png')

# ══════════════════════════════════════════════════════
# Stage 6: Simulation Charts (read data, replot)
# ══════════════════════════════════════════════════════
OUT6 = Path('./simulation_results')
if not sim_base.empty:
    # fig_win_rate
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7))
    for i, tier in enumerate(['T1','T2','T3']):
        d = sim_base[sim_base['tier']==tier]['position'].value_counts(normalize=True).sort_index()
        ax0.bar(d.index+i*0.25, d.values, 0.22, color=TIER_CLR[tier], alpha=0.85, label=tier)
    ax0.set_xlabel('完赛名次'); ax0.set_ylabel('概率')
    ax0.set_title('名次分布 (分梯队)'); ax0.legend(); ax0.grid(axis='y',alpha=0.3)
    x=np.arange(3); w=0.3
    for j,(sc,lb) in enumerate([(True,'有安全车'),(False,'无安全车')]):
        sub = sim_base[sim_base['has_sc']==sc] if sc else sim_base[~sim_base['has_sc']]
        means=[sub[sub['tier']==t]['position'].mean() for t in ['T1','T2','T3']]
        ax1.bar(x+j*w,means,w,alpha=0.85,label=lb)
    ax1.set_xticks(x+w/2); ax1.set_xticklabels(['T1(争冠组)','T2(中游组)','T3(后方组)'])
    ax1.set_ylabel('平均名次 (越小越好)'); ax1.set_title('安全车对平均名次的影响')
    ax1.legend(); ax1.grid(axis='y',alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT6/'fig_win_rate.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    print('[OK] sim_fig_win_rate.png')

    # fig_strategy_heatmap
    if not sim_grid.empty and 'pit_w1' in sim_grid.columns:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        for i, tier in enumerate(['T1','T2','T3']):
            td = sim_grid[(sim_grid['tier']==tier)&sim_grid['pit_w1'].notna()]
            if td.empty: continue
            piv = td.pivot_table(values='position',index='pit_w1',columns='pit_w2',aggfunc='mean')
            sns.heatmap(piv,annot=True,fmt='.1f',cmap='RdYlGn_r',center=5,ax=axes[i],cbar_kws={'label':'平均名次'})
            axes[i].set_title(f'{tier} — 策略效果'); axes[i].set_xlabel('二停窗口'); axes[i].set_ylabel('一停窗口')
        fig.suptitle('进站窗口策略热力图',fontsize=14,fontweight='bold')
        fig.tight_layout(); fig.savefig(OUT6/'fig_strategy_heatmap.png',dpi=200,bbox_inches='tight'); plt.close(fig)
        print('[OK] sim_fig_strategy_heatmap.png')

    # fig_marginal_effect
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, tier in enumerate(['T1','T2','T3']):
        td = sim_base[sim_base['tier']==tier].copy()
        bins = np.arange(0, 120, 5)
        td['gb'] = pd.cut(td['gap_to_winner'], bins, labels=bins[1:])
        mp = td.groupby('gb',observed=False)['position'].agg(['mean','std','count'])
        vd = mp[mp['count']>5]; xv = vd.index.astype(float)
        axes[i].errorbar(xv,vd['mean'],yerr=vd['std']/np.sqrt(vd['count']),fmt='o-',capsize=3,color=TIER_CLR[tier],lw=2)
        axes[i].set_xlabel('与冠军时间差 (秒)'); axes[i].set_ylabel('平均名次')
        axes[i].set_title(f'{tier} — 时间差距边际效应'); axes[i].grid(alpha=0.3)
    fig.suptitle('时间差距对名次的边际效应',fontsize=14,fontweight='bold')
    fig.tight_layout(); fig.savefig(OUT6/'fig_marginal_effect.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    print('[OK] sim_fig_marginal_effect.png')

    # fig_radar
    cats = ['胜率','均速','一致性','SC抗性','进站可靠性','策略弹性']
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True))
    angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()+[0]
    for tier in ['T1','T2','T3']:
        td = sim_base[sim_base['tier']==tier]
        if td.empty: continue
        wr=(td['position']==1).mean(); sp=1/(td['total_time'].mean()/5000+0.001)
        cn=1/(td['total_time'].std()/50+0.001)
        sc_r=1.0
        if 'has_sc' in td.columns:
            a=td[td['has_sc']]['position'].mean(); b=td[~td['has_sc']]['position'].mean()
            sc_r=b/max(a,0.001)
        rel=1-td['pit_errors'].mean()/max(td['pit_errors'].max(),1)
        ela=td['gap_to_winner'].std()/max(td['gap_to_winner'].std(),0.001)
        vals=np.array([wr,sp,cn,sc_r,rel,ela]); vals=vals/max(vals.max(),0.001)
        vp=vals.tolist()+[vals[0]]
        ax.fill(angles,vp,alpha=0.2,color=TIER_CLR[tier])
        ax.plot(angles,vp,lw=2.5,color=TIER_CLR[tier],label=tier)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=11)
    ax.set_title('梯队策略能力雷达图',fontsize=14,fontweight='bold',pad=25)
    ax.legend(loc='upper right',fontsize=11,bbox_to_anchor=(1.3,1.1))
    fig.tight_layout(); fig.savefig(OUT6/'fig_radar_strategy.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    print('[OK] sim_fig_radar_strategy.png')

print(f'\nAll charts regenerated with font: {CN}')
print(f'  statistics/     — Stage 4 (descriptive)')
print(f'  hypothesis_results/ — Stage 5 (hypothesis)')
print(f'  simulation_results/ — Stage 6 (monte carlo)')
