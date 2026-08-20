#!/usr/bin/env python3
"""Regenerate simulation charts with CJK font — standalone, no simulation re-run needed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

# ═══════════════════════════════════════════════════════
# Font setup must happen before pyplot import
# ═══════════════════════════════════════════════════════
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

fonts = {f.name for f in fm.fontManager.ttflist}
CN = next((n for n in ['Microsoft YaHei', 'SimHei', 'Source Han Sans SC'] if n in fonts), None)

# Pre-configure before pyplot exists
matplotlib.rcParams['font.sans-serif'] = [CN] if CN else ['sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Seaborn v0.12+ supports font in set_theme
if CN:
    sns.set_theme(style='whitegrid', font=CN, rc={'axes.unicode_minus': False})
else:
    sns.set_theme(style='whitegrid')

# Force re-apply after seaborn (seaborn may reorder sans-serif list)
plt.rcParams['font.sans-serif'] = [CN, 'DejaVu Sans'] if CN else ['DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

print(f'Chart font: {CN}')
print(f'rcParams font.sans-serif: {plt.rcParams["font.sans-serif"]}')

# ═══════════════════════════════════════════════════════
IN_DIR = Path('./simulation_results')
TIER_CLR = {'T1': '#E63946', 'T2': '#2A9D8F', 'T3': '#457B9D'}

def load_data():
    df = pd.read_parquet(IN_DIR / 'simulation_base.parquet')
    grid = IN_DIR / 'simulation_grid.parquet'
    df_g = pd.read_parquet(grid) if grid.exists() else pd.DataFrame()
    return df, df_g

def fig1_win_rate(df):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7))
    for i, t in enumerate(['T1','T2','T3']):
        d = df[df['tier']==t]['position'].value_counts(normalize=True).sort_index()
        ax0.bar(d.index+i*0.25, d.values, 0.22, color=TIER_CLR[t], alpha=0.85, label=t)
    ax0.set_xlabel('完赛名次'); ax0.set_ylabel('概率')
    ax0.set_title('名次分布 (分梯队)'); ax0.legend(); ax0.grid(axis='y',alpha=0.3)

    x=np.arange(3); w=0.3
    for j,(sc,lb) in enumerate([(True,'有安全车'),(False,'无安全车')]):
        sub = df[df['has_sc']==sc] if sc else df[~df['has_sc']]
        means=[sub[sub['tier']==t]['position'].mean() for t in ['T1','T2','T3']]
        ax1.bar(x+j*w,means,w,alpha=0.85,label=lb)
    ax1.set_xticks(x+w/2); ax1.set_xticklabels(['T1','T2','T3'])
    ax1.set_ylabel('平均名次 (越小越好)')
    ax1.set_title('安全车对平均名次的影响'); ax1.legend(); ax1.grid(axis='y',alpha=0.3)
    return fig

def fig2_strategy_heatmap(df_g):
    if df_g.empty or 'pit_w1' not in df_g.columns: return None
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for i, t in enumerate(['T1','T2','T3']):
        ax = axes[i]; td = df_g[(df_g['tier']==t)&df_g['pit_w1'].notna()]
        if td.empty: continue
        piv = td.pivot_table(values='position',index='pit_w1',columns='pit_w2',aggfunc='mean')
        sns.heatmap(piv,annot=True,fmt='.1f',cmap='RdYlGn_r',center=5,ax=ax,cbar_kws={'label':'平均名次'})
        ax.set_title(f'{t} — 策略效果')
        ax.set_xlabel('二停窗口'); ax.set_ylabel('一停窗口')
    fig.suptitle('进站窗口策略热力图 (颜色=平均名次)',fontsize=14,fontweight='bold')
    fig.tight_layout(); return fig

def fig3_marginal(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, t in enumerate(['T1','T2','T3']):
        ax = axes[i]; td = df[df['tier']==t].copy()
        bins = np.arange(0, 120, 5)
        td['gb'] = pd.cut(td['gap_to_winner'], bins, labels=bins[1:])
        mp = td.groupby('gb',observed=False)['position'].agg(['mean','std','count'])
        vd = mp[mp['count']>5]; xv = vd.index.astype(float)
        ax.errorbar(xv,vd['mean'],yerr=vd['std']/np.sqrt(vd['count']),fmt='o-',capsize=3,color=TIER_CLR[t],lw=2)
        ax.set_xlabel('与冠军时间差 (秒)'); ax.set_ylabel('平均名次')
        ax.set_title(f'{t} — 时间差距边际效应'); ax.grid(alpha=0.3)
    fig.suptitle('时间差距对名次的边际效应',fontsize=14,fontweight='bold')
    fig.tight_layout(); return fig

def fig4_radar(df):
    cats = ['胜率','均速','一致性','SC抗性','进站可靠性','策略弹性']
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True))
    angles = np.linspace(0,2*np.pi,len(cats),endpoint=False).tolist()+[0]
    for tier in ['T1','T2','T3']:
        td = df[df['tier']==tier]
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
    fig.tight_layout(); return fig

def main():
    print('Loading data...')
    df, df_g = load_data()
    print(f'{len(df)} rows, {df["sim_id"].nunique()} sims')

    for fname, desc, func, arg in [
        ('fig_win_rate.png','胜率分布',fig1_win_rate,df),
        ('fig_strategy_heatmap.png','策略热力图',fig2_strategy_heatmap,df_g),
        ('fig_marginal_effect.png','边际效应',fig3_marginal,df),
        ('fig_radar_strategy.png','雷达图',fig4_radar,df),
    ]:
        try:
            fig = func(arg)
            if fig:
                fig.savefig(IN_DIR/fname, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f'  [OK] {fname}')
        except Exception as e:
            print(f'  [FAIL] {fname}: {e}')

    print(f'\nCharts saved to {IN_DIR.resolve()}/')

if __name__ == '__main__':
    main()
