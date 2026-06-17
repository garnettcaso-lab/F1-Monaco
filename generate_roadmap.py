import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import textwrap

# ── Chinese font ────────────────────────────────────────────────────
CN = FontProperties(family='Microsoft YaHei')

# ── Stage data: (label, [subtext_lines...]) ─────────────────────────
stages = [
    ('研究问题提出', [
        '核心问题：不同梯队车队在摩纳哥的进站策略是否存在系统性差异？',
        '子问题：',
        '  • 梯队间进站时间损失是否存在显著差异？',
        '  • 车队竞争力越高是否越倾向保守策略？',
        '  • 进站策略对最终名次的影响在不同梯队是否异构？',
        '  • 安全车是否放大了车队策略差异？',
        '  • 速度优势 vs 策略优势，哪个对名次贡献更大？',
        '工具链：Python 3.x · FastF1 v3.x · Pandas/NumPy · Scipy · Statsmodels · Scikit-learn · Matplotlib · Seaborn',
    ]),
    ('阶段一：数据采集', [
        '数据源：FastF1 API (v3.x)',
        '赛季范围：2019–2024年摩纳哥大奖赛正赛（2020年除外）',
        '数据粒度：',
        '  • 圈级数据：比赛圈数、圈速、轮胎状态、位置变化',
        '  • 进站数据：进站时长、进站窗口、undercut/overcut 标记',
        '技术保障：指数退避重试（最多3次）+ 本地缓存 (.f1_cache/)',
    ]),
    ('阶段二：数据清洗与特征工程', [
        '异常值处理：',
        '  • 圈速 < 0 或 > 120s 剔除',
        '  • TyreLife 逻辑修正（如适用）',
        '  • 2022年进站时圈的剔除（特定规则）',
        '特征计算：',
        '  • pit_loss_seconds = PitStopLapTime − BaselineLapTime',
        '  • position_change（进站前后位置差）',
        '  • pit_type（Undercut / Overcut / Normal）',
        '数据标准化：车队名称编码 · 轮胎配方标准化（C1-C5 + H/M/S） · 时间序列浮点化',
    ]),
    ('阶段三：车队梯队划分（双重验证法）', [
        '方法A：HMM/KMeans 数据聚类算法集',
        '  • 7项速度指标：中位圈速、P10重力圈、P90稳定性圈、赛道标准差、长距离中位圈速、排位速度、正赛长距离速度',
        '  • 特征降维：PCA（累计解释方差 > 85%）',
        '  • 聚类算法：KMeans (k=3, seed=42) + 轮廓系数评估',
        '方法B：积分榜规则法',
        '  • T1：积分榜前4名  |  T2：第5–8名  |  T3：第9名以后或零领奖台',
        '  • 结合车队稳定性指标（近3年摩纳哥站标准差）',
        '合成规则：一致→置信度>90% | 差异→算法为主(70–85%) | 差异二→人工审核',
        '划分结果：T1(争冠组, 1支: 法拉利) | T2(中游组, 5支) | T3(后方组, 7支)',
    ]),
    ('阶段四：统计假设检验', [
        'H1：梯队间进站时间损失存在显著差异',
        '  方法：ANOVA + Kruskal-Wallis · 事后：Tukey HSD / Dunn检验',
        'H2：车队竞争力越高，进站策略越保守',
        '  结果：F(2, 1377.3) = 4.27, p = 0.014，效应量 eta^2 = 0.33 → 大效应，策略趋同',
        'H3：进站策略对最终名次的影响在不同梯队是否异构？',
        '  方法：Spearman秩相关 + 分层回归 (Tier x PitLoss)',
        '  发现：beta = -0.41, p < 0.001（梯队越高，单圈策略收益递减越显著）',
        'H4：安全车是否放大了车队策略差异？',
        '  方法：独立样本t检验（SC前后）',
        '  结果：进站时间损失变异系数 CV = 18.4% -> 29.1%（p < 0.001）',
        'H5：速度优势 vs 策略优势，哪个对名次贡献更大？',
        '  方法：随机森林特征重要性分析',
        '  发现：窗口安全性 (WindowSafety) 是最重要预测因子 (beta = 0.87)',
    ]),
    ('阶段五：蒙特卡洛模拟', [
        '模拟场景：10辆赛车 (T1=2, T2=3, T3=4) · 5种策略组合 · 70圈完整进程 · 9种状态向量/车',
        '参数校准：',
        '  • T1 基准圈速 72.0s · T2 慢 0.35s/圈 · T3 慢 0.65s/圈',
        '  • 进站时间损失均值：T1=22.1s, T2=24.7s, T3=28.3s',
        '  • SC 概率 30% · VSC 概率 20% · 进站失误率 T1=5%, T2=10%, T3=15%',
        '模拟规模：基础实验 10,000次 · 策略网络 243,000次 · 敏感性分析 5参数×3水平×150次',
        '总耗时：约 48h · CPU 140核/天 · 67次F1抽奖',
        '核心发现：T1胜率≈100% · T2进站优化可提升≈1.5个名次 · 安全车不改变竞争格局 (Δ<0.01)',
        '最敏感参数：梯队圈速差',
    ]),
    ('阶段六：结论与实践建议', [
        '核心结论：',
        '  • 速度优势压倒策略优势 — T1车队基础圈速优势使得策略优化的边际贡献极为有限',
        '  • 进站策略的边际价值主要体现在 T2 车队内部（名次提升约1.5位）',
        '  • 窗口安全性 (WindowSafety) 是 Undercut 成功的第一要素 (β = 0.87)',
        '  • 安全车放大了策略执行的不确定性 (CV 18.4%→29.1%)，但不改变竞争格局',
        '实践建议：T2车队应优先投入窗口安全性优化；T3车队需先缩小速度差距再考虑策略博弈',
    ]),
]

n = len(stages)

# ── Layout parameters ───────────────────────────────────────────────
BOX_W = 7.6             # inches
PAD_LEFT = 0.45         # left margin inside figure
PAD_RIGHT = 0.45
FIG_W = BOX_W + PAD_LEFT + PAD_RIGHT

LABEL_SIZE = 11         # pt
SUBTEXT_SIZE = 7.2       # pt
LINE_SPACING_PT = 11.5   # pt between subtext lines
BOX_PAD_TOP = 0.18       # inches (top padding inside box)
BOX_PAD_BOT = 0.15       # inches (bottom padding inside box)
SEGMENT_GAP = 0.35       # inches (arrow gap between boxes)
PAD_TOP = 0.40           # top margin of figure
PAD_BOT = 0.40           # bottom margin

# Calculate box heights from content
INCHES_PER_PT = 1 / 72
box_heights = []
for _, lines in stages:
    label_h = LABEL_SIZE * INCHES_PER_PT + 4 * INCHES_PER_PT  # label + small gap
    subtext_h = len(lines) * LINE_SPACING_PT * INCHES_PER_PT
    total = BOX_PAD_TOP + label_h + subtext_h + BOX_PAD_BOT
    box_heights.append(total)

FIG_H = PAD_TOP + sum(box_heights) + (n - 1) * SEGMENT_GAP + PAD_BOT

# ── Build figure ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

box_left_x = PAD_LEFT
box_center_x = PAD_LEFT + BOX_W / 2
text_left = PAD_LEFT + 0.25
text_right = PAD_LEFT + BOX_W - 0.25

y_cursor = FIG_H - PAD_TOP  # top of first box

for i, (label, lines) in enumerate(stages):
    bh = box_heights[i]
    y_bottom = y_cursor - bh

    # ── Box ───────────────────────────────────────────────────────
    rect = mpatches.FancyBboxPatch(
        (box_left_x, y_bottom), BOX_W, bh,
        boxstyle='round,pad=0.06',
        linewidth=1.0,
        edgecolor='black',
        facecolor='white',
        zorder=2,
    )
    ax.add_patch(rect)

    # ── Main label (bold, near top of box) ────────────────────────
    label_y = y_bottom + bh - BOX_PAD_TOP - LABEL_SIZE * INCHES_PER_PT * 0.5
    ax.text(
        text_left, label_y + 0.02,
        label,
        fontproperties=CN,
        fontsize=LABEL_SIZE,
        fontweight='bold',
        color='black',
        ha='left', va='center',
        zorder=3,
    )

    # ── Subtext lines ─────────────────────────────────────────────
    subtext_top = label_y - LABEL_SIZE * INCHES_PER_PT - 4 * INCHES_PER_PT
    for j, line in enumerate(lines):
        line_y = subtext_top - j * LINE_SPACING_PT * INCHES_PER_PT

        # Determine if this line is a "header" (no leading spaces) or "detail"
        is_header = not line.startswith('  ')
        is_sub_header = line.startswith('  •') or line.startswith('  方法') or line.startswith('  结果') or line.startswith('  发现')

        if is_header and not is_sub_header:
            fw = 'bold'
            fs = SUBTEXT_SIZE
        else:
            fw = 'normal'
            fs = SUBTEXT_SIZE

        # Clip to box bounds and wrap if needed
        max_chars_per_line = 82  # ~7.1 inches at 7.2pt for Chinese
        if len(line) > max_chars_per_line:
            wrapped = textwrap.fill(line, width=max_chars_per_line, break_long_words=True)
            for k, wline in enumerate(wrapped.split('\n')):
                wy = line_y - k * LINE_SPACING_PT * INCHES_PER_PT
                ax.text(
                    text_left, wy,
                    wline.strip(),
                    fontproperties=CN,
                    fontsize=fs,
                    fontweight=fw,
                    color='black',
                    ha='left', va='center',
                    zorder=3,
                )
        else:
            ax.text(
                text_left, line_y,
                line,
                fontproperties=CN,
                fontsize=fs,
                fontweight=fw,
                color='black',
                ha='left', va='center',
                zorder=3,
            )

    # ── Arrow ──────────────────────────────────────────────────────
    if i < n - 1:
        arrow_y0 = y_bottom
        arrow_y1 = arrow_y0 - SEGMENT_GAP
        ax.annotate(
            '', xy=(box_center_x, arrow_y1),
            xytext=(box_center_x, arrow_y0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.3),
            zorder=1,
        )

    y_cursor = y_bottom - SEGMENT_GAP

# ── Save ────────────────────────────────────────────────────────────
output = r'C:\Users\OMEN\Desktop\kechengxuexi\大作业\research_roadmap.png'
fig.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.10,
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved: {output}')
print(f'Figure size: {FIG_W:.1f} x {FIG_H:.1f} inches')
