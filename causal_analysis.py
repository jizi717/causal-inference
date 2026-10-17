# -*- coding: utf-8 -*-
"""
增值服务对客户留存的因果效应评估（A/B 对比 → 分层检验 → 逻辑回归控制 → 倾向得分匹配）

数据集：IBM Telco Customer Churn（7,043 名真实客户）
业务问题：订购「在线安全/技术支持」等增值服务，是否真的能降低流失？
          直接用组间流失率差异判断会受混淆变量干扰（订购者往往本身就是在网老客户），
          因此用分层分析 + 回归控制 + 倾向得分匹配（PSM）逐层逼近因果效应。

统计方法：卡方检验（含 Cramér's V 效应量）、分层分析（排除 Simpson 悖论）、
         逻辑回归控制混淆、PSM 最近邻匹配、标准化均数差（SMD）平衡性检验、
         效应量与功效分析。
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(BASE), "churn-prediction", "data", "Telco-Customer-Churn.csv")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)
RNG = 42
rng = np.random.default_rng(RNG)


# ---------------------------------------------------------------- 1. 数据
def load():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # 处理变量 T：是否订购增值服务（在线安全 或 技术支持）
    df["T"] = ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")).astype(int)
    # 仅保留开通互联网的客户（未开通者无法订购该服务，属于不适用人群）
    df = df[df["InternetService"] != "No"].copy()

    # 协变量
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_two_year"] = (df["Contract"] == "Two year").astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    df["is_senior"] = df["SeniorCitizen"].astype(int)
    df["has_partner"] = (df["Partner"] == "Yes").astype(int)
    df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
    df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df["is_female"] = (df["gender"] == "Female").astype(int)
    svc = ["OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
    df["n_other_services"] = (df[svc] == "Yes").sum(axis=1)
    print(f"[1] 样本: {len(df):,} 名互联网客户，处理组 {df['T'].sum():,}，"
          f"对照组 {int((1 - df['T']).sum()):,}")
    return df


COVS = ["tenure", "MonthlyCharges", "TotalCharges", "is_fiber", "is_month_to_month",
        "is_two_year", "is_electronic_check", "is_autopay", "is_senior",
        "has_partner", "has_dependents", "is_paperless", "is_female", "n_other_services"]


# ---------------------------------------------------------------- 2. 朴素对比 + 卡方检验
def naive_compare(df):
    a = df[df["T"] == 1]["Churn"]     # 处理组
    b = df[df["T"] == 0]["Churn"]     # 对照组
    table = np.array([[a.sum(), len(a) - a.sum()],
                      [b.sum(), len(b) - b.sum()]])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.sum()
    cramer_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    # 两组比例差的 95% 置信区间
    p1, p2 = a.mean(), b.mean()
    se = np.sqrt(p1 * (1 - p1) / len(a) + p2 * (1 - p2) / len(b))
    diff = p2 - p1                      # 对照组 - 处理组（正数表示处理组流失更低）
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    # 相对风险与需治疗人数 NNT
    rr = p1 / p2
    nnt = 1 / diff if diff > 0 else np.nan

    res = {
        "处理组流失率": round(float(p1), 4),
        "对照组流失率": round(float(p2), 4),
        "流失率差异(对照-处理)": round(float(diff), 4),
        "差异95%置信区间": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "卡方统计量": round(float(chi2), 3),
        "p值": float(f"{p:.3e}"),
        "CramersV效应量": round(float(cramer_v), 4),
        "相对风险RR": round(float(rr), 4),
        "NNT需服务人数": round(float(nnt), 2),
    }
    print("\n[2] 朴素组间对比")
    for k, v in res.items():
        print(f"    {k}: {v}")
    return res, table


# ---------------------------------------------------------------- 3. 分层分析（Simpson 悖论检验）
def stratified(df):
    df = df.copy()
    df["tenure_bin"] = pd.cut(df["tenure"], [0, 6, 12, 24, 48, 72],
                              labels=["0-6月", "7-12月", "13-24月", "25-48月", "49-72月"],
                              include_lowest=True)
    rows = []
    for b, g in df.groupby("tenure_bin", observed=True):
        a, c = g[g["T"] == 1]["Churn"], g[g["T"] == 0]["Churn"]
        if len(a) < 20 or len(c) < 20:
            continue
        t = np.array([[a.sum(), len(a) - a.sum()], [c.sum(), len(c) - c.sum()]])
        chi2, p, _, _ = stats.chi2_contingency(t)
        rows.append({"在网时长": str(b), "处理组流失率": round(float(a.mean()), 4),
                     "对照组流失率": round(float(c.mean()), 4),
                     "差异": round(float(c.mean() - a.mean()), 4),
                     "p值": float(f"{p:.4g}"), "处理组n": len(a), "对照组n": len(c)})
    s = pd.DataFrame(rows)
    print("\n[3] 按在网时长分层（检验结论在各层是否一致）")
    print(s.to_string(index=False))

    # 加权平均（Mantel-Haenszel 近似：按对照组样本量加权）
    w = s["对照组n"].values
    weighted = float(np.average(s["差异"].values, weights=w))
    print(f"    加权平均差异: {weighted:.4f}（与朴素差异对比，判断是否存在 Simpson 悖论）")
    return s, weighted


# ---------------------------------------------------------------- 4. 逻辑回归控制混淆
def regression_adjust(df):
    X = df[COVS].copy()
    X["T"] = df["T"].values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=3000, C=1.0, random_state=RNG)
    lr.fit(Xs, df["Churn"].values)
    coef_t = float(lr.coef_[0][list(X.columns).index("T")])
    odds_ratio = float(np.exp(coef_t))

    # Bootstrap 置信区间
    ors = []
    for _ in range(200):
        idx = rng.choice(len(X), size=len(X), replace=True)
        try:
            m = LogisticRegression(max_iter=1000, random_state=RNG).fit(
                Xs[idx], df["Churn"].values[idx])
            ors.append(np.exp(m.coef_[0][list(X.columns).index("T")]))
        except Exception:
            pass
    ci = (float(np.percentile(ors, 2.5)), float(np.percentile(ors, 97.5)))

    res = {"T的回归系数(标准化)": round(coef_t, 4),
           "优势比OR": round(odds_ratio, 4),
           "OR的95%置信区间": [round(ci[0], 4), round(ci[1], 4)],
           "Bootstrap次数": len(ors)}
    print("\n[4] 逻辑回归控制混淆变量后的处理效应")
    for k, v in res.items():
        print(f"    {k}: {v}")
    return res


# ---------------------------------------------------------------- 5. 倾向得分匹配 PSM
def psm(df):
    X = df[COVS].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    ps_model = LogisticRegression(max_iter=3000, random_state=RNG).fit(Xs, df["T"].values)
    ps = ps_model.predict_proba(Xs)[:, 1]
    d = df[["Churn", "T"] + COVS].copy()
    d["ps"] = ps

    # 共同支持域
    lo = max(d[d["T"] == 1]["ps"].min(), d[d["T"] == 0]["ps"].min())
    hi = min(d[d["T"] == 1]["ps"].max(), d[d["T"] == 0]["ps"].max())
    d = d[(d["ps"] >= lo) & (d["ps"] <= hi)].copy()
    print(f"\n[5] 倾向得分匹配（共同支持域 [{lo:.3f}, {hi:.3f}]，剩余 {len(d):,} 人）")

    # 平衡性检验：标准化均数差 SMD
    def smd(col):
        a, b = d.loc[d["T"] == 1, col], d.loc[d["T"] == 0, col]
        sd = np.sqrt((a.var() + b.var()) / 2)
        return abs(a.mean() - b.mean()) / sd if sd > 0 else 0.0

    cols = COVS + ["ps"]
    before = {c: round(float(smd(c)), 4) for c in cols}

    # 最近邻 1:1 匹配：全局按倾向得分最近邻，caliper=0.05 剔除过远匹配
    treat = d[d["T"] == 1]
    ctrl = d[d["T"] == 0]
    nn = NearestNeighbors(n_neighbors=1).fit(ctrl[["ps"]].values)
    dist, idx = nn.kneighbors(treat[["ps"]].values)
    keep = dist.flatten() <= 0.05
    matched_t = treat.iloc[keep]
    matched_c = ctrl.iloc[idx.flatten()[keep]]
    matched = pd.concat([matched_t, matched_c])
    print(f"    匹配成功: {len(matched_t):,} 对（最近邻 1:1，caliper=0.05，"
          f"剔除 {int((~keep).sum()):,} 个无合格匹配样本）")

    after = {c: round(float(abs(matched.loc[matched['T'] == 1, c].mean() -
                                matched.loc[matched['T'] == 0, c].mean()) /
                            max(np.sqrt((matched.loc[matched['T'] == 1, c].var() +
                                         matched.loc[matched['T'] == 0, c].var()) / 2), 1e-9)), 4)
             for c in cols}

    balance = pd.DataFrame({"匹配前SMD": before, "匹配后SMD": after})
    balance["是否平衡(<0.1)"] = (balance["匹配后SMD"] < 0.1).map({True: "是", False: "否"})
    print("\n    协变量平衡性检验（SMD < 0.1 视为平衡）")
    print(balance.to_string())

    # ATT：匹配后处理效应
    mt = matched[matched["T"] == 1]["Churn"]
    mc = matched[matched["T"] == 0]["Churn"]
    att = float(mc.mean() - mt.mean())
    se = np.sqrt(mt.mean() * (1 - mt.mean()) / len(mt) + mc.mean() * (1 - mc.mean()) / len(mc))
    t_stat, p_val = stats.ttest_ind(mc, mt)

    res = {
        "匹配对数": int(len(matched_t)),
        "匹配后处理组流失率": round(float(mt.mean()), 4),
        "匹配后对照组流失率": round(float(mc.mean()), 4),
        "ATT(匹配后处理效应)": round(att, 4),
        "ATT的95%置信区间": [round(att - 1.96 * se, 4), round(att + 1.96 * se, 4)],
        "t统计量": round(float(t_stat), 4),
        "p值": float(f"{p_val:.4g}"),
        "平衡协变量占比": f"{int((balance['匹配后SMD'] < 0.1).sum())}/{len(balance)}",
    }
    print("\n    匹配后处理效应 ATT")
    for k, v in res.items():
        print(f"      {k}: {v}")
    return res, balance, d


# ---------------------------------------------------------------- 6. 功效分析
def power_analysis(naive_res, n_treat, n_ctrl):
    p1 = naive_res["处理组流失率"]
    p2 = naive_res["对照组流失率"]
    # Cohen's h 效应量
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    n_needed = int(np.ceil(((1.96 + 0.84) / h) ** 2 * 2))   # alpha=0.05, power=0.8
    res = {"Cohen's h 效应量": round(float(h), 4),
           "每组所需最小样本量(alpha=0.05, power=0.8)": n_needed,
           "实际处理组样本量": int(n_treat),
           "实际对照组样本量": int(n_ctrl),
           "统计功效是否充足": "是" if min(n_treat, n_ctrl) >= n_needed else "否"}
    print("\n[6] 功效分析")
    for k, v in res.items():
        print(f"    {k}: {v}")
    return res


# ---------------------------------------------------------------- 7. 绘图
def plot(df, naive_res, strat_df, psm_res, balance):
    # 组间流失率对比
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    d = [naive_res["处理组流失率"], naive_res["对照组流失率"]]
    ax[0].bar(["处理组\n(订购增值服务)", "对照组\n(未订购)"], d,
              color=["#16a085", "#c0392b"], alpha=.9)
    for i, v in enumerate(d):
        ax[0].text(i, v + .01, f"{v:.1%}", ha="center", fontweight="bold")
    ax[0].set_ylabel("流失率"); ax[0].set_title("朴素组间对比（未控制混淆）")
    ax[0].set_ylim(0, max(d) * 1.25)

    # 分层
    x = np.arange(len(strat_df))
    ax[1].plot(x, strat_df["处理组流失率"], "o-", color="#16a085", label="处理组")
    ax[1].plot(x, strat_df["对照组流失率"], "s-", color="#c0392b", label="对照组")
    ax[1].set_xticks(x); ax[1].set_xticklabels(strat_df["在网时长"], rotation=15)
    ax[1].set_ylabel("流失率"); ax[1].set_title("按在网时长分层后的流失率")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "group_comparison.png"), dpi=150); plt.close()

    # SMD 平衡图
    plt.figure(figsize=(7.5, 5))
    b = balance.drop(index=["ps"]) if "ps" in balance.index else balance
    y = np.arange(len(b))
    plt.scatter(b["匹配前SMD"], y, s=55, color="#c0392b", label="匹配前")
    plt.scatter(b["匹配后SMD"], y, s=55, color="#16a085", label="匹配后")
    plt.axvline(0.1, color="gray", ls="--", lw=1, label="平衡阈值 0.1")
    plt.yticks(y, b.index); plt.xlabel("标准化均数差 SMD")
    plt.title("PSM 协变量平衡性（Love Plot）"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "smd_balance.png"), dpi=150); plt.close()

    # 效应对比
    plt.figure(figsize=(7, 4.4))
    labels = ["朴素差异", "分层加权", f"PSM(ATT={psm_res['ATT(匹配后处理效应)']:.3f})"]
    vals = [naive_res["流失率差异(对照-处理)"], None, psm_res["ATT(匹配后处理效应)"]]
    plt.bar([0, 2], [vals[0], vals[2]], color=["#95a5a6", "#2c3e50"], width=.6)
    plt.xticks([0, 2], [labels[0], labels[2]])
    plt.ylabel("流失率降低幅度"); plt.title("混淆控制前后：处理效应变化")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "effect_comparison.png"), dpi=150); plt.close()


def main():
    df = load()
    naive_res, table = naive_compare(df)
    strat_df, weighted = stratified(df)
    reg_res = regression_adjust(df)
    psm_res, balance, matched_d = psm(df)
    power = power_analysis(naive_res, int(df["T"].sum()), int((1 - df["T"]).sum()))
    plot(df, naive_res, strat_df, psm_res, balance)

    metrics = {"朴素对比": naive_res, "分层加权差异": round(weighted, 4),
               "回归控制": reg_res, "倾向得分匹配": psm_res, "功效分析": power}
    with open(os.path.join(OUT, "causal_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    strat_df.to_csv(os.path.join(OUT, "stratified_result.csv"), index=False, encoding="utf-8-sig")
    balance.to_csv(os.path.join(OUT, "balance_result.csv"), encoding="utf-8-sig")
    print(f"\n全部结果已输出至: {OUT}")


if __name__ == "__main__":
    main()
