"""
AI 퀀트 모델 8단계 - 변경 3 (마지막): 펀딩비를 코인별 "평소 대비"로
** 모의고사만, 수능은 절대 안 씀 **

바꾸는 것 하나
  funding (그대로 %)  ->  funding_rel = 지금 펀딩비 - 그 코인의 직전 30일 펀딩비 중앙값
  직전 30일 = 지금 시각은 빼고 과거만 (미래 정보 없음)
  과거가 7일(168시간)보다 짧으면 빈칸 -> 그 시간은 모든 예보관 채점에서 같이 뺌 (공정 비교)
  나머지 힌트(vol_4h, vol_24h, rsi, oi_1h, oi_24h)와 설정은 전부 그대로

판정 (결과 보기 전 확정, 이전과 같음): 로지스틱 v3 vs 변동성 예보관
  ① AUC 차이 95% 범위 > 0   ② 향상도 차이 95% 범위 > 0   둘 다
참고 (판정 아님)
  처음 보는 코인에서의 손해(v1: -0.034)가 줄었나 -> 펀딩비가 진짜 범인이었는지
  트리 v3 점수
결과가 어떻든 이번이 마지막 변경

실행:  python ai08_funding_rel.py      (ai07_multi_data.py 다음, 약 5분)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from ai01_label import OUT
from ai04_baselines import VOL, by_coin_group, calibration, scores
from ai05_student import fit, paired_boot
from ai06_tree import GAP, MAX_TREES, PARAMS

V1 = ["vol_4h", "vol_24h", "rsi", "funding", "oi_1h", "oi_24h"]
V3 = ["vol_4h", "vol_24h", "rsi", "funding_rel", "oi_1h", "oi_24h"]
WINDOW = "30D"
MIN_HIST = 168


def add_funding_rel(df):
    out = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_index()
        f = g["funding"]
        med = f.rolling(WINDOW, closed="left", min_periods=MIN_HIST).median()   # 지금은 빼고 과거만
        g = g.assign(funding_rel=f - med)
        out.append(g)
    return pd.concat(out).sort_index(kind="stable")


def tree_fit(train, cols):
    cut = train.index[int(len(train) * 0.8)]
    a, b = train[train.index <= cut - GAP], train[train.index > cut]
    m = HistGradientBoostingClassifier(max_iter=MAX_TREES, **PARAMS).fit(a[cols], a["label"])
    losses = [brier_score_loss(b["label"], p[:, 1]) for p in m.staged_predict_proba(b[cols])]
    best = int(np.argmin(losses)) + 1
    return HistGradientBoostingClassifier(max_iter=best, **PARAMS).fit(train[cols], train["label"]), best


def judge(title, y, p_new, p_old, base, days, show=True):
    d = paired_boot(y, p_new, p_old, base, days)
    _, sk_n, auc_n = scores(y, p_new, base)
    _, sk_o, auc_o = scores(y, p_old, base)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    print(f"\n{title} (95% 범위)")
    tag = lambda ok: (f"  {'통과' if ok else '미통과'}" if show else "")
    print(f"  ① AUC 차이    {auc_n - auc_o:+.3f}  ({lo[0]:+.3f} ~ {hi[0]:+.3f}){tag(lo[0] > 0)}")
    print(f"  ② 향상도 차이 {sk_n - sk_o:+.1%}  ({lo[1]:+.1%} ~ {hi[1]:+.1%}){tag(lo[1] > 0)}")
    print(f"  부트스트랩에서 이긴 비율: AUC {(d[:, 0] > 0).mean():.0%}, 향상도 {(d[:, 1] > 0).mean():.0%}")
    if show:
        print(f"  판정: {'합격' if lo[0] > 0 and lo[1] > 0 else '불합격'}")


if __name__ == "__main__":
    df = pd.read_csv(OUT / "ALL_split.csv", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index, utc=True)
    df = add_funding_rel(df)
    before = len(df)
    df = df.dropna(subset=["funding_rel"])
    print(f"데이터: 32종목 (ALL), funding_rel 계산 -> 과거 7일 미만 {before - len(df):,}시간 제외 (모든 예보관 공통)")

    train, test = df[df["mock"] == "train"], df[df["mock"] == "test"]
    assert (test["final"] != "test").all(), "모의고사에 수능 데이터가 섞임"
    print(f"모의 공부 {len(train):,}시간 / 모의고사 {len(test):,}시간")

    print("\nfunding vs funding_rel 분포 (모의 공부 기간)")
    print(train[["funding", "funding_rel"]].describe(percentiles=[.05, .5, .95]).T[["5%", "50%", "95%"]]
          .round(4).to_string())

    y = test["label"].values
    base = np.full(len(test), train["label"].mean())
    days = test.index.floor("D").values

    m_v, m_1, m_3 = fit(train, VOL), fit(train, V1), fit(train, V3)
    p_v = m_v.predict_proba(test[VOL])[:, 1]
    p_1 = m_1.predict_proba(test[V1])[:, 1]
    p_3 = m_3.predict_proba(test[V3])[:, 1]
    t_3, n_trees = tree_fit(train, V3)
    p_t = t_3.predict_proba(test[V3])[:, 1]

    print("\n로지스틱 v3가 배운 것 (표준화 계수)")
    for k, c in zip(V3, m_3[-1].coef_[0]):
        print(f"  {k:12s} {c:+.3f}")

    rows = []
    for name, p in [("③ 변동성 (2개)", p_v), ("로지스틱 v1 (funding)", p_1),
                    ("로지스틱 v3 (funding_rel)", p_3), (f"트리 v3 ({n_trees}그루)", p_t)]:
        b, sk, auc = scores(y, p, base)
        rows.append({"예보관": name, "Brier": f"{b:.4f}", "향상도": f"{sk:+.1%}", "AUC": f"{auc:.3f}"})
    print("\n모의고사 점수")
    print(pd.DataFrame(rows).to_string(index=False))

    judge("로지스틱 v3 - 변동성  [판정]", y, p_3, p_v, base, days, show=True)
    judge("로지스틱 v3 - 로지스틱 v1  [참고: 펀딩비 가공 효과]", y, p_3, p_1, base, days, show=False)
    judge("트리 v3 - 변동성  [참고]", y, p_t, p_v, base, days, show=False)

    g = by_coin_group(train, test, {"변동성": p_v, "v1": p_1, "v3": p_3, "트리v3": p_t})
    print("\n[참고] 공부 때 본 코인 vs 처음 보는 코인 (판정 아님)")
    print(g.to_string(index=False))

    print("\n로지스틱 v3 - 확률이 정직한가")
    print(calibration(y, p_3).to_string(
        formatters={"예보한_확률": "{:.1%}".format, "실제_폭발": "{:.1%}".format}))
