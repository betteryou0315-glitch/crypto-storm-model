"""
AI 퀀트 모델 4단계 - 기준 점수 (모의고사만, 수능은 절대 안 씀)

상대 3명 (모의 공부 기간으로만 배우고, 모의고사로 채점)
 ① 찍기 선수     : 매 시간 "공부 기간 평균 폭발 비율"
 ② 규칙 예보관   : "셋 다" 켜짐 / 꺼짐 각각의 공부 기간 폭발 비율
 ③ 변동성 예보관 : vol_4h, vol_24h 만 보는 로지스틱 회귀  <- 학생이 이겨야 할 진짜 상대

채점
 Brier  : 확률 오차 (낮을수록 좋음)
 향상도 : 찍기 선수보다 Brier가 몇 % 좋아졌나 (높을수록 좋음, 0% = 찍기와 같음)
 AUC    : 터진 순간에 더 높은 확률을 줬나 (0.5 = 동전, 1.0 = 완벽)
 95% 범위: 날 단위로 묶어서 다시 뽑기(부트스트랩) 500번 -> 점수가 운으로 얼마나 흔들리나
           (1시간 문제끼리 정답이 겹치므로 시간 단위가 아니라 날 단위로 묶음)

실행:  pip install scikit-learn   (처음 한 번)
       python ai04_baselines.py          (BTC만)
       python ai04_baselines.py --all    (32종목)
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai01_label import OUT, SYMBOL

VOL = ["vol_4h", "vol_24h"]
N_BOOT = 500


ALL = "--all" in sys.argv          # --all : 32종목 (ai07_multi_data.py 결과), 없으면 BTC만


def load_mock():
    name = "ALL" if ALL else SYMBOL
    df = pd.read_csv(OUT / f"{name}_split.csv", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index(kind="stable")       # 코인을 섞었으면 시간 순서로 다시 정렬
    train, test = df[df["mock"] == "train"], df[df["mock"] == "test"]
    assert (test["final"] != "test").all(), "모의고사에 수능 데이터가 섞임"
    print(f"데이터: {'32종목 (ALL)' if ALL else SYMBOL}")
    return train, test


def by_coin_group(train, test, preds):
    """모의고사를 '공부 때 본 코인' / '처음 보는 코인'으로 나눠 AUC (32종목일 때만)"""
    if "symbol" not in test:
        return None
    seen = set(train["symbol"])
    rows = []
    for gname, mask in [("본 코인", test["symbol"].isin(seen)), ("처음 보는 코인", ~test["symbol"].isin(seen))]:
        t = test[mask]
        if t["label"].nunique() < 2:
            continue
        row = {"그룹": gname, "코인": t["symbol"].nunique(), "시간": f"{len(t):,}",
               "폭발": f"{t['label'].mean():.1%}"}
        for k, p in preds.items():
            row[f"{k} AUC"] = f"{roc_auc_score(t['label'], p[mask.values]):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def fit_predict(train, test):
    y = train["label"]
    preds = {}
    # ① 찍기
    preds["① 찍기"] = np.full(len(test), y.mean())
    # ② 규칙
    p_on = y[train["rule"] == 1].mean()
    p_off = y[train["rule"] == 0].mean()
    preds["② 규칙"] = np.where(test["rule"] == 1, p_on, p_off)
    # ③ 변동성
    m = make_pipeline(StandardScaler(), LogisticRegression())
    m.fit(train[VOL], y)
    preds["③ 변동성"] = m.predict_proba(test[VOL])[:, 1]
    info = {"규칙 켜짐 확률": p_on, "규칙 꺼짐 확률": p_off,
            "변동성 계수(표준화)": {k: round(float(c), 3) for k, c in zip(VOL, m[-1].coef_[0])}}
    return preds, info


def scores(y, p, base):
    b = brier_score_loss(y, p)
    auc = roc_auc_score(y, p) if np.unique(p).size > 1 else 0.5
    return b, 1 - b / brier_score_loss(y, base), auc


def boot(y, preds, days, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    pos = {d: np.where(days == d)[0] for d in uniq}
    out = {k: [] for k in preds}
    for _ in range(N_BOOT):
        idx = np.concatenate([pos[d] for d in rng.choice(uniq, len(uniq))])
        for k, p in preds.items():
            out[k].append(scores(y[idx], p[idx], preds["① 찍기"][idx]))
    return {k: np.array(v) for k, v in out.items()}


def calibration(y, p, bins=5):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"예보": p, "실제": y, "칸": q}).groupby("칸").agg(
        예보한_확률=("예보", "mean"), 실제_폭발=("실제", "mean"), 시간수=("실제", "size"))
    return t


if __name__ == "__main__":
    train, test = load_mock()
    print(f"모의 공부 {len(train):,}시간 ({train.index.min():%Y-%m-%d} ~ {train.index.max():%Y-%m-%d})")
    print(f"모의고사  {len(test):,}시간 ({test.index.min():%Y-%m-%d} ~ {test.index.max():%Y-%m-%d})")
    print(f"모의고사 실제 폭발 비율 {test['label'].mean():.1%}, '셋 다' 신호 {int(test['rule'].sum())}시간\n")

    preds, info = fit_predict(train, test)
    for k, v in info.items():
        print(f"  {k}: {v if isinstance(v, dict) else f'{v:.1%}'}")

    y = test["label"].values
    days = test.index.floor("D").values
    bs = boot(y, preds, days)
    rows = []
    for k, p in preds.items():
        b, sk, auc = scores(y, p, preds["① 찍기"])
        lo, hi = np.percentile(bs[k], [2.5, 97.5], axis=0)
        rows.append({"상대": k, "Brier": f"{b:.4f}",
                     "향상도": f"{sk:+.1%} ({lo[1]:+.1%} ~ {hi[1]:+.1%})",
                     "AUC": f"{auc:.3f} ({lo[2]:.3f} ~ {hi[2]:.3f})"})
    print("\n모의고사 점수  (괄호 = 95% 범위)")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n③ 변동성 예보관 - 확률이 정직한가 (예보한 확률을 5칸으로 나눔)")
    print(calibration(y, preds["③ 변동성"]).to_string(
        formatters={"예보한_확률": "{:.1%}".format, "실제_폭발": "{:.1%}".format}))
