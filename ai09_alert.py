"""
연구 노트 08 - 변동성 예보 + 과열 특보 (판정)

변동성 예보관 : 로지스틱 (vol_4h, vol_24h)
합친 모델     : 로지스틱 (vol_4h, vol_24h, rule)     rule = 연구 01 "셋 다" (0/1)

판정 기준 (노트 08에 결과 보기 전 등록)
 ① 핵심    : rule = 1 인 시간만, Brier(변동성) - Brier(합친) > 0,
             특보가 켜진 날 단위 부트스트랩 500회 95% 범위 아래 끝 > 0
 ② 안전장치 : 전체 시간, 향상도 차이(합친 - 변동성) 95% 범위 아래 끝 > -0.5%p
 최소 표본  : 시험 기간 특보 100시간 이상, 20일 이상

실행
 python ai09_alert.py --mock                 1차 판정 (모의고사), 한 번만
 python ai09_alert.py --final --confirm      최종 판정 (수능), 한 번만, 1차 통과 기록 필요
 잠금 파일: data_ai/ai09_mock.json, data_ai/ai09_final.json  (지우면 다시 돌릴 수 있지만, 지우지 말 것)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from ai01_label import OUT
from ai04_baselines import N_BOOT, VOL, calibration, scores
from ai05_student import fit, paired_boot

COMB = VOL + ["rule"]
MIN_HOURS, MIN_DAYS = 100, 20
SAFETY = -0.005


def load(stage):
    df = pd.read_csv(OUT / "ALL_split.csv", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index(kind="stable")
    col = "mock" if stage == "mock" else "final"
    train, test = df[df[col] == "train"], df[df[col] == "test"]
    if stage == "mock":
        assert (test["final"] != "test").all(), "모의고사에 수능 데이터가 섞임"
    return train, test


def alert_boot(y, p_v, p_c, days, seed=0):
    """특보 시간만, 특보가 켜진 날 단위로 다시 뽑아서 Brier 차이 (변동성 - 합친)"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    pos = {d: np.where(days == d)[0] for d in uniq}
    out = []
    for _ in range(N_BOOT):
        idx = np.concatenate([pos[d] for d in rng.choice(uniq, len(uniq))])
        out.append(brier_score_loss(y[idx], p_v[idx]) - brier_score_loss(y[idx], p_c[idx]))
    return np.array(out)


def run(stage):
    lock = OUT / f"ai09_{stage}.json"
    if lock.exists():
        print(f"이미 판정함 ({lock}). 결과:")
        print(lock.read_text(encoding="utf-8"))
        return
    if stage == "final":
        mock = OUT / "ai09_mock.json"
        if not mock.exists() or not json.loads(mock.read_text(encoding="utf-8"))["통과"]:
            print("1차 판정(모의고사) 통과 기록이 없어서 수능을 실행하지 않음")
            return

    train, test = load(stage)
    name = "1차 판정: 모의고사" if stage == "mock" else "최종 판정: 수능"
    print(f"[{name}]  공부 {len(train):,}시간 ({train.index.min():%Y-%m-%d} ~ {train.index.max():%Y-%m-%d})"
          f" / 시험 {len(test):,}시간 ({test.index.min():%Y-%m-%d} ~ {test.index.max():%Y-%m-%d})")

    m_v, m_c = fit(train, VOL), fit(train, COMB)
    p_v = m_v.predict_proba(test[VOL])[:, 1]
    p_c = m_c.predict_proba(test[COMB])[:, 1]
    print("\n합친 모델이 배운 것 (표준화 계수)")
    for k, c in zip(COMB, m_c[-1].coef_[0]):
        print(f"  {k:8s} {c:+.3f}")

    y = test["label"].values
    on = (test["rule"] == 1).values
    n_on, d_on = int(on.sum()), test.index[on].floor("D").nunique()
    print(f"\n특보 켜진 시간: {n_on:,}시간, {d_on}일, 코인 {test.loc[on, 'symbol'].nunique()}개")
    result = {"단계": stage, "특보_시간": n_on, "특보_날": int(d_on)}
    if n_on < MIN_HOURS or d_on < MIN_DAYS:
        print(f"판정 불가: 특보가 {MIN_HOURS}시간 / {MIN_DAYS}일 미만")
        result.update(통과=False, 판정="판정 불가")
        lock.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # ① 특보 순간
    y1, v1, c1 = y[on], p_v[on], p_c[on]
    b_v, b_c = brier_score_loss(y1, v1), brier_score_loss(y1, c1)
    d1 = alert_boot(y1, v1, c1, test.index[on].floor("D").values)
    lo1, hi1 = np.percentile(d1, [2.5, 97.5])
    ok1 = lo1 > 0
    print("\n① 특보 순간 (rule = 1)")
    print(f"  실제 폭발 {y1.mean():.1%}  |  변동성 예보 평균 {v1.mean():.1%}  |  합친 모델 예보 평균 {c1.mean():.1%}")
    print(f"  Brier 변동성 {b_v:.4f} -> 합친 {b_c:.4f}  개선 {b_v - b_c:+.4f}  ({lo1:+.4f} ~ {hi1:+.4f})"
          f"  {'통과' if ok1 else '미통과'}")
    print(f"  부트스트랩에서 합친 모델이 이긴 비율 {(d1 > 0).mean():.0%}")

    # ② 전체 시간
    base = np.full(len(test), train["label"].mean())
    _, sk_v, auc_v = scores(y, p_v, base)
    _, sk_c, auc_c = scores(y, p_c, base)
    d2 = paired_boot(y, p_c, p_v, base, test.index.floor("D").values)[:, 1]
    lo2, hi2 = np.percentile(d2, [2.5, 97.5])
    ok2 = lo2 > SAFETY
    print("\n② 전체 시간 (안전장치)")
    print(f"  향상도 변동성 {sk_v:+.2%} -> 합친 {sk_c:+.2%}  차이 {sk_c - sk_v:+.2%}  ({lo2:+.2%} ~ {hi2:+.2%})"
          f"  기준: 아래 끝 > {SAFETY:+.1%}  {'통과' if ok2 else '미통과'}")
    print(f"  (참고) AUC 변동성 {auc_v:.3f} -> 합친 {auc_c:.3f}")

    passed = bool(ok1 and ok2)
    print(f"\n판정: {'통과' if passed else '미통과'}")

    print("\n(참고) 특보 순간을 변동성 높낮이로 나눔 (변동성 예보 중앙값 기준)")
    t = pd.DataFrame({"y": y1, "v": v1, "c": c1})
    t["변동성"] = np.where(t["v"] >= np.median(v1), "높음", "낮음")
    print(t.groupby("변동성").agg(실제=("y", "mean"), 변동성_예보=("v", "mean"),
                                 합친_예보=("c", "mean"), 시간=("y", "size"))
          .to_string(formatters={"실제": "{:.1%}".format, "변동성_예보": "{:.1%}".format,
                                 "합친_예보": "{:.1%}".format}))
    print("\n합친 모델 - 전체 시간 정직함")
    print(calibration(y, p_c).to_string(
        formatters={"예보한_확률": "{:.1%}".format, "실제_폭발": "{:.1%}".format}))

    result.update(통과=passed, 판정="통과" if passed else "미통과",
                  특보_실제=round(float(y1.mean()), 4), 특보_변동성예보=round(float(v1.mean()), 4),
                  특보_합친예보=round(float(c1.mean()), 4),
                  핵심_Brier개선=[round(float(b_v - b_c), 5), round(float(lo1), 5), round(float(hi1), 5)],
                  안전_향상도차이=[round(float(sk_c - sk_v), 5), round(float(lo2), 5), round(float(hi2), 5)])
    lock.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {lock}")


if __name__ == "__main__":
    if "--mock" in sys.argv:
        run("mock")
    elif "--final" in sys.argv:
        if "--confirm" not in sys.argv:
            print("수능은 한 번뿐이에요. 정말 실행하려면 --final --confirm")
        else:
            run("final")
    else:
        print(__doc__)
