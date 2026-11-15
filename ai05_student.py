"""
AI 퀀트 모델 5단계 - 학생 공부시키기 (모의고사만, 수능은 절대 안 씀)

학생     : 로지스틱 회귀, 힌트 6개 (vol_4h, vol_24h, rsi, funding, oi_1h, oi_24h)
상대     : 변동성 예보관, 로지스틱 회귀, 힌트 2개 (vol_4h, vol_24h)
-> 모델 종류를 똑같이 맞춰서 차이가 '힌트'에서만 나오게 함

힌트는 가공 없이 그대로 넣음 (표준화만). 로지스틱은 "클수록 더/덜 터진다" 한 방향만 배움
-> 예: "RSI가 아주 높거나 아주 낮을 때 둘 다 터진다" 같은 양쪽 모양은 못 배움 (6단계 트리에서)

합격 기준 (점수 보기 전 확정, 연구 노트 07)
 ① AUC 차이 (학생 - 변동성) > 0, 날 단위 부트스트랩 95% 범위가 0보다 위
 ② 향상도 차이 > 0, 마찬가지
 둘 다 만족해야 합격

실행:  python ai05_student.py          (v1: 힌트 6개)   뒤에 --all 붙이면 32종목
       python ai05_student.py --v2     (v2: 변경 1, 방향 힌트 ret_4h, ret_24h 추가 = 8개)
변경 기록: 변경 1 = 방향 힌트 추가 (모의고사 점수를 보고 공부법을 바꾼 횟수 1회)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ai02_features import FEATURES_V1, FEATURES_V2
from ai04_baselines import N_BOOT, VOL, by_coin_group, calibration, load_mock, scores


def fit(train, cols):
    m = make_pipeline(StandardScaler(), LogisticRegression())
    m.fit(train[cols], train["label"])
    return m


def paired_boot(y, p_s, p_v, base, days, seed=0):
    """같은 날들을 뽑아서 학생과 상대를 동시에 채점 -> 점수 '차이'의 흔들림"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    pos = {d: np.where(days == d)[0] for d in uniq}
    diffs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([pos[d] for d in rng.choice(uniq, len(uniq))])
        _, sk_s, auc_s = scores(y[idx], p_s[idx], base[idx])
        _, sk_v, auc_v = scores(y[idx], p_v[idx], base[idx])
        diffs.append((auc_s - auc_v, sk_s - sk_v))
    return np.array(diffs)


if __name__ == "__main__":
    import sys
    V2 = "--v2" in sys.argv
    train, test = load_mock()
    y = test["label"].values
    base = np.full(len(test), train["label"].mean())
    days = test.index.floor("D").values

    FEATURES = FEATURES_V2 if V2 else FEATURES_V1
    vol = fit(train, VOL)
    stu = fit(train, FEATURES)
    p_v = vol.predict_proba(test[VOL])[:, 1]
    p_s = stu.predict_proba(test[FEATURES])[:, 1]
    print(f"학생 버전: {'v2 (변경 1: 방향 힌트 추가)' if V2 else 'v1'}, 힌트 {len(FEATURES)}개\n")

    print("학생이 배운 것 (표준화 계수: + 클수록 더 터짐 / - 클수록 덜 터짐, 크기 = 영향력)")
    coef = pd.Series(stu[-1].coef_[0], index=FEATURES)
    for k, c in coef.items():
        print(f"  {k:8s} {c:+.3f}")

    rows = []
    trio = [("③ 변동성 (2개)", p_v)]
    if V2:
        p_1 = fit(train, FEATURES_V1).predict_proba(test[FEATURES_V1])[:, 1]
        trio.append(("학생 v1 (6개)", p_1))
    trio.append((f"학생 {'v2' if V2 else 'v1'} ({len(FEATURES)}개)", p_s))
    for name, p in trio:
        b, sk, auc = scores(y, p, base)
        rows.append({"예보관": name, "Brier": f"{b:.4f}", "향상도": f"{sk:+.1%}", "AUC": f"{auc:.3f}"})
    print("\n모의고사 점수")
    print(pd.DataFrame(rows).to_string(index=False))

    d = paired_boot(y, p_s, p_v, base, days)
    _, sk_s, auc_s = scores(y, p_s, base)
    _, sk_v, auc_v = scores(y, p_v, base)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    ok1, ok2 = lo[0] > 0, lo[1] > 0
    print("\n학생 - 변동성 (95% 범위)")
    print(f"  ① AUC 차이    {auc_s - auc_v:+.3f}  ({lo[0]:+.3f} ~ {hi[0]:+.3f})  {'통과' if ok1 else '미통과'}")
    print(f"  ② 향상도 차이 {sk_s - sk_v:+.1%}  ({lo[1]:+.1%} ~ {hi[1]:+.1%})  {'통과' if ok2 else '미통과'}")
    print(f"  부트스트랩에서 학생이 AUC로 이긴 비율 {(d[:, 0] > 0).mean():.0%}, 향상도로 이긴 비율 {(d[:, 1] > 0).mean():.0%}")
    print(f"\n판정: {'합격' if ok1 and ok2 else '불합격'}")

    g = by_coin_group(train, test, {"변동성": p_v, "학생": p_s})
    if g is not None:
        print("\n[참고] 공부 때 본 코인 vs 처음 보는 코인 (판정 아님)")
        print(g.to_string(index=False))

    print("\n학생 - 확률이 정직한가 (예보한 확률을 5칸으로 나눔)")
    print(calibration(y, p_s).to_string(
        formatters={"예보한_확률": "{:.1%}".format, "실제_폭발": "{:.1%}".format}))
