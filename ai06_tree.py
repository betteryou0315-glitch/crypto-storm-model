"""
AI 퀀트 모델 6단계 - 트리 학생 (변경 2: 힌트는 v1 그대로, 학생만 바꿈)
** 모의고사만, 수능은 절대 안 씀 **

트리 학생 = 그래디언트 부스팅 (작은 나무 여러 그루를 차례로 쌓음, scikit-learn 기본 포함)
설정은 모의고사 점수를 보기 전에 고정 (점수 보고 조정하지 않음)
  나무 깊이 3      : 질문을 최대 3번 이어감 (스무고개 3단계) -> 조합은 보되 너무 복잡하지 않게
  잎당 최소 200시간 : 200시간(약 8일)보다 적은 경우로는 규칙을 못 만듦 -> 작은 칸 우연 암기 방지
  학습률 0.05, L2 1.0 : 천천히, 조심스럽게 배움
  나무 개수        : 공부 기간 안에서 정함 (아래)

나무 개수 정하기 (모의고사를 안 쓰고)
  공부 기간의 앞 80% 로 배우고, 25시간 비우고, 뒤 20% 로 채점해서 가장 좋은 개수를 고름
  -> 그 개수로 공부 기간 전체를 다시 배움
  (무작위로 떼면 옆 문제끼리 답이 겹쳐서 너무 낙관적인 개수를 고르게 됨)

합격 기준 (연구 노트 07, 이전과 동일): 트리 vs 변동성 예보관
  ① AUC 차이 95% 범위 > 0   ② 향상도 차이 95% 범위 > 0   둘 다

뭘 배웠는지 보기
  힌트 중요도 : 모의고사에서 그 힌트를 뒤섞어 망가뜨리면 AUC가 얼마나 떨어지나
  힌트별 모양 : 다른 힌트는 그대로 두고 그 힌트만 바꿀 때 예보 확률이 어떻게 변하나

실행:  python ai06_tree.py          (BTC만)
       python ai06_tree.py --all    (32종목)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from ai02_features import FEATURES_V1 as FEATURES
from ai04_baselines import VOL, by_coin_group, calibration, load_mock, scores
from ai05_student import fit, paired_boot

PARAMS = dict(max_depth=3, min_samples_leaf=200, learning_rate=0.05,
              l2_regularization=1.0, early_stopping=False, random_state=0)
MAX_TREES = 600
GAP = pd.Timedelta(hours=25)


def choose_trees(train):
    cut = train.index[int(len(train) * 0.8)]
    a = train[train.index <= cut - GAP]
    b = train[train.index > cut]
    m = HistGradientBoostingClassifier(max_iter=MAX_TREES, **PARAMS).fit(a[FEATURES], a["label"])
    losses = [brier_score_loss(b["label"], p[:, 1])
              for p in m.staged_predict_proba(b[FEATURES])]
    best = int(np.argmin(losses)) + 1
    return best, losses


def importance(model, test, y, n_rep=5, seed=0):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(test[FEATURES])[:, 1])
    out = {}
    for c in FEATURES:
        drops = []
        for _ in range(n_rep):
            x = test[FEATURES].copy()
            x[c] = rng.permutation(x[c].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(x)[:, 1]))
        out[c] = np.mean(drops)
    return pd.Series(out).sort_values(ascending=False)


def shape(model, train, col, qs=(0.05, 0.25, 0.5, 0.75, 0.95), n=3000, seed=0):
    x = train[FEATURES].sample(min(n, len(train)), random_state=seed)
    vals = train[col].quantile(list(qs)).values
    res = []
    for v in vals:
        x2 = x.copy()
        x2[col] = v
        res.append(model.predict_proba(x2)[:, 1].mean())
    return vals, res


if __name__ == "__main__":
    train, test = load_mock()
    y = test["label"].values
    base = np.full(len(test), train["label"].mean())
    days = test.index.floor("D").values

    best, losses = choose_trees(train)
    print(f"나무 개수: 공부 기간 뒤 20%로 고름 -> {best}그루 "
          f"(Brier 1그루 {losses[0]:.4f} -> 최적 {losses[best - 1]:.4f} -> {MAX_TREES}그루 {losses[-1]:.4f})")
    tree = HistGradientBoostingClassifier(max_iter=best, **PARAMS).fit(train[FEATURES], train["label"])

    p_v = fit(train, VOL).predict_proba(test[VOL])[:, 1]
    p_l = fit(train, FEATURES).predict_proba(test[FEATURES])[:, 1]
    p_t = tree.predict_proba(test[FEATURES])[:, 1]

    rows = []
    for name, p in [("③ 변동성 (로지스틱 2개)", p_v), ("학생 v1 (로지스틱 6개)", p_l), ("트리 학생 (6개)", p_t)]:
        b, sk, auc = scores(y, p, base)
        rows.append({"예보관": name, "Brier": f"{b:.4f}", "향상도": f"{sk:+.1%}", "AUC": f"{auc:.3f}"})
    print("\n모의고사 점수")
    print(pd.DataFrame(rows).to_string(index=False))

    for title, other, judge in [("트리 - 변동성  [판정]", p_v, True), ("트리 - 로지스틱 v1  [참고]", p_l, False)]:
        d = paired_boot(y, p_t, other, base, days)
        _, sk_t, auc_t = scores(y, p_t, base)
        _, sk_o, auc_o = scores(y, other, base)
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        print(f"\n{title} (95% 범위)")
        print(f"  ① AUC 차이    {auc_t - auc_o:+.3f}  ({lo[0]:+.3f} ~ {hi[0]:+.3f})"
              + (f"  {'통과' if lo[0] > 0 else '미통과'}" if judge else ""))
        print(f"  ② 향상도 차이 {sk_t - sk_o:+.1%}  ({lo[1]:+.1%} ~ {hi[1]:+.1%})"
              + (f"  {'통과' if lo[1] > 0 else '미통과'}" if judge else ""))
        print(f"  부트스트랩에서 트리가 이긴 비율: AUC {(d[:, 0] > 0).mean():.0%}, 향상도 {(d[:, 1] > 0).mean():.0%}")
        if judge:
            print(f"  판정: {'합격' if lo[0] > 0 and lo[1] > 0 else '불합격'}")

    g = by_coin_group(train, test, {"변동성": p_v, "로지스틱": p_l, "트리": p_t})
    if g is not None:
        print("\n[참고] 공부 때 본 코인 vs 처음 보는 코인 (판정 아님)")
        print(g.to_string(index=False))

    print("\n트리가 중요하게 본 힌트 (모의고사에서 뒤섞으면 AUC가 얼마나 떨어지나)")
    for k, v in importance(tree, test, y).items():
        print(f"  {k:8s} {v:+.4f}")

    print("\n힌트별 모양 (그 힌트만 바꿀 때 평균 예보 확률, 공부 기간 분포의 5/25/50/75/95% 지점)")
    for c in FEATURES:
        vals, res = shape(tree, train, c)
        cells = "  ".join(f"{v:>7.3f}:{r:5.1%}" for v, r in zip(vals, res))
        print(f"  {c:8s} {cells}")

    print("\n트리 학생 - 확률이 정직한가")
    print(calibration(y, p_t).to_string(
        formatters={"예보한_확률": "{:.1%}".format, "실제_폭발": "{:.1%}".format}))
