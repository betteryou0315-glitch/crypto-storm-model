"""
AI 퀀트 모델 3단계 - 문제 나누기 (워크포워드: 모의고사 1번 + 수능 1번)

            공부                      | 비움 |  시험
 모의고사:  2023-11 ~ 2024-10 (1년)    | 24h  |  2024-11 ~ 2025-10
 수능(1회): 2023-11 ~ 2025-10 (2년)    | 24h  |  2025-11 ~ 2026-10

비움(엠바고): 공부 문제의 정답은 "봉 마감 후 24시간"을 본다.
            그 24시간이 시험 기간에 걸치는 공부 문제는 버린다
            -> 봉 시작 시각 <= 시험 시작 - 25시간 인 문제만 공부용

실행:  python ai03_split.py      (ai02_features.py 다음)
결과:  data_ai/BTCUSDT_split.csv  (힌트 + 정답 + mock / final 칸)
       mock  = train(모의 공부) / test(모의고사) / 빈칸
       final = train(수능 공부) / test(수능)     / 빈칸
수능 칸은 정답 비율을 일부러 안 보여줌 -> 수능 날까지 아무것도 보지 않는다
"""
import numpy as np
import pandas as pd

from ai01_label import OUT, SYMBOL
from ai02_features import FEATURES

DATA_START = pd.Timestamp("2023-11-01", tz="UTC")
MOCK_TEST = (pd.Timestamp("2024-11-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC"))
FINAL_TEST = (pd.Timestamp("2025-11-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))
EMBARGO = pd.Timedelta(hours=25)          # 봉 1시간 + 정답 24시간


def assign(t, test_start, test_end):
    out = np.full(len(t), "", dtype=object)
    out[(t >= DATA_START) & (t <= test_start - EMBARGO)] = "train"
    out[(t >= test_start) & (t < test_end)] = "test"
    return out


def split(df):
    df = df.copy()
    # 0으로 나눈 곳(무한대) -> 빈칸
    inf = np.isinf(df[FEATURES]).sum()
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["label"])
    t = df.index
    df["mock"] = assign(t, *MOCK_TEST)
    df["final"] = assign(t, *FINAL_TEST)
    return df, inf


def report(df, inf):
    if inf.sum():
        print("무한대 값 -> 빈칸으로 바꿈:", inf[inf > 0].to_dict())
    print(f"쓸 수 있는 시간: {len(df):,}개\n")
    rows = []
    for name, col, part, show in [("모의 공부", "mock", "train", True),
                                  ("모의고사", "mock", "test", True),
                                  ("수능 공부", "final", "train", True),
                                  ("수능", "final", "test", False)]:
        s = df[df[col] == part]
        rows.append({"세트": name,
                     "기간": f"{s.index.min():%Y-%m-%d} ~ {s.index.max():%Y-%m-%d}",
                     "시간 수": f"{len(s):,}",
                     "날 수": s.index.floor("D").nunique(),
                     "폭발 비율": f"{s['label'].mean():.1%}" if show else "(안 봄)",
                     "셋 다 신호": int(s["rule"].sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    # 경계 확인: 공부 마지막 문제의 정답 창이 시험 시작 전에 끝나는가
    for col, (ts, _) in [("mock", MOCK_TEST), ("final", FINAL_TEST)]:
        last = df[df[col] == "train"].index.max()
        end = last + EMBARGO
        print(f"\n[{col}] 공부 마지막 봉 {last:%Y-%m-%d %H:%M} -> 정답 창 끝 {end:%Y-%m-%d %H:%M}"
              f" / 시험 시작 {ts:%Y-%m-%d %H:%M}  {'OK' if end <= ts else '겹침!'}")


if __name__ == "__main__":
    df = pd.read_csv(OUT / f"{SYMBOL}_dataset.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df, inf = split(df)
    df.to_csv(OUT / f"{SYMBOL}_split.csv")
    report(df, inf)
