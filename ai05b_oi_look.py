"""
AI 퀀트 모델 5-b - "OI가 쌓였다가 빠지는 순간 = 청산?" 직접 들여다보기
** 모의 공부 기간(2023-11 ~ 2024-10)만 사용. 모의고사·수능은 안 봄 **

표 1. OI 4칸 : oi_24h 증가/감소 x oi_1h 증가/감소 (0 기준, 숫자를 고르지 않음)
표 2. 의심하는 칸(24시간 쌓임 + 1시간 빠짐)을 '방금 1시간 가격 움직임'으로 나눔
      청산이라면 OI 급감 + 가격 급변이 같이 와야 함
      가격 급변 = 방금 1시간 종가 변화 >= 평소 하루 변동폭의 0.25배 (연구 04 단위 재사용)
      큰 하락 + OI 빠짐 = 롱 청산 모양 / 큰 상승 + OI 빠짐 = 숏 청산(스퀴즈) 모양

각 칸에 같이 보여주는 것
  폭발    : 실제 폭발 비율
  변동성 기대 : 변동성 예보관(vol_4h, vol_24h)이 그 칸에 준 평균 확률
  초과    : 폭발 - 변동성 기대  -> "최근 많이 움직여서"로 설명 안 되는 부분 (연구 01-#12 방식)

주의: 결과를 보고 칸을 고르는 탐색이다. 여기서 본 모양은 '단서'일 뿐이고,
      모의고사·수능에서 따로 확인해야 한다.

실행:  python ai05b_oi_look.py
"""
import numpy as np
import pandas as pd

from ai01_label import SYMBOL, load_bars, make_label
from ai04_baselines import VOL, load_mock
from ai05_student import fit

MOVE = 0.25


def add_price_move(df):
    bars = load_bars(SYMBOL)
    lab = make_label(bars)
    ret1 = (lab["close"] - lab["close"].shift(1)) / lab["dist"]   # 평소 하루 변동폭 단위
    df = df.copy()
    df["ret_1h"] = ret1.reindex(df.index).values
    return df


def table(df, by, order=None):
    t = df.groupby(by, observed=True).agg(
        폭발=("label", "mean"), 변동성_기대=("p_vol", "mean"),
        시간수=("label", "size"), 날수=("day", "nunique"))
    t["초과"] = t["폭발"] - t["변동성_기대"]
    if order:
        t = t.reindex(order)
    t = t[["폭발", "변동성_기대", "초과", "시간수", "날수"]]
    pct = "{:.1%}".format
    return t.to_string(formatters={"폭발": pct, "변동성_기대": pct,
                                   "초과": lambda x: f"{x * 100:+.1f}%p"})


if __name__ == "__main__":
    train, _ = load_mock()                      # 모의고사(test)는 받기만 하고 안 씀
    df = add_price_move(train)
    df["p_vol"] = fit(train, VOL).predict_proba(train[VOL])[:, 1]
    df["day"] = df.index.floor("D")
    print(f"모의 공부 기간만: {len(df):,}시간, 전체 폭발 {df['label'].mean():.1%}\n")

    df["OI 24시간"] = np.where(df["oi_24h"] > 0, "쌓임", "빠짐")
    df["OI 1시간"] = np.where(df["oi_1h"] > 0, "쌓이는 중", "빠지는 중")
    print("표 1. OI 4칸")
    print(table(df, ["OI 24시간", "OI 1시간"]))

    sus = df[(df["oi_24h"] > 0) & (df["oi_1h"] <= 0)].copy()
    sus["방금 1시간 가격"] = np.select(
        [sus["ret_1h"] <= -MOVE, sus["ret_1h"] >= MOVE], ["큰 하락", "큰 상승"], "보통")
    print(f"\n표 2. 의심하는 칸 (24시간 쌓임 + 1시간 빠지는 중) {len(sus):,}시간을 가격 움직임으로 나눔")
    print(table(sus, "방금 1시간 가격", ["큰 하락", "보통", "큰 상승"]))

    # 비교: OI가 쌓이는 중인데 가격이 크게 움직인 경우 (청산이 아닌 급변)
    oth = df[df["oi_1h"] > 0].copy()
    oth["방금 1시간 가격"] = np.select(
        [oth["ret_1h"] <= -MOVE, oth["ret_1h"] >= MOVE], ["큰 하락", "큰 상승"], "보통")
    print(f"\n비교. OI 1시간 쌓이는 중 {len(oth):,}시간 (가격이 움직여도 청산 모양은 아님)")
    print(table(oth, "방금 1시간 가격", ["큰 하락", "보통", "큰 상승"]))
