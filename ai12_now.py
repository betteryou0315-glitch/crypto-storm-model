"""
지금 위험한 코인 보기

1. 먼저 ai11_live.py 를 돌려서 최신 봉까지 기록을 채움 (이미 최신이면 그냥 넘어감)
2. 기록(data_ai/live_log.csv)에서 코인마다 가장 최근 1시간봉의 예보를 꺼내 위험한 순서로 보여줌

보는 법
  합친 확률 : 앞으로 24시간 안에 '평소 하루 변동폭'만큼 위나 아래로 움직일 확률 (방향은 모름)
  평소      : 학습 기간 평균 폭발 비율 (약 24%)
  [특보]    : 연구 01 "셋 다" 과열 신호 켜짐 (펀딩비 >= 0.02%, RSI >= 70, OI 1시간 증가)
  기준 거리 : 이 코인의 '평소 하루 변동폭' (= 폭발로 치는 거리, 가격 대비 %)

실행:  python ai12_now.py          (전체 32종목)
       python ai12_now.py 10       (위험한 순서로 10개만)
"""
import json
import sys

import pandas as pd


from ai11_live import LOG, MODEL, main as update


def show(top=None):
    log = pd.read_csv(LOG, index_col=0, low_memory=False)
    log.index = pd.to_datetime(log.index, utc=True)
    base = json.loads(MODEL.read_text(encoding="utf-8"))["공부 기간 폭발 비율"]
    last = log.sort_index().groupby("symbol").tail(1).sort_values("p_comb", ascending=False)
    if top:
        last = last.head(top)
    kst = last.index.max().tz_convert("Asia/Seoul") + pd.Timedelta(hours=1)
    print(f"\n기준: 한국 {kst:%m-%d %H:00} 에 끝난 1시간봉까지  |  평소 {base:.0%}")
    print(f"{'코인':14s} {'합친 확률':>8s} {'평소 대비':>8s} {'변동성만':>8s}  특보")
    for sym, r in last.iterrows():
        x = r["p_comb"] / base
        bar = "#" * int(round(r["p_comb"] * 20))
        flag = "[특보]" if r["rule"] == 1 else ""
        print(f"{r['symbol']:14s} {r['p_comb']:8.0%} {x:7.1f}배 {r['p_vol']:8.0%}  {flag:6s} {bar}")
    stale = last[last.index < last.index.max()]
    if len(stale):
        print(f"\n주의: 최신 봉이 아닌 코인 {len(stale)}개 ({', '.join(stale['symbol'])}) — 데이터가 늦거나 끊김")
    print("\n이건 '얼마나 크게 움직일까'의 확률이에요. 위로인지 아래로인지는 몰라요.")


if __name__ == "__main__":
    update()
    show(int(sys.argv[1]) if len(sys.argv) > 1 else None)
