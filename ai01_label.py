"""
AI 퀀트 모델 1단계 - 정답지 만들기 (BTC 하나만)

질문: 매 1시간봉이 끝날 때마다
      "다음 24시간 안에 가격이 '평소 하루 변동폭' 이상 위나 아래로 움직였나?"
      있으면 1 (폭발), 없으면 0 (잔잔)

평소 하루 변동폭 = 오늘(UTC) 이전의 완성된 14일 일봉 (고가 - 저가) 평균
                  (완성된 날 = 1시간봉 24개가 다 있는 날, 14일 중 하나라도 빠지면 정답 없음)
기준 거리 = 변동폭 평균 x K (K = 1)
(참고용으로 예전 방식인 고정 ±5% 결과도 같이 보여줌)

실행:  python ai01_label.py
결과:  data_ai/BTCUSDT_1h.csv   (받은 1시간봉, 다음에 또 안 받아도 됨)
       data_ai/BTCUSDT_label.csv (시간별 정답지)
"""
import io
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

SYMBOL = "BTCUSDT"
START = "2023-11"          # 연구 01~06과 같은 기간
END = "2026-10"
HORIZON = 24               # 다음 24시간
K = 1.0                    # 기준 거리 = 14일 변동폭 평균 x K
RANGE_DAYS = 14
MOVE = 0.05                # 참고용 고정 ±5%

OUT = Path("data_ai")
OUT.mkdir(exist_ok=True)
COLS = ["open_time", "open", "high", "low", "close", "volume"]


# ---------- 1. 1시간봉 받기 (data.binance.vision 월별 파일) ----------
def download_month(symbol, ym):
    url = (f"https://data.binance.vision/data/futures/um/monthly/klines/"
           f"{symbol}/1h/{symbol}-1h-{ym}.zip")
    try:
        with urlopen(url, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except HTTPError as e:
        if e.code == 404:          # 상장 전 달 = 파일 없음
            return None
        raise
    raw = z.read(z.namelist()[0]).decode()
    first = raw.split(",", 1)[0]
    has_header = not first.strip().isdigit()     # 최근 파일은 머리줄이 있음
    df = pd.read_csv(io.StringIO(raw), header=0 if has_header else None)
    df = df.iloc[:, :6]
    df.columns = COLS
    return df


def load_bars(symbol):
    cache = OUT / f"{symbol}_1h.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        parts = []
        for ym in pd.period_range(START, END, freq="M").strftime("%Y-%m"):
            print(f"  {symbol} 받는 중 {ym}")
            part = download_month(symbol, ym)
            if part is not None:
                parts.append(part)
        if not parts:
            raise ValueError(f"{symbol}: 기간 안에 1시간봉 파일이 하나도 없음")
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(cache, index=False)
    t = df["open_time"].astype("int64")
    t = t.where(t < 10**14, t // 1000)           # 마이크로초로 된 파일 대비
    df["time"] = pd.to_datetime(t, unit="ms", utc=True)
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ---------- 2. 평소 하루 변동폭 (오늘 이전 완성된 14일) ----------
def daily_range_avg(b, days=RANGE_DAYS):
    day = b.index.floor("D")
    g = b.groupby(day)
    rng = g["high"].max() - g["low"].min()
    rng[g["close"].count() < 24] = float("nan")          # 덜 찬 날은 빈칸
    avg = rng.rolling(days, min_periods=days).mean().shift(1)   # 오늘 제외
    return pd.Series(avg.reindex(day).values, index=b.index)


# ---------- 3. 정답지 만들기 ----------
def make_label(bars, horizon=HORIZON, k=K):
    # 빠진 시간을 빈칸으로 채움 -> 빈칸이 낀 미래 구간은 정답 없음 처리
    full = pd.date_range(bars.index[0], bars.index[-1], freq="1h")
    b = bars.reindex(full)

    # 다음 24개 봉(지금 봉 제외)의 최고가 / 최저가
    # 뒤집어서 rolling -> 다시 뒤집기 = "앞으로 24개", shift(-1) = 지금 봉 제외
    fut_high = b["high"][::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    fut_low = b["low"][::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)

    dist = daily_range_avg(b) * k                         # 기준 거리 (가격 단위)
    up = fut_high >= b["close"] + dist
    down = fut_low <= b["close"] - dist
    label = (up | down).astype(float)
    bad = fut_high.isna() | fut_low.isna() | b["close"].isna() | dist.isna()
    label[bad] = float("nan")

    # 참고: 예전 고정 ±5%
    fixed = ((fut_high >= b["close"] * (1 + MOVE)) |
             (fut_low <= b["close"] * (1 - MOVE))).astype(float)
    fixed[bad] = float("nan")

    return pd.DataFrame({"close": b["close"], "fut_high": fut_high, "fut_low": fut_low,
                         "dist": dist, "dist_pct": dist / b["close"],
                         "label": label, "label_fixed5": fixed})


def report(lab):
    ok = lab.dropna(subset=["label"])
    print(f"\n정답 있는 시간: {len(ok):,}개  (정답 없음 {lab['label'].isna().sum()}개 제외"
          f" - 처음 15일은 변동폭 평균이 없어서 빠짐)")
    print(f"폭발(1) 비율: {ok['label'].mean():.1%}   (참고: 고정 ±5% {ok['label_fixed5'].mean():.1%})")
    print("\n연도별")
    t = ok.groupby(ok.index.year).agg(
        폭발비율=("label", "mean"), 고정5퍼=("label_fixed5", "mean"),
        기준거리_중앙=("dist_pct", "median"), 시간수=("label", "size"))
    pct = "{:.1%}".format
    print(t.to_string(formatters={"폭발비율": pct, "고정5퍼": pct, "기준거리_중앙": pct}))


if __name__ == "__main__":
    bars = load_bars(SYMBOL)
    print(f"{SYMBOL} 1시간봉 {len(bars):,}개  ({bars.index[0]} ~ {bars.index[-1]})")
    lab = make_label(bars)
    lab.to_csv(OUT / f"{SYMBOL}_label.csv")
    report(lab)
