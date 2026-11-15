"""
AI 퀀트 모델 2단계 - 힌트 만들기 (BTC 하나만)

매 1시간봉이 끝나는 순간(봉 마감 시각)에 이미 알 수 있는 정보만 사용한다.

힌트 (8개)
 ① 최근 변동성 (평소 하루 변동폭 대비, 정답지와 같은 단위)
    vol_4h   : 최근 4개 봉(지금 봉 포함)의 최고가-최저가 / 14일 변동폭 평균
    vol_24h  : 최근 24개 봉의 최고가-최저가 / 14일 변동폭 평균
 ② 가격 과열
    rsi      : RSI(14), 1시간봉 종가
 ①-2 방향 있는 가격 변화 (변경 1에서 추가)
    ret_4h   : (지금 종가 - 4봉 전 종가) / 14일 변동폭 평균
    ret_24h  : (지금 종가 - 24봉 전 종가) / 14일 변동폭 평균
 ③ 레버리지 심리
    funding  : 봉 마감 전 마지막으로 정산된 펀딩비 (8시간 환산, %)
    oi_1h    : OI(코인 개수) 1시간 변화율 (%)
    oi_24h   : OI(코인 개수) 24시간 변화율 (%)
 참고 (힌트 아님, 4단계 기준 점수용)
    rule     : 연구 01 "셋 다" 신호 = funding >= 0.02% & rsi >= 70 & oi_1h > 0

실행:  python ai02_features.py      (ai01_label.py 와 같은 폴더)
결과:  data_ai/BTCUSDT_dataset.csv  (시간별 힌트 + 정답)
"""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from ai01_label import OUT, START, END, SYMBOL, load_bars, make_label

BASE = "https://data.binance.vision/data/futures/um"
FEATURES_V1 = ["vol_4h", "vol_24h", "rsi", "funding", "oi_1h", "oi_24h"]
# 변경 1 (2026-11-14, 모의 공부 기간 탐색 ai05b 에서 "급락이 급등보다 무섭다" 단서)
FEATURES_V2 = FEATURES_V1 + ["ret_4h", "ret_24h"]
FEATURES = FEATURES_V2                     # 빈칸 처리·나누기는 전체 힌트 기준


# ---------- 데이터 받기 ----------
def _read_zip(url):
    try:
        with urlopen(url, timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
    except HTTPError as e:
        if e.code == 404:
            return None                      # 그날 파일이 없음
        raise
    return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))


def load_oi(symbol):
    """5분 metrics (하루 파일) -> create_time, oi(코인 개수)"""
    cache = OUT / f"{symbol}_oi.csv"
    if not cache.exists():
        days = pd.date_range(START + "-01", pd.Period(END).end_time.normalize(), freq="D")
        urls = [f"{BASE}/daily/metrics/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip" for d in days]
        print(f"  {symbol} OI 하루 파일 {len(urls)}개 받는 중 (몇 분 걸림)")
        with ThreadPoolExecutor(16) as ex:
            parts = [p for p in ex.map(_read_zip, urls) if p is not None]
        print(f"  받은 날 {len(parts)} / {len(urls)}")
        if not parts:
            raise ValueError(f"{symbol}: OI 파일이 하나도 없음")
        df = pd.concat(parts, ignore_index=True)[["create_time", "sum_open_interest"]]
        df.to_csv(cache, index=False)
    df = pd.read_csv(cache)
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    return (df.rename(columns={"sum_open_interest": "oi"})
              .drop_duplicates("create_time").sort_values("create_time"))


def load_funding(symbol):
    """월별 펀딩비 -> calc_time(정산 시각), funding(8시간 환산 %)"""
    cache = OUT / f"{symbol}_funding.csv"
    if not cache.exists():
        months = pd.period_range(START, END, freq="M").strftime("%Y-%m")
        urls = [f"{BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip" for m in months]
        print(f"  {symbol} 펀딩비 월 파일 {len(urls)}개 받는 중")
        with ThreadPoolExecutor(8) as ex:
            parts = [p for p in ex.map(_read_zip, urls) if p is not None]
        if not parts:
            raise ValueError(f"{symbol}: 펀딩비 파일이 하나도 없음")
        pd.concat(parts, ignore_index=True).to_csv(cache, index=False)
    df = pd.read_csv(cache)
    t = df["calc_time"].astype("int64")
    df["calc_time"] = pd.to_datetime(t.where(t < 10**14, t // 1000), unit="ms", utc=True)
    # 정산 주기 칸이 없거나 비어 있는 옛 파일은 8시간으로 봄
    hours = df["funding_interval_hours"].fillna(8) if "funding_interval_hours" in df else 8
    df["funding"] = df["last_funding_rate"] * 8 / hours * 100
    return df[["calc_time", "funding"]].drop_duplicates("calc_time").sort_values("calc_time")


# ---------- 힌트 계산 ----------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def asof(times, events, col):
    """각 시각 '이하'의 마지막 값 (미래 값은 절대 안 가져옴)"""
    ns = "datetime64[ns, UTC]"                         # 시간 단위(정밀도)를 하나로 통일
    left = pd.DataFrame({"t": pd.DatetimeIndex(times).astype(ns)})
    right = events.rename(columns={events.columns[0]: "t"})
    right["t"] = right["t"].astype(ns)
    m = pd.merge_asof(left, right, on="t", direction="backward")
    return m[col].values


def build_features(bars, oi, funding):
    lab = make_label(bars)                           # 빈 시간까지 채운 시간표 + 정답
    b = bars.reindex(lab.index)
    close_time = lab.index + pd.Timedelta(hours=1)    # 봉 마감 시각 = 판단 시각
    rng_avg = lab["dist"]                             # 14일 변동폭 평균 (K=1)

    f = pd.DataFrame(index=lab.index)
    hi4, lo4 = b["high"].rolling(4, min_periods=4).max(), b["low"].rolling(4, min_periods=4).min()
    hi24, lo24 = b["high"].rolling(24, min_periods=24).max(), b["low"].rolling(24, min_periods=24).min()
    f["vol_4h"] = (hi4 - lo4) / rng_avg
    f["vol_24h"] = (hi24 - lo24) / rng_avg
    # 방향이 있는 가격 변화 (오르면 +, 내리면 -), 평소 하루 변동폭 단위
    f["ret_4h"] = (b["close"] - b["close"].shift(4)) / rng_avg
    f["ret_24h"] = (b["close"] - b["close"].shift(24)) / rng_avg
    f["rsi"] = rsi(b["close"])
    f["funding"] = asof(close_time, funding, "funding")

    o = oi.assign(seen=oi["create_time"])
    oi_now = pd.Series(asof(close_time, o, "oi"), index=lab.index)
    seen = pd.DatetimeIndex(pd.to_datetime(pd.Series(asof(close_time, o, "seen")), utc=True))
    ns = "datetime64[ns, UTC]"
    stale = (close_time.astype(ns) - seen.astype(ns)) > pd.Timedelta(minutes=30)
    oi_now[np.asarray(stale)] = np.nan                # 30분 넘게 끊긴 OI는 안 씀
    f["oi_1h"] = (oi_now / oi_now.shift(1) - 1) * 100
    f["oi_24h"] = (oi_now / oi_now.shift(24) - 1) * 100

    f["rule"] = ((f["funding"] >= 0.02) & (f["rsi"] >= 70) & (f["oi_1h"] > 0)).astype(int)
    f["label"] = lab["label"]
    return f


def report(f):
    print(f"\n전체 시간 {len(f):,}개")
    print("\n힌트별 빈칸 비율과 분포")
    d = f[FEATURES].describe(percentiles=[.05, .5, .95]).T
    d.insert(0, "빈칸", f[FEATURES].isna().mean())
    print(d[["빈칸", "5%", "50%", "95%"]].to_string(
        formatters={"빈칸": "{:.1%}".format, "5%": "{:.3f}".format,
                    "50%": "{:.3f}".format, "95%": "{:.3f}".format}))
    usable = f.dropna(subset=FEATURES + ["label"])
    print(f"\n힌트 + 정답 다 있는 시간: {len(usable):,}개")
    print(f"규칙 '셋 다' 신호가 켜진 시간: {int(usable['rule'].sum())}개 ({usable['rule'].mean():.2%})")
    # 힌트와 정답의 관계는 일부러 안 봄 -> 3단계에서 공부용/시험용을 나눈 뒤에 공부용만 본다


if __name__ == "__main__":
    bars = load_bars(SYMBOL)
    oi = load_oi(SYMBOL)
    funding = load_funding(SYMBOL)
    print(f"OI {len(oi):,}개 ({oi['create_time'].iloc[0]} ~ {oi['create_time'].iloc[-1]})")
    print(f"펀딩비 정산 {len(funding):,}회")
    f = build_features(bars, oi, funding)
    f.to_csv(OUT / f"{SYMBOL}_dataset.csv")
    report(f)
