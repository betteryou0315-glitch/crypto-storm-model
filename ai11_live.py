"""
실시간 확인 - 매일 기록 (연구 노트 08 "실시간 확인 규칙")

하루 한 번 실행. 바이낸스 공식 API(fapi.binance.com, 무료·키 없음)에서 최신 데이터를 받아
이미 끝난 1시간봉마다 두 예보관의 확률과 특보 여부를 data_ai/live_log.csv 에 추가한다.
  - 2026-11-15 00:00 UTC 봉부터 기록 (그 전은 안 씀)
  - 같은 (코인, 시각)은 한 번만 기록 -> 하루에 여러 번 돌려도, 며칠 빼먹어도 괜찮음
  - 빼먹은 날은 최대 25일까지 채움 (바이낸스 OI 기록이 최근 30일만 있어서)
  - 모델은 data_ai/live_model.json 고정값만 씀 (다시 공부 안 함)
  - 힌트 정의는 연구 07과 같음 (ai02_features.build_features 그대로 사용)

실행:  python ai11_live.py
먼저:  python ai10_freeze.py  (한 번만)
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ai01_label import OUT
from ai02_features import build_features

API = "https://fapi.binance.com"
LIVE_START = pd.Timestamp("2026-11-15 00:00", tz="UTC")
BACKFILL_MAX = pd.Timedelta(days=25)
HISTORY = pd.Timedelta(days=17)            # 14일 변동폭 평균 + 여유
MODEL = OUT / "live_model.json"
LOG = OUT / "live_log.csv"
HERE = Path(__file__).resolve().parent
NEED = ["vol_4h", "vol_24h", "rsi", "funding", "oi_1h"]   # rule 계산에 필요한 힌트


# ---------- API ----------
def get(path, **params):
    url = f"{API}{path}?{urlencode(params)}"
    for i in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "ai-quant-live"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if i == 2:
                raise
            time.sleep(2 * (i + 1))


def ms(t):
    return int(pd.Timestamp(t).timestamp() * 1000)


def fetch_bars(sym, start, now):
    rows = get("/fapi/v1/klines", symbol=sym, interval="1h", startTime=ms(start), limit=1500)
    df = pd.DataFrame([r[:7] for r in rows],
                      columns=["open_time", "open", "high", "low", "close", "volume", "close_time"])
    df = df[df["close_time"].astype("int64") < ms(now)]          # 아직 안 끝난 봉 제외
    df["time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_oi(sym, start, now):
    out, t = [], pd.Timestamp(start)
    step = pd.Timedelta(hours=40)                                # 5분 x 480개 (한 번에 최대 500개)
    while t < now:
        end = min(t + step, now)
        rows = get("/futures/data/openInterestHist", symbol=sym, period="5m",
                   startTime=ms(t), endTime=ms(end), limit=500)
        out += rows
        t = end
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["create_time", "oi"])
    df["create_time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df["oi"] = df["sumOpenInterest"].astype(float)
    return df[["create_time", "oi"]].drop_duplicates("create_time").sort_values("create_time")


def fetch_funding(sym, start):
    rows = get("/fapi/v1/fundingRate", symbol=sym, startTime=ms(start - pd.Timedelta(days=3)), limit=1000)
    df = pd.DataFrame(rows)
    df["calc_time"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)
    # 정산 주기 = 직전 정산과의 간격 (1/2/4/8시간 중 가장 가까운 값), 첫 번째는 중앙값
    gap = df["calc_time"].diff().dt.total_seconds() / 3600
    gap = gap.fillna(gap.median() if gap.notna().any() else 8)
    hours = gap.apply(lambda h: min([1, 2, 4, 8], key=lambda c: abs(c - h)))
    df["funding"] = df["fundingRate"].astype(float) * 8 / hours * 100
    return df[["calc_time", "funding"]]


# ---------- 예보 ----------
def predict(m, X):
    z = m["intercept"] + ((X[m["features"]].values - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"])
    return 1 / (1 + np.exp(-z))


def one_coin(sym, model, since, now):
    first = max(since, now.floor("h") - BACKFILL_MAX)
    bars = fetch_bars(sym, first - HISTORY, now)
    if bars.empty:
        return None, "1시간봉 없음"
    oi = fetch_oi(sym, first - pd.Timedelta(hours=26), now)
    funding = fetch_funding(sym, first)
    f = build_features(bars, oi, funding)
    f = f[(f.index >= first) & (f.index + pd.Timedelta(hours=1) <= now)].copy()
    f[NEED] = f[NEED].replace([np.inf, -np.inf], np.nan)
    ok = f[NEED].notna().all(axis=1)
    skipped = int((~ok).sum())
    f = f[ok]
    if f.empty:
        return None, f"힌트 빈칸으로 기록할 시간 없음 (빈칸 {skipped})"
    f["p_vol"] = predict(model["vol"], f)
    f["p_comb"] = predict(model["comb"], f)
    f.insert(0, "symbol", sym)
    return f.drop(columns=["label"]), (f"빈칸 {skipped}시간 제외" if skipped else "")


def main(now=None):
    if not MODEL.exists():
        print("먼저 python ai10_freeze.py 를 한 번 실행하세요")
        return
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    uni = next(p for p in [HERE / "universe" / "train32.txt", HERE / "train32.txt"] if p.exists())
    syms = [s.strip().lstrip("﻿") for s in uni.read_text(encoding="utf-8").split() if s.strip()]
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))

    log = pd.read_csv(LOG, index_col=0, parse_dates=True) if LOG.exists() else pd.DataFrame()
    if not log.empty:
        log.index = pd.to_datetime(log.index, utc=True)
    done = set(zip(log["symbol"], log.index)) if not log.empty else set()

    new, notes = [], []
    for sym in syms:
        since = LIVE_START
        if not log.empty and (log["symbol"] == sym).any():
            since = max(LIVE_START, log.index[log["symbol"] == sym].max() + pd.Timedelta(hours=1))
        if since + pd.Timedelta(hours=1) > now:
            continue
        try:
            f, note = one_coin(sym, model, since, now)
        except Exception as e:
            notes.append(f"{sym}: 실패 ({e})")
            continue
        if note:
            notes.append(f"{sym}: {note}")
        if f is not None:
            keep = [(sym, t) not in done for t in f.index]
            new.append(f[keep])

    added = pd.concat(new) if new else pd.DataFrame()
    if not added.empty:
        added["logged_at"] = now.isoformat(timespec="seconds")
        out = pd.concat([log, added]) if not log.empty else added
        out.sort_index(kind="stable").to_csv(LOG)
    total = len(log) + len(added)
    print(f"[{now:%Y-%m-%d %H:%M} UTC] 새로 기록 {len(added):,}시간, 누적 {total:,}시간")
    if not added.empty:
        on = added[added["rule"] == 1]
        print(f"  이번 기록 기간 {added.index.min():%Y-%m-%d %H:%M} ~ {added.index.max():%Y-%m-%d %H:%M}")
        print(f"  특보 켜진 시간 {len(on)} ({', '.join(sorted(on['symbol'].unique())) or '없음'})")
        last = added.sort_index().groupby("symbol").tail(1)
        top = last.sort_values("p_comb", ascending=False).head(5)
        print("  지금 가장 위험한 코인 5개 (합친 모델, 앞으로 24시간 폭발 확률)")
        for _, r in top.iterrows():
            print(f"    {r['symbol']:14s} {r['p_comb']:.0%}{'  [특보]' if r['rule'] == 1 else ''}")
    for n in notes:
        print("  " + n)
    # 정답은 일부러 계산하지 않음 -> 채점은 기준(특보 100시간·20일)을 넘은 뒤 따로


if __name__ == "__main__":
    main()
