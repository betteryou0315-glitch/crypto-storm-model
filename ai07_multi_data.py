"""
AI 퀀트 모델 7단계 - 교과서 늘리기: 코인 32개 (연구 05 단서 찾기 세트)

코인마다 1~3단계를 그대로 반복한 뒤 한 표로 합친다
  1시간봉 / OI / 펀딩비 받기 -> 정답지 + 힌트 -> 무한대 처리 -> 모의/수능 나누기 (시간 경계 동일)
정답지·힌트는 전부 "평소 대비" 단위라 코인끼리 섞어도 같은 뜻

코인 목록: universe/train32.txt (고정, 바꾸지 않음)
받은 파일은 코인별로 data_ai/ 에 저장 -> 중간에 끊겨도 다시 실행하면 이어서 받음
처음 실행은 OI 하루 파일이 코인당 ~1,100개라 오래 걸림 (코인당 1~3분, 전체 30분~1시간 예상)

실행:  python ai07_multi_data.py
결과:  data_ai/ALL_split.csv   (symbol 칸 추가, 나머지는 BTCUSDT_split.csv 와 같은 모양)
       수능 칸은 이번에도 폭발 비율을 안 보여줌
"""
from pathlib import Path

import pandas as pd

from ai01_label import OUT, load_bars
from ai02_features import build_features, load_funding, load_oi
from ai03_split import split

HERE = Path(__file__).resolve().parent


def find_universe(name="train32.txt"):
    """universe 폴더 안 -> 스크립트와 같은 폴더 순서로 찾음"""
    for p in [HERE / "universe" / name, HERE / name]:
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} 을(를) 못 찾음. {HERE} 또는 {HERE / 'universe'} 에 넣어주세요")


def one_coin(sym):
    cache = OUT / f"{sym}_split.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df, None
    bars = load_bars(sym)
    f = build_features(bars, load_oi(sym), load_funding(sym))
    df, inf = split(f)
    df.to_csv(cache)
    return df, inf


if __name__ == "__main__":
    universe = find_universe()
    syms = [s.strip().lstrip("﻿") for s in universe.read_text(encoding="utf-8").split() if s.strip()]
    print(f"목록 파일: {universe}")
    print(f"코인 {len(syms)}개: {', '.join(syms)}\n")
    if len(syms) != 32:
        print(f"주의: 목록이 32개가 아니라 {len(syms)}개예요. 파일 내용을 확인하세요\n")
    parts, fails, rows = [], [], []
    for i, sym in enumerate(syms, 1):
        print(f"[{i}/{len(syms)}] {sym}")
        try:
            df, inf = one_coin(sym)
        except Exception as e:                     # 한 코인이 실패해도 나머지는 계속
            print(f"  실패: {e}")
            fails.append((sym, str(e)))
            continue
        df.insert(0, "symbol", sym)
        parts.append(df)
        n = lambda col, part: int((df[col] == part).sum())
        mt = df[df["mock"] == "train"]
        rows.append({"코인": sym, "시작": f"{df.index.min():%Y-%m-%d}",
                     "쓸 시간": len(df),
                     "모의공부": n("mock", "train"), "모의고사": n("mock", "test"),
                     "수능": n("final", "test"),
                     "모의공부 폭발": f"{mt['label'].mean():.1%}" if len(mt) else "-",
                     "셋다(모의공부/모의고사/수능)": f"{int(mt['rule'].sum())}/"
                                               f"{int(df.loc[df['mock'] == 'test', 'rule'].sum())}/"
                                               f"{int(df.loc[df['final'] == 'test', 'rule'].sum())}"})

    all_df = pd.concat(parts)
    all_df.to_csv(OUT / "ALL_split.csv")
    print("\n코인별 요약 (수능 폭발 비율은 안 봄)")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n전체")
    for name, col, part in [("모의 공부", "mock", "train"), ("모의고사", "mock", "test"),
                            ("수능 공부", "final", "train"), ("수능", "final", "test")]:
        s = all_df[all_df[col] == part]
        rate = "(안 봄)" if name == "수능" else f"{s['label'].mean():.1%}"
        print(f"  {name:6s} {len(s):>8,}시간  코인 {s['symbol'].nunique():>2}개  "
              f"날 {s.index.floor('D').nunique():>4}일  폭발 {rate}  셋다 {int(s['rule'].sum())}")
    if fails:
        print("\n실패한 코인 (결과 보기 전 기술적 제외, 다른 코인으로 채우지 않음)")
        for s, e in fails:
            print(f"  {e}")
