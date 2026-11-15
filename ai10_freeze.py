"""
실시간 확인용 모델 얼리기 (한 번만 실행) — 연구 노트 08 "실시간 확인 규칙"

2023-11 ~ 2026-10, 32종목 전체(ALL_split.csv의 모든 시간)로
  변동성 예보관  : 로지스틱 (vol_4h, vol_24h)
  합친 모델      : 로지스틱 (vol_4h, vol_24h, rule)
을 한 번 공부시키고, 표준화 값과 계수를 data_ai/live_model.json 에 고정한다.
기록 기간 동안 다시 공부시키지 않는다 (이미 파일이 있으면 덮어쓰지 않음).

실행:  python ai10_freeze.py
"""
import json
from datetime import datetime, timezone

import pandas as pd

from ai01_label import OUT
from ai04_baselines import VOL
from ai05_student import fit

COMB = VOL + ["rule"]
PATH = OUT / "live_model.json"


def export(model, cols):
    sc, lr = model[0], model[-1]
    return {"features": cols, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


if __name__ == "__main__":
    if PATH.exists():
        print(f"이미 얼린 모델이 있어요 ({PATH}). 실시간 기간 동안 바꾸지 않아요.")
        print(PATH.read_text(encoding="utf-8"))
        raise SystemExit
    df = pd.read_csv(OUT / "ALL_split.csv", index_col=0, low_memory=False)
    df.index = pd.to_datetime(df.index, utc=True)
    m_v, m_c = fit(df, VOL), fit(df, COMB)
    out = {"얼린 시각": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "공부 기간": f"{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}",
           "공부 시간 수": len(df), "코인 수": int(df["symbol"].nunique()),
           "공부 기간 폭발 비율": round(float(df["label"].mean()), 4),
           "vol": export(m_v, VOL), "comb": export(m_c, COMB)}
    PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n저장: {PATH}")
