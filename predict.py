"""BD532 Lab 3.2 — งานทำนายราคาปิดล่วงหน้า 7 วันทำการด้วยโมเดล deep learning (ONNX)
(GitHub Actions รันทุกเช้าวันทำการก่อนตลาดเปิด)

รันจริง (ดาวน์โหลดจาก Yahoo):   python predict.py
รันแบบออฟไลน์ (ไฟล์ตัวอย่าง):    python predict.py --prices-csv samples/prices_sample.csv
จำลองเวลาที่รัน (ไม่เขียนไฟล์):  python predict.py --prices-csv samples/prices_sample.csv --now 2026-11-12T15:00:00Z --dry-run

ผลลัพธ์ (หน้าเว็บอ่านจากไฟล์เหล่านี้):
  docs/data/latest.json            คำทำนายล่าสุด + ราคา 60 วันย้อนหลังสำหรับกราฟ
  docs/data/predictions_log.csv    บันทึกทุกคำทำนายที่เคยเผยแพร่ + ราคาจริงเมื่อครบกำหนด
  docs/data/monitoring.json        ความแม่นยำจริงหลัง deploy เทียบ naive และ coverage ของช่วงคาดการณ์
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stock_pipeline as sp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "data"
LOG = OUT / "predictions_log.csv"
LOG_COLS = ["run_at", "as_of", "ticker", "h", "target_date", "last_close", "pred_close", "lo", "hi",
            "model_version", "actual_close"]
CHART_DAYS = 60            # จำนวนวันย้อนหลังที่แสดงบนกราฟ
DOWNLOAD_DAYS = 200        # ดาวน์โหลดย้อนหลังกี่วันตามปฏิทิน (ต้องได้วันทำการ ≥ MIN_HISTORY)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices-csv", help="ใช้ไฟล์ราคาแทนการดาวน์โหลด (ทดสอบแบบออฟไลน์)")
    ap.add_argument("--now", help="จำลองเวลาที่รัน (ISO 8601 เช่น 2026-11-13T12:20:00Z)")
    ap.add_argument("--dry-run", action="store_true", help="ทำนายและแสดงผล แต่ไม่เขียนไฟล์ใน docs/data")
    return ap.parse_args()


def load_artifacts():
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    if meta["feature_version"] != sp.FEATURE_VERSION:
        sys.exit(f"❌ feature_version ไม่ตรง: โมเดล {meta['feature_version']} แต่ stock_pipeline.py {sp.FEATURE_VERSION}")
    predict_fn = sp.onnx_predictor(ROOT / "models" / "forecaster.onnx")   # onnxruntime: ไม่ต้องมี PyTorch
    return meta, predict_fn


def update_log(fc: pd.DataFrame, meta: dict, as_of, run_at: str, prices: pd.DataFrame) -> pd.DataFrame:
    """เพิ่มคำทำนายรอบนี้ลงบันทึก (ถ้า as_of นี้เคยทำนายแล้วจะไม่เขียนทับ) แล้วเติมราคาจริงของคำทำนายที่ครบกำหนด"""
    if LOG.exists():
        log = pd.read_csv(LOG, parse_dates=["as_of", "target_date"])
    else:
        log = pd.DataFrame(columns=LOG_COLS)
    done = ((log["as_of"] == as_of) & (log["model_version"] == meta["model_version"])).any() if len(log) else False
    if done:
        print(f"ℹ️  as_of {as_of.date()} ถูกบันทึกไว้แล้ว (เช่น รันซ้ำ หรือวันหยุดตลาด) — ไม่เขียนทับคำทำนายที่เผยแพร่ไปแล้ว")
    else:
        new = fc.assign(run_at=run_at, as_of=as_of, model_version=meta["model_version"], actual_close=np.nan)
        new = new.round({"last_close": 4, "pred_close": 4, "lo": 4, "hi": 4})[LOG_COLS]
        log = new if log.empty else pd.concat([log, new], ignore_index=True)

    # เติมราคาจริงให้คำทำนายที่ถึงวันเป้าหมายแล้ว
    actual = prices.rename(columns={"Ticker": "ticker", "Date": "target_date", "Close": "close_now"})
    log = log.drop(columns=["close_now"], errors="ignore").merge(
        actual[["ticker", "target_date", "close_now"]], on=["ticker", "target_date"], how="left")
    log["actual_close"] = log["actual_close"].fillna(log["close_now"].round(4))
    log = log.drop(columns="close_now").sort_values(["as_of", "ticker", "h"]).reset_index(drop=True)
    log.to_csv(LOG, index=False, date_format="%Y-%m-%d")
    return log


def monitoring(log: pd.DataFrame, meta: dict) -> dict:
    m = log.dropna(subset=["actual_close"]).copy()
    rows = []
    for h in sp.HORIZONS:
        g = m[m["h"] == h]
        test = meta["metrics"]["test"].get(str(h), {})
        row = {"h": h, "n": int(len(g)),
               "test_model_err": test.get(f"{meta['algorithm']} %err"), "test_coverage": test.get("coverage 80%")}
        if len(g):
            a = g["actual_close"]
            row |= {
                "model_err": round(float((abs(g["pred_close"] - a) / a).mean() * 100), 3),
                "naive_err": round(float((abs(g["last_close"] - a) / a).mean() * 100), 3),
                "coverage": round(float(((a >= g["lo"]) & (a <= g["hi"])).mean()), 3),
                "direction_acc": round(float((np.sign(g["pred_close"] - g["last_close"])
                                              == np.sign(a - g["last_close"])).mean()), 3),
            }
            row["skill_vs_naive"] = round((1 - row["model_err"] / row["naive_err"]) * 100, 2) if row["naive_err"] else None
        rows.append(row)
    n_total = int(len(m))
    cov = float(((m["actual_close"] >= m["lo"]) & (m["actual_close"] <= m["hi"])).mean()) if n_total else None
    alerts = []
    if n_total >= 50 and cov is not None and cov < meta["interval"]["level"] - 0.15:
        alerts.append(f"coverage จริง {cov:.0%} ต่ำกว่าเป้า {meta['interval']['level']:.0%} มาก — ช่วงคาดการณ์แคบเกินไป")
    return {"n_matured": n_total, "coverage_all": None if cov is None else round(cov, 3),
            "interval_level": meta["interval"]["level"], "by_horizon": rows, "alerts": alerts}


def main():
    args = parse_args()
    meta, predict_fn = load_artifacts()
    tickers = meta["tickers"]

    if args.now:
        now = pd.Timestamp(args.now)
        now = now.tz_localize("UTC") if now.tz is None else now.tz_convert("UTC")
    elif args.prices_csv:
        now = None                                   # ไฟล์ตัวอย่างเป็นข้อมูลเก่า ไม่ตรวจความสด
    else:
        now = pd.Timestamp.now(tz="UTC")
    run_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")

    # 1) ข้อมูล
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv, parse_dates=["Date"])
        source = f"csv:{Path(args.prices_csv).name}"
    else:
        start = (now.tz_convert(sp.NY) - pd.Timedelta(days=DOWNLOAD_DAYS)).date()
        prices = sp.download_prices(tickers, start=str(start))
        source = "yahoo-finance"
    if now is not None:
        before = prices["Date"].max()
        prices = sp.drop_incomplete_bar(prices, now)
        if prices["Date"].max() != before:
            print(f"ℹ️  ตัดแท่งของ {before.date()} ทิ้ง เพราะตลาดยังไม่ปิด (เวลานิวยอร์ก {now.tz_convert(sp.NY):%H:%M})")

    # 2) ตรวจคุณภาพข้อมูล — ไม่ผ่าน = หยุด ไม่เขียนไฟล์ใดๆ (หน้าเว็บยังแสดงผลรอบก่อน)
    errors, warns = sp.validate_prices(prices, tickers, now_utc=now)
    for w in warns:
        print("⚠️ ", w)
    if errors:
        print("❌ ข้อมูลไม่ผ่านการตรวจ:", *errors, sep="\n   - ", file=sys.stderr)
        sys.exit(1)

    # 3) feature → หน้าต่าง 60 วัน → ONNX → ราคา (ฟังก์ชันเดียวกับใน Colab)
    feats = sp.build_features(prices)
    X, latest = sp.latest_windows(feats, tickers)
    as_of = latest["Date"].iloc[0]
    target_dates = sp.next_sessions(as_of, len(sp.HORIZONS))
    fc = sp.forecast(predict_fn, meta, X, latest)
    fc["target_date"] = fc["h"].map(lambda h: target_dates[h - 1])

    show = fc[fc["h"].isin([1, 7])].pivot(index="ticker", columns="h", values="pred_close").round(2)
    show.insert(0, "last_close", latest.set_index("Ticker")["Close"].round(2))
    print(f"✅ as_of {as_of.date()} → h1 {target_dates[0].date()} … h7 {target_dates[-1].date()} | "
          f"model {meta['model_version']} | source {source}")
    print(show.rename(columns={1: "h1", 7: "h7"}).to_string())
    if args.dry_run:
        print("ℹ️  --dry-run: ไม่เขียนไฟล์")
        return

    # 4) บันทึก + monitoring
    OUT.mkdir(parents=True, exist_ok=True)
    log = update_log(fc, meta, as_of, run_at, prices)
    mon = monitoring(log, meta)
    mon["updated_at"] = run_at

    # 5) ไฟล์ที่หน้าเว็บอ่าน
    out = {
        "generated_at": run_at, "as_of": str(as_of.date()), "source": source,
        "target_dates": [str(d.date()) for d in target_dates],
        "model": {"version": meta["model_version"], "algorithm": meta["algorithm"], "architecture": meta["architecture"],
                  "trained_until": meta["data"]["trained_until"], "interval_level": meta["interval"]["level"],
                  "test_period": [meta["splits"]["test"]["start"], meta["splits"]["test"]["end"]]},
        "warnings": warns,
        "tickers": [],
    }
    for t in tickers:
        hist = prices[prices["Ticker"] == t].tail(CHART_DAYS)
        f = fc[fc["ticker"] == t]
        out["tickers"].append({
            "ticker": t, "last_close": round(float(hist["Close"].iloc[-1]), 4),
            "history": {"dates": hist["Date"].dt.strftime("%Y-%m-%d").tolist(),
                        "close": hist["Close"].round(4).tolist()},
            "forecast": [{"h": int(r.h), "date": str(r.target_date.date()), "pred": round(r.pred_close, 4),
                          "lo": round(r.lo, 4), "hi": round(r.hi, 4)} for r in f.itertuples()],
        })
    (OUT / "latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "monitoring.json").write_text(json.dumps(mon, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"📝 เขียน {OUT.relative_to(ROOT)}/latest.json, monitoring.json, predictions_log.csv | monitoring: คำทำนายที่ครบกำหนดแล้ว {mon['n_matured']} รายการ" + (f" | ⚠️ {mon['alerts']}" if mon["alerts"] else ""))


if __name__ == "__main__":
    main()