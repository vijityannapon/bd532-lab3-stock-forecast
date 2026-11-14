"""BD532 Lab 3.2 — ตรวจ artifacts ก่อนทำนายทุกครั้ง (GitHub Actions รันก่อน predict.py)

ถ้าข้อใดไม่ผ่าน → exit code 1 → workflow หยุด ไม่ทำนายและไม่ deploy หน้าเว็บ (หน้าเว็บยังแสดงผลรอบก่อน)
รัน:  python check_artifacts.py
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

import stock_pipeline as sp

ROOT = Path(__file__).resolve().parent
meta = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
TOL = 1e-5      # ONNX บน CPU ต่างรุ่นกัน (เช่น Colab vs GitHub Actions) อาจต่างกันระดับเศษทศนิยม float32 จึงไม่ใช้ 0
failed = 0


def check(name, ok, detail=""):
    global failed
    failed += not ok
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))


# 1) เวอร์ชัน library ต้องตรงกับตอน export
v = meta["versions"]
check("onnxruntime / pandas / numpy ตรงกับตอน export",
      ort.__version__ == v["onnxruntime"] and pd.__version__ == v["pandas"] and np.__version__ == v["numpy"],
      f"Colab ort {v['onnxruntime']} pandas {v['pandas']} numpy {v['numpy']} | "
      f"ที่นี่ ort {ort.__version__} pandas {pd.__version__} numpy {np.__version__}")
py_train, py_here = ".".join(v["python"].split(".")[:2]), ".".join(platform.python_version().split(".")[:2])
if py_train != py_here:
    print(f"⚠️  Python ต่างจากตอนเทรน (เทรน {py_train} | ที่นี่ {py_here}) — ตั้ง python-version ใน workflow ให้ตรง")

# 2) โค้ดเตรียมข้อมูลเวอร์ชันเดียวกับที่ใช้เทรน และ input ของโมเดลตรงกับที่โค้ดสร้าง
check("feature_version ตรงกัน", meta["feature_version"] == sp.FEATURE_VERSION,
      f"โมเดล {meta['feature_version']} | stock_pipeline.py {sp.FEATURE_VERSION}")
check("SEQ_FEATURES ตรงกัน", meta["input"]["features"] == sp.SEQ_FEATURES)
inp = ort.InferenceSession(str(ROOT / "models/forecaster.onnx"), providers=["CPUExecutionProvider"]).get_inputs()[0]
check("input ของ ONNX ตรงกับหน้าต่างที่สร้าง", list(inp.shape[1:]) == [sp.SEQ_LEN, len(sp.SEQ_FEATURES)],
      f"ONNX {inp.name} {inp.shape} | stock_pipeline (batch, {sp.SEQ_LEN}, {len(sp.SEQ_FEATURES)})")

# 3) ผลทำนายจากข้อมูลตัวอย่างต้องเท่ากับที่ได้ใน Colab (จับ training/serving skew)
expected = json.loads((ROOT / "samples/expected_forecast.json").read_text())
prices = pd.read_csv(ROOT / "samples/prices_sample.csv", parse_dates=["Date"])
X, idx = sp.latest_windows(sp.build_features(prices), meta["tickers"])
got = sp.forecast(sp.onnx_predictor(ROOT / "models/forecaster.onnx"), meta, X, idx)
exp = pd.DataFrame(expected["rows"])
same_keys = got[["ticker", "h"]].equals(exp[["ticker", "h"]])
rel = float(np.max(np.abs(got[["pred_close", "lo", "hi"]].to_numpy() / exp[["pred_close", "lo", "hi"]].to_numpy() - 1))) \
    if same_keys else float("inf")
check("ผลทำนายตรงกับ Colab", same_keys and rel < TOL and str(idx["Date"].iloc[0].date()) == expected["as_of"],
      f"{len(got)} แถว × (ค่ากลาง, ขอบล่าง, ขอบบน), ต่างกันมากที่สุด {rel:.1e} (เกณฑ์ < {TOL:.0e})")

# 4) ตัดแท่งที่ตลาดยังไม่ปิด
d = prices["Date"].max()
during = pd.Timestamp(d.date()).tz_localize(sp.NY) + pd.Timedelta(hours=11)     # 11:00 นิวยอร์ก ระหว่างตลาดเปิด
after = pd.Timestamp(d.date()).tz_localize(sp.NY) + pd.Timedelta(hours=17)      # 17:00 นิวยอร์ก หลังตลาดปิด
check("ตัดแท่งของวันนี้ถ้าตลาดยังเปิดอยู่",
      sp.drop_incomplete_bar(prices, during.tz_convert("UTC"))["Date"].max() < d
      and sp.drop_incomplete_bar(prices, after.tz_convert("UTC"))["Date"].max() == d)

# 5) การตรวจข้อมูลจับข้อมูลเสียได้
check("validate_prices จับตารางว่างและหุ้นที่หายไป",
      bool(sp.validate_prices(prices.iloc[0:0], meta["tickers"])[0])
      and bool(sp.validate_prices(prices[prices.Ticker != meta["tickers"][0]], meta["tickers"])[0]))

print(f"\nmodel {meta['model_version']} ({meta['algorithm']}) | {'ผ่านทั้งหมด' if not failed else f'ไม่ผ่าน {failed} ข้อ'}")
sys.exit(1 if failed else 0)