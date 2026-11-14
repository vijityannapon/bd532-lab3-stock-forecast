"""stock_pipeline.py — โค้ดที่ใช้ร่วมกันทั้งตอนเทรน (Colab, Lab 3.1) และตอนทำนายจริง (GitHub Actions, Lab 3.2)

ไฟล์นี้คือ "สัญญา" ระหว่าง training กับ serving: ดาวน์โหลดข้อมูล, ตรวจข้อมูล, สร้าง feature, สร้างหน้าต่างลำดับเวลา
(sequence window) ให้โมเดล deep learning, normalize และแปลงผลทำนายเป็นราคา
ถ้าแก้ไฟล์นี้ ต้องเทรนใหม่และ export ใหม่ทั้งชุด (check_artifacts.py จะจับได้ถ้าแก้ฝั่งเดียว)

ไฟล์นี้ไม่ import PyTorch: โมเดลถูกส่งเข้ามาเป็นฟังก์ชัน predict_fn(X) -> quantiles
ตอนเทรนใช้โมเดล PyTorch ตอนรันจริงใช้ไฟล์ ONNX ผ่าน onnxruntime (ติดตั้งเล็กกว่า PyTorch หลายสิบเท่า)

ทุก feature ใช้หน้าต่างย้อนหลังจำกัด จึงคำนวณจากข้อมูลย้อนหลังไม่กี่เดือนได้ค่าเท่ากับตอนเทรนที่ใช้ข้อมูลสิบปีทุกประการ
"""
from __future__ import annotations

import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

FEATURE_VERSION = "v2-seq"
MARKET = "^GSPC"                     # ดัชนี S&P 500 ใช้เป็น feature ภาวะตลาด
HORIZONS = list(range(1, 8))         # ทำนายราคาปิดล่วงหน้า 1–7 วันทำการ
QUANTILES = [0.1, 0.5, 0.9]          # โมเดลทำนาย 3 quantile: ขอบล่าง, ค่ากลาง (median), ขอบบนของช่วง 80%
MAX_WINDOW = 60                      # หน้าต่างย้อนหลังที่ยาวที่สุดที่ feature ตาราง (FEATURES) ใช้
SEQ_LEN = 60                         # โมเดล deep learning เห็นข้อมูลย้อนหลัง 60 วันทำการต่อการทำนาย 1 ครั้ง
MIN_HISTORY = 100                    # วันขั้นต่ำต่อหุ้น: warm-up ของ feature 20 วัน + หน้าต่าง 60 วัน + เผื่อ
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]
NY = ZoneInfo("America/New_York")

FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",      # ผลตอบแทน (log) ย้อนหลัง k วัน
    "vol_5", "vol_20",                                          # ความผันผวน
    "sma_ratio_5", "sma_ratio_20", "sma_ratio_50",              # ราคาเทียบเส้นค่าเฉลี่ย
    "dist_high_60", "rsi_14", "hl_range", "volume_z_20",
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "rel_ret_20",
    "dow",                                                      # วันในสัปดาห์ 0=จันทร์
]

# input ของโมเดล deep learning: ค่ารายวัน 9 ช่องต่อวัน × 60 วัน (ใช้หน้าต่างย้อนหลังไม่เกิน 20 วัน)
# ไม่ใส่ ret_2…ret_20 หรือ sma_ratio_50 เพราะโมเดลลำดับเวลาเห็นวันก่อนหน้าทั้ง 60 วันอยู่แล้ว เรียนรู้เองได้
SEQ_FEATURES = ["ret_1", "hl_range", "volume_z_20", "vol_20", "sma_ratio_20", "rsi_14",
                "mkt_ret_1", "mkt_vol_20", "rel_ret_20"]


# ---------------------------------------------------------------- ดาวน์โหลดข้อมูล
def download_prices(tickers, start, end=None, retries=3, pause=10.0) -> pd.DataFrame:
    """ดาวน์โหลดราคารายวันจาก Yahoo Finance แล้วคืนเป็นตารางแบบ long: Date, Ticker, Open, High, Low, Close, Volume

    ใช้ auto_adjust=False: Close คือราคาปิดที่ซื้อขายจริง (Yahoo ปรับเฉพาะ stock split ให้แล้ว)
    yf.download() ไม่ raise error เมื่อดาวน์โหลดไม่สำเร็จ แต่คืนตารางว่างหรือคอลัมน์ที่เป็น NaN จึงต้อง validate_prices() ต่อเสมอ
    """
    import yfinance as yf

    symbols = list(dict.fromkeys([*tickers, MARKET]))
    raw = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(symbols, start=start, end=end, interval="1d", auto_adjust=False,
                              actions=False, progress=False, threads=False, group_by="column",
                              multi_level_index=True)
        except Exception as e:                       # เช่น YFRateLimitError
            print(f"  download attempt {attempt}: {type(e).__name__}: {e}")
            raw = None
        if raw is not None and not raw.empty:
            break
        if attempt < retries:
            time.sleep(pause * attempt)              # รอนานขึ้นทุกครั้ง (backoff)
    if raw is None or raw.empty:
        raise RuntimeError("ดาวน์โหลดข้อมูลจาก Yahoo Finance ไม่สำเร็จ (ตารางว่าง)")
    return to_long(raw)


def to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """แปลงผลของ yf.download (คอลัมน์ 2 ชั้น: Price × Ticker) เป็นตาราง long"""
    frames = []
    for t in raw.columns.get_level_values(1).unique():
        df = raw.xs(t, axis=1, level=1)[PRICE_COLS].dropna(subset=["Close"]).copy()
        df["Ticker"] = t
        frames.append(df)
    out = pd.concat(frames)
    out.index.name = "Date"
    out.columns.name = None
    out = out.reset_index()
    d = pd.to_datetime(out["Date"])
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    out["Date"] = d.dt.normalize()
    return out[["Date", "Ticker", *PRICE_COLS]].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def drop_incomplete_bar(prices: pd.DataFrame, now_utc: pd.Timestamp, close_buffer_min: int = 30) -> pd.DataFrame:
    """ตัดแท่งของ "วันนี้" ทิ้งถ้าตลาดยังปิดไม่เสร็จ

    ถ้ารันระหว่างตลาดเปิด Yahoo จะส่งแท่งของวันนี้มาด้วย โดย Close คือราคาล่าสุด ณ ขณะนั้น (ยังไม่ใช่ราคาปิด)
    ถ้าไม่ตัดทิ้ง โมเดลจะทำนายจากราคาที่ยังไม่จบวัน
    """
    now_ny = pd.Timestamp(now_utc).tz_convert(NY)
    today = pd.Timestamp(now_ny.date())
    closed = now_ny.hour * 60 + now_ny.minute >= 16 * 60 + close_buffer_min
    if closed:
        return prices
    return prices[prices["Date"] < today].reset_index(drop=True)


# ---------------------------------------------------------------- ตรวจคุณภาพข้อมูล
def validate_prices(prices: pd.DataFrame, tickers, now_utc=None, max_age_days: int = 5):
    """คืน (errors, warnings) — มี error แปลว่าห้ามนำข้อมูลนี้ไปทำนาย"""
    errors, warnings = [], []
    if prices is None or prices.empty:
        return ["ไม่มีข้อมูลราคาเลย"], warnings
    need = list(dict.fromkeys([*tickers, MARKET]))
    counts = prices.groupby("Ticker").size()
    for t in need:
        n = int(counts.get(t, 0))
        if n < MIN_HISTORY:
            errors.append(f"{t}: มีข้อมูล {n} วัน น้อยกว่าขั้นต่ำ {MIN_HISTORY} วัน")
    if prices.duplicated(["Ticker", "Date"]).any():
        errors.append("มีแถวซ้ำ (Ticker, Date)")
    if prices["Close"].isna().any() or (prices["Close"] <= 0).any():
        errors.append("มีราคาปิดว่างหรือไม่เป็นบวก")
    if (prices["High"] < prices["Low"]).any():
        errors.append("มีแถวที่ High < Low")
    last = prices[prices["Ticker"].isin(need)].groupby("Ticker")["Date"].max()
    if last.nunique() > 1:
        errors.append(f"วันที่ล่าสุดของแต่ละหุ้นไม่ตรงกัน: {last.astype(str).to_dict()}")
    if now_utc is not None and len(last):
        age = (pd.Timestamp(now_utc).tz_convert(NY).tz_localize(None).normalize() - last.max()).days
        if age > max_age_days:
            errors.append(f"ข้อมูลล่าสุดคือ {last.max().date()} เก่าเกิน {max_age_days} วัน (ตลาดหยุดยาวหรือดาวน์โหลดไม่ได้)")
    lr = np.log(prices["Close"]).groupby(prices["Ticker"]).diff().abs()
    jumps = prices.loc[lr > 0.4, ["Ticker", "Date"]]
    if len(jumps):
        warnings.append(f"ราคาเปลี่ยนเกิน ±40% ในวันเดียว {len(jumps)} ครั้ง (split ที่ยังไม่ถูกปรับ?): "
                        f"{jumps.head(3).astype(str).values.tolist()}")
    return errors, warnings


# ---------------------------------------------------------------- feature
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    # ใช้ค่าเฉลี่ยแบบหน้าต่างคงที่ (ไม่ใช้ ewm) เพื่อให้ค่าไม่ขึ้นกับว่าดาวน์โหลดข้อมูลย้อนหลังมาเท่าไร
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    down = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _one_ticker(g: pd.DataFrame) -> pd.DataFrame:
    c = g["Close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=g.index)
    for k in (1, 2, 3, 5, 10, 20):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["vol_5"] = lr.rolling(5).std()
    f["vol_20"] = lr.rolling(20).std()
    for k in (5, 20, 50):
        f[f"sma_ratio_{k}"] = c / c.rolling(k).mean() - 1
    f["dist_high_60"] = c / c.rolling(MAX_WINDOW).max() - 1
    f["rsi_14"] = _rsi(c, 14) / 100
    f["hl_range"] = np.log(g["High"] / g["Low"])
    lv = np.log1p(g["Volume"].astype(float))
    f["volume_z_20"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    return f


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """คืนตาราง Date, Ticker, Close + FEATURES (แถวที่ข้อมูลย้อนหลังยังไม่พอจะมี NaN)"""
    p = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    feats = pd.concat([_one_ticker(g) for _, g in p.groupby("Ticker", sort=False)])
    out = pd.concat([p[["Date", "Ticker", "Close"]], feats.loc[p.index]], axis=1)

    m = p[p["Ticker"] == MARKET].set_index("Date")["Close"]
    mlr = np.log(m).diff()
    mkt = pd.DataFrame({
        "mkt_ret_1": np.log(m / m.shift(1)),
        "mkt_ret_5": np.log(m / m.shift(5)),
        "mkt_ret_20": np.log(m / m.shift(20)),
        "mkt_vol_20": mlr.rolling(20).std(),
    })
    out = out[out["Ticker"] != MARKET].merge(mkt, left_on="Date", right_index=True, how="left")
    out["rel_ret_20"] = out["ret_20"] - out["mkt_ret_20"]
    out["dow"] = out["Date"].dt.dayofweek
    return out[["Date", "Ticker", "Close", *FEATURES]].reset_index(drop=True)


def add_targets(feat: pd.DataFrame) -> pd.DataFrame:
    """เพิ่มเป้าหมาย y_h = log(Close[t+h] / Close[t]) สำหรับ h = 1..7 (ใช้ตอนเทรนเท่านั้น)"""
    out = feat.sort_values(["Ticker", "Date"]).copy()
    g = out.groupby("Ticker")["Close"]
    for h in HORIZONS:
        out[f"y_{h}"] = np.log(g.shift(-h) / out["Close"])
    return out


# ---------------------------------------------------------------- หน้าต่างลำดับเวลา (sequence window)
def make_windows(feat: pd.DataFrame):
    """สร้างหน้าต่างย้อนหลังทุกวันของทุกหุ้น (ใช้ตอนเทรน)

    คืน X shape (N, SEQ_LEN, len(SEQ_FEATURES)) — X[i] คือ SEQ_FEATURES ของวัน t−59 … t
    และ idx: แถวของวัน t (มี Date, Ticker, Close, vol_20 และ y_h ถ้ามี) เรียงตรงกับ X
    """
    Xs, idxs = [], []
    for _, g in feat.sort_values(["Ticker", "Date"]).groupby("Ticker", sort=False):
        a = g[SEQ_FEATURES].to_numpy(np.float32)
        if len(a) < SEQ_LEN:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, SEQ_LEN, axis=0).transpose(0, 2, 1)
        ok = ~np.isnan(w).any(axis=(1, 2))
        Xs.append(w[ok])
        idxs.append(g.iloc[SEQ_LEN - 1:][ok])
    return np.ascontiguousarray(np.concatenate(Xs)), pd.concat(idxs).reset_index(drop=True)


def latest_windows(feat: pd.DataFrame, tickers):
    """หน้าต่างล่าสุด 1 อันต่อหุ้น (ใช้ตอนทำนายจริง) — ทุกหุ้นต้องจบที่วันเดียวกันและไม่มีค่าว่าง"""
    X, rows = [], []
    for t in tickers:
        g = feat[feat["Ticker"] == t].sort_values("Date").tail(SEQ_LEN)
        if len(g) < SEQ_LEN:
            raise ValueError(f"{t}: มีข้อมูลไม่ถึง {SEQ_LEN} วันสำหรับหน้าต่าง")
        if g[SEQ_FEATURES].isna().any().any():
            raise ValueError(f"{t}: หน้าต่างล่าสุดมีค่าว่าง (ข้อมูลย้อนหลังไม่พอหรือมีวันหาย)")
        X.append(g[SEQ_FEATURES].to_numpy(np.float32))
        rows.append(g.iloc[-1])
    idx = pd.DataFrame(rows).reset_index(drop=True)
    if idx["Date"].nunique() != 1:
        raise ValueError(f"วันที่ล่าสุดของแต่ละหุ้นไม่ตรงกัน: {idx.set_index('Ticker')['Date'].astype(str).to_dict()}")
    return np.stack(X), idx


def normalize(X: np.ndarray, meta: dict) -> np.ndarray:
    """standardize ด้วย mean/std ที่คำนวณจากชุด train (เก็บใน meta) แล้วตัดค่าสุดโต่งที่ ±5"""
    mean = np.asarray(meta["scaler"]["mean"], np.float32)
    std = np.asarray(meta["scaler"]["std"], np.float32)
    return np.clip((X - mean) / std, -5, 5).astype(np.float32)


def target_scale(idx: pd.DataFrame) -> np.ndarray:
    """ตัวหารของ target: vol_20 × √h — โมเดลทำนายผลตอบแทนในหน่วย "กี่เท่าของความผันผวนปกติ" shape (N, 7)"""
    return idx["vol_20"].to_numpy()[:, None] * np.sqrt(np.asarray(HORIZONS, float))[None, :]


# ---------------------------------------------------------------- ทำนาย
def onnx_predictor(path):
    """โหลดโมเดล ONNX แล้วคืนฟังก์ชัน predict_fn(X) -> (N, 7, 3)"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # 1 thread: ลำดับการบวกเลขทศนิยมคงที่ ผลจึงตรงกันทุกเครื่อง
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return lambda X: sess.run(None, {"x": np.ascontiguousarray(X, np.float32)})[0]


def forecast(predict_fn, meta: dict, X: np.ndarray, idx: pd.DataFrame) -> pd.DataFrame:
    """ทำนายราคาปิด h = 1..7 พร้อมช่วงคาดการณ์ 80%

    predict_fn(X normalized) คืน quantile ของ z = log return / (vol_20 × √h) shape (N, 7, 3)
    ช่วงคาดการณ์ขยาย/หดด้วยค่า conformal (meta["interval"]["cqr"]) ที่ปรับจากชุด validation
    """
    q = np.asarray(predict_fn(normalize(X, meta)), dtype=np.float64)
    adj = np.array([meta["interval"]["cqr"][str(h)] for h in HORIZONS])
    scale = target_scale(idx)
    mid = q[..., 1] * scale
    lo = np.minimum((q[..., 0] - adj) * scale, mid)
    hi = np.maximum((q[..., 2] + adj) * scale, mid)
    rows = []
    for i, t in enumerate(idx["Ticker"]):
        c = float(idx["Close"].iloc[i])
        for j, h in enumerate(HORIZONS):
            rows.append({"ticker": t, "h": h, "last_close": c, "pred_logret": float(mid[i, j]),
                         "pred_close": c * float(np.exp(mid[i, j])),
                         "lo": c * float(np.exp(lo[i, j])), "hi": c * float(np.exp(hi[i, j]))})
    return pd.DataFrame(rows)


def next_sessions(after: pd.Timestamp, n: int = 7) -> list[pd.Timestamp]:
    """วันทำการถัดไป n วันของตลาด NYSE หลังวันที่ after (ข้ามวันหยุดตลาด)"""
    after = pd.Timestamp(after).normalize()
    try:
        import pandas_market_calendars as mcal
        days = mcal.get_calendar("XNYS").valid_days(after + pd.Timedelta(days=1), after + pd.Timedelta(days=30))
        days = [pd.Timestamp(d).tz_localize(None).normalize() for d in days]
    except ImportError:                                       # สำรอง: จันทร์–ศุกร์ (ไม่รู้จักวันหยุดตลาด)
        days = list(pd.bdate_range(after + pd.Timedelta(days=1), periods=n))
    return days[:n]
