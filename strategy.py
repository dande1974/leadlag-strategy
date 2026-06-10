#!/usr/bin/env python3
"""
日米業種リードラグ戦略 - 自動実行スクリプト
論文: 部分空間正則化付きPCAを用いた日米業種リードラグ投資戦略

GitHub Actions で毎朝自動実行。
output/latest.json に結果を保存 → Claude が自動取得して分析。
"""

import json
import sys
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

warnings.filterwarnings('ignore')

# ============================================================
# 定数設定
# ============================================================
L      = 250    # ウィンドウ長（営業日）
K      = 3      # 主成分数
LAMBDA = 0.9    # 正則化係数
Q      = 0.3    # 分位点
TOP_N  = 5      # LONG/SHORT 各銘柄数

JST = pytz.timezone('Asia/Tokyo')

US_TICKERS = ['XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLU']

JP_TICKER_MAP = {
    '1617.T': ('食品',           'ディフェンシブ'),
    '1618.T': ('エネルギー資源',  'シクリカル'),
    '1619.T': ('建設・資材',      'シクリカル'),
    '1620.T': ('素材・化学',      'シクリカル'),
    '1621.T': ('医薬品',          'ディフェンシブ'),
    '1622.T': ('自動車・輸送機',  'シクリカル'),
    '1623.T': ('鉄鋼・非鉄',      'シクリカル'),
    '1624.T': ('機械',            'シクリカル'),
    '1625.T': ('電機・精密',      'シクリカル'),
    '1626.T': ('情報・サービス他','シクリカル'),
    '1627.T': ('電力・ガス',      'ディフェンシブ'),
    '1628.T': ('運輸・物流',      'ニュートラル'),
    '1629.T': ('商社・卸売',      'ニュートラル'),
    '1630.T': ('小売',            'ニュートラル'),
    '1631.T': ('銀行',            'シクリカル'),
    '1632.T': ('金融(除く銀行)', 'シクリカル'),
    '1633.T': ('不動産',          'ニュートラル'),
}

JP_NAMES_ORDERED = [JP_TICKER_MAP[t][0]
                    for t in sorted(JP_TICKER_MAP.keys())]

OUTPUT_DIR    = Path('output')
LATEST_JSON   = OUTPUT_DIR / 'latest.json'
LAST_POS_JSON = OUTPUT_DIR / 'last_position.json'


# ============================================================
# ユーティリティ
# ============================================================

def get_jst_now():
    return datetime.now(JST)


def build_jp_trading_days(years: int = 2) -> list:
    """
    1617.T の実データから JP 営業日リストを構築。
    祝日リスト不要・永続メンテフリー。
    """
    end   = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=years * 365 + 60)
    df = yf.download('1617.T',
                     start=start.strftime('%Y-%m-%d'),
                     end=end.strftime('%Y-%m-%d'),
                     progress=False, auto_adjust=True)
    return sorted([d.date() for d in df.index])


def prev_jp_day(d, trading_days: list):
    """d より前の最後の JP 営業日を返す。"""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    candidates = [x for x in trading_days if x < d]
    return candidates[-1] if candidates else None


def next_jp_day(d, trading_days: list):
    """d より後の最初の JP 営業日を返す。"""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    candidates = [x for x in trading_days if x > d]
    return candidates[0] if candidates else None


# ============================================================
# データ取得
# ============================================================

def fetch_us_close(start: str, end: str) -> pd.DataFrame:
    df = yf.download(US_TICKERS, start=start, end=end,
                     progress=False, auto_adjust=True)['Close']
    df = df.dropna(how='all')
    df.index = pd.to_datetime(df.index).date
    return df[US_TICKERS]


def fetch_jp_ohlcv(start: str, end: str) -> dict:
    """各 JP ETF の Open/Close を dict で返す。"""
    result = {}
    for ticker in JP_TICKER_MAP:
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index).date
            result[ticker] = df[['Open', 'Close']]
    return result


# ============================================================
# コアアルゴリズム: 部分空間正則化 PCA
# ============================================================

def build_B_matrix(us_ret: pd.DataFrame, jp_ret: pd.DataFrame):
    """
    Subspace-Regularized PCA で (n_J × n_U) の B 行列を構築。

    us_ret : DataFrame (L × 11)  US Close-to-Close リターン
    jp_ret : DataFrame (L × 17)  JP Close-to-Close リターン
    戻り値 : B (ndarray 17×11), eigenvalues_top3 (list)
    """
    us = us_ret.values   # (L, 11)
    jp = jp_ret.values   # (L, 17)
    n_U, n_J = us.shape[1], jp.shape[1]

    # Z スコア化
    z_U = (us - us.mean(0)) / (us.std(0) + 1e-8)
    z_J = (jp - jp.mean(0)) / (jp.std(0) + 1e-8)

    # 結合行列 [JP | US]  (L, n_J + n_U)
    X = np.hstack([z_J, z_U])

    # データ共分散（÷L しない → 固有値スケールが論文と一致）
    C_data = X.T @ X  # (28, 28)

    # 事前部分空間 C0: クロス共分散ブロックのみ
    C_cross = z_J.T @ z_U   # (n_J, n_U)
    C0 = np.zeros((n_J + n_U, n_J + n_U))
    C0[:n_J, n_J:] = C_cross
    C0[n_J:, :n_J] = C_cross.T

    # 正則化共分散
    C_reg = LAMBDA * C0 + (1 - LAMBDA) * C_data  # (28, 28)

    # 固有値分解（対称行列）
    evals, evecs = np.linalg.eigh(C_reg)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Top-K 固有ベクトルを JP / US に分割
    V_top = evecs[:, :K]
    V_J   = V_top[:n_J, :]   # (n_J, K)
    V_U   = V_top[n_J:, :]   # (n_U, K)

    B = V_J @ V_U.T           # (n_J, n_U)
    return B, evals[:3].tolist()


def compute_signal(B, us_latest_ret, us_mu, us_sigma):
    """
    signal = B @ z_U

    B              : (17, 11)
    us_latest_ret  : (11,)  最新 US Close-to-Close リターン（小数）
    us_mu / sigma  : ウィンドウ平均・標準偏差
    戻り値          : signal (17,), z_U (11,)
    """
    z_U    = (us_latest_ret - us_mu) / (us_sigma + 1e-8)
    signal = B @ z_U
    return signal, z_U


# ============================================================
# 検証
# ============================================================

def verify_positions(prev_pos: dict, jp_ohlcv: dict, verify_date: date) -> dict:
    """前回ポジションを verify_date の JP データで Open-to-Close 検証。"""
    name2ticker = {v[0]: k for k, v in JP_TICKER_MAP.items()}
    results = {}
    long_sum = short_sum = 0.0

    for name in prev_pos.get('long_positions', []):
        ticker = name2ticker.get(name)
        if ticker and ticker in jp_ohlcv and verify_date in jp_ohlcv[ticker].index:
            row = jp_ohlcv[ticker].loc[verify_date]
            op, cl = float(row['Open']), float(row['Close'])
            roc    = (cl - op) / op * 100
            contrib = roc / TOP_N
            long_sum += contrib
            results[name] = {'side': 'LONG', 'open': op, 'close': cl,
                             'roc_pct': round(roc, 4), 'contrib': round(contrib, 4)}

    for name in prev_pos.get('short_positions', []):
        ticker = name2ticker.get(name)
        if ticker and ticker in jp_ohlcv and verify_date in jp_ohlcv[ticker].index:
            row = jp_ohlcv[ticker].loc[verify_date]
            op, cl = float(row['Open']), float(row['Close'])
            roc    = (cl - op) / op * 100
            contrib = -roc / TOP_N
            short_sum += contrib
            results[name] = {'side': 'SHORT', 'open': op, 'close': cl,
                             'roc_pct': round(roc, 4), 'contrib': round(contrib, 4)}

    R = long_sum + short_sum
    return {
        'verify_date'        : str(verify_date),
        'long_side_pct'      : round(long_sum, 4),
        'short_side_pct'     : round(short_sum, 4),
        'strategy_return_pct': round(R, 4),
        'win_loss'           : '勝ち' if R > 0 else '負け',
        'results'            : results,
    }


# ============================================================
# メイン
# ============================================================

def main():
    now_jst   = get_jst_now()
    today_jst = now_jst.date()
    print(f"✅ 実行日時 (JST): {now_jst.strftime('%Y-%m-%d %H:%M')}")
    print(f"   実行日         : {today_jst}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---------- 前回ポジション読み込み ----------
    prev_pos = None
    if LAST_POS_JSON.exists():
        with open(LAST_POS_JSON) as f:
            prev_pos = json.load(f)
        print(f"✅ 前回ポジション読み込み完了")
        print(f"   前回LONG : {prev_pos['long_positions']}")
        print(f"   前回SHORT: {prev_pos['short_positions']}")

    # ---------- Step 1: JP 営業日カレンダー ----------
    print("[1/7] JP 営業日カレンダー構築中...")
    jp_days = build_jp_trading_days()
    print(f"   📅 {len(jp_days)} 営業日（実データ由来）")

    # ---------- Step 2: 取得範囲決定 ----------
    window_end = prev_jp_day(today_jst, jp_days)
    if window_end is None:
        print("ERROR: JP 直前営業日が見つかりません"); sys.exit(1)

    buf_days  = (L + 80) * 2
    fetch_s   = (datetime.now() - timedelta(days=buf_days)).strftime('%Y-%m-%d')
    fetch_e   = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

    # ---------- Step 3: US データ取得 ----------
    print("[2/7] 米国 SPDR ETF データ取得中...")
    us_close_raw = fetch_us_close(fetch_s, fetch_e)
    us_dates     = sorted(us_close_raw.index.tolist())
    us_data_date = us_dates[-1]
    print(f"   米国データ最新日: {us_data_date}")

    # ---------- Step 4: JP データ取得 ----------
    print("[3/7] 日本 TOPIX-17 ETF データ取得中...")
    jp_ohlcv = fetch_jp_ohlcv(fetch_s, fetch_e)

    # Close 系列を DataFrame に集約
    jp_close_dict = {t: odf['Close'] for t, odf in jp_ohlcv.items()}
    jp_close_raw  = pd.DataFrame(jp_close_dict)
    jp_close_raw.index = pd.to_datetime(jp_close_raw.index).date \
        if not isinstance(jp_close_raw.index[0], date) else jp_close_raw.index

    avail_jp = [d for d in jp_days
                if d in jp_close_raw.index
                and not jp_close_raw.loc[d].isna().any()]

    # ---------- Step 5: ウィンドウ構築 ----------
    print("[4/7] リターン計算...")
    # JP ウィンドウ (L 日 Close-to-Close)
    jp_win_dates  = [d for d in avail_jp if d <= window_end][-(L+1):]
    jp_close_win  = jp_close_raw.loc[jp_win_dates]
    jp_ret_win    = jp_close_win.pct_change().dropna().tail(L)

    # カラム名を日本語に、順序を統一
    jp_ret_win.columns = [JP_TICKER_MAP[c][0] for c in jp_ret_win.columns]
    jp_ret_win         = jp_ret_win[[n for n in JP_NAMES_ORDERED
                                     if n in jp_ret_win.columns]]

    # US ウィンドウ (L+1 日 → L 日リターン + 最新1日)
    us_win_dates   = [d for d in us_dates if d <= us_data_date][-(L+2):]
    us_close_win   = us_close_raw.loc[us_win_dates]
    us_ret_full    = us_close_win.pct_change().dropna()
    us_ret_win     = us_ret_full.tail(L)        # B 行列用
    us_latest_ret  = us_ret_full.iloc[-1].values # シグナル用

    print(f"[4/7] 標準化（ウィンドウ終端: {window_end}  長さ: {len(jp_ret_win)} 日）")

    # ---------- Step 6-7: B 行列 → シグナル ----------
    print("[5/7] 事前部分空間・C0 構築...")
    print("[6/7] 正則化PCA 固有分解...")
    B, eig3 = build_B_matrix(us_ret_win, jp_ret_win)
    print(f"[7/7] B行列完成  固有値上位3: {[round(e, 4) for e in eig3]}")

    us_mu    = us_ret_win.mean().values
    us_sigma = us_ret_win.std().values
    signal, z_U = compute_signal(B, us_latest_ret, us_mu, us_sigma)

    # ---------- ポジション決定 ----------
    jp_names     = jp_ret_win.columns.tolist()
    sig_series   = pd.Series(signal, index=jp_names).sort_values(ascending=False)
    long_names   = sig_series.head(TOP_N).index.tolist()
    short_names  = sig_series.tail(TOP_N).index.tolist()

    target_date = next_jp_day(today_jst, jp_days) \
                  or next_jp_day(window_end, jp_days)

    print(f"✅ シグナル計算完了")
    print(f"   米国データ日付  : {us_data_date}")
    print(f"   日本株売買対象日: {target_date}")
    print(f"   LONG : {long_names}")
    print(f"   SHORT: {short_names}")

    # ---------- 前回ポジション検証 ----------
    verification = None
    if prev_pos:
        pt = prev_pos.get('target_date', '')
        try:
            prev_target_date = date.fromisoformat(str(pt))
        except Exception:
            prev_target_date = None

        if prev_target_date and prev_target_date in avail_jp:
            verify_date = prev_target_date
        else:
            verify_date = prev_jp_day(today_jst, jp_days)

        if verify_date:
            verification = verify_positions(prev_pos, jp_ohlcv, verify_date)
            R = verification['strategy_return_pct']
            print(f"✅ 検証完了  R = {R:+.4f}%  {verification['win_loss']}")

    # ---------- US リターン詳細 ----------
    us_prev = us_close_win.iloc[-2].values if len(us_close_win) >= 2 \
              else np.zeros(len(US_TICKERS))
    us_curr = us_close_win.iloc[-1].values
    us_detail = {
        t: {
            'prev_close': round(float(us_prev[i]), 4),
            'curr_close': round(float(us_curr[i]), 4),
            'r_cc_pct'  : round(float(us_latest_ret[i]) * 100, 4),
            'z_U'       : round(float(z_U[i]), 6),
        }
        for i, t in enumerate(US_TICKERS)
    }

    # ---------- 全銘柄シグナル ----------
    name2ticker = {v[0]: k for k, v in JP_TICKER_MAP.items()}
    signals_ranked = {}
    for name in sig_series.index:
        t   = name2ticker.get(name, '')
        signals_ranked[name] = {
            'signal'  : round(float(sig_series[name]), 6),
            'code'    : t.replace('.T', ''),
            'class'   : JP_TICKER_MAP[t][1] if t else '',
            'position': 'LONG' if name in long_names
                        else ('SHORT' if name in short_names else '-'),
        }

    # ---------- JSON 組み立て ----------
    output = {
        'execution_datetime': now_jst.strftime('%Y-%m-%d %H:%M'),
        'analysis': {
            'us_data_date'    : str(us_data_date),
            'window_end'      : str(window_end),
            'window_length'   : int(len(jp_ret_win)),
            'eigenvalues_top3': [round(e, 6) for e in eig3],
            'params'          : {'K': K, 'lambda': LAMBDA, 'q': Q},
        },
        'us_returns'    : us_detail,
        'signals_ranked': signals_ranked,
        'positions': {
            'target_date'    : str(target_date),
            'long_positions' : long_names,
            'short_positions': short_names,
        },
        'verification': verification,
    }

    # ---------- ファイル保存 ----------
    with open(LATEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ latest.json 保存: {LATEST_JSON}")

    new_pos = {
        'target_date'       : str(target_date),
        'long_positions'    : long_names,
        'short_positions'   : short_names,
        'us_data_date'      : str(us_data_date),
        'execution_datetime': now_jst.strftime('%Y-%m-%d %H:%M'),
    }
    with open(LAST_POS_JSON, 'w', encoding='utf-8') as f:
        json.dump(new_pos, f, ensure_ascii=False, indent=2)
    print(f"✅ last_position.json 保存: {LAST_POS_JSON}")

    print("\n=== 完了 ===")
    if verification:
        print(f"前回検証 R = {verification['strategy_return_pct']:+.4f}% "
              f"{verification['win_loss']}")


if __name__ == '__main__':
    main()
