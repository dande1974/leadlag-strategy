#!/usr/bin/env python3
"""
日米業種リードラグ戦略 - 自動実行スクリプト
論文: 部分空間正則化付きPCAを用いた日米業種リードラグ投資戦略
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
# 定数
# ============================================================
L      = 250
K      = 3
LAMBDA = 0.9
Q      = 0.3
TOP_N  = 5
JST    = pytz.timezone('Asia/Tokyo')

US_TICKERS = ['XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','XLK','XLU']

JP_TICKER_MAP = {
    '1617.T': ('食品',            'ディフェンシブ'),
    '1618.T': ('エネルギー資源',   'シクリカル'),
    '1619.T': ('建設・資材',       'シクリカル'),
    '1620.T': ('素材・化学',       'シクリカル'),
    '1621.T': ('医薬品',           'ディフェンシブ'),
    '1622.T': ('自動車・輸送機',   'シクリカル'),
    '1623.T': ('鉄鋼・非鉄',       'シクリカル'),
    '1624.T': ('機械',             'シクリカル'),
    '1625.T': ('電機・精密',       'シクリカル'),
    '1626.T': ('情報・サービス他', 'シクリカル'),
    '1627.T': ('電力・ガス',       'ディフェンシブ'),
    '1628.T': ('運輸・物流',       'ニュートラル'),
    '1629.T': ('商社・卸売',       'ニュートラル'),
    '1630.T': ('小売',             'ニュートラル'),
    '1631.T': ('銀行',             'シクリカル'),
    '1632.T': ('金融(除く銀行)',   'シクリカル'),
    '1633.T': ('不動産',           'ニュートラル'),
}
JP_NAMES_ORDERED = [JP_TICKER_MAP[t][0] for t in sorted(JP_TICKER_MAP.keys())]

OUTPUT_DIR    = Path('output')
LATEST_JSON   = OUTPUT_DIR / 'latest.json'
LAST_POS_JSON = OUTPUT_DIR / 'last_position.json'


# ============================================================
# ユーティリティ
# ============================================================

def get_jst_now():
    return datetime.now(JST)

def to_ts(d):
    return pd.Timestamp(d)

def build_jp_trading_days(years: int = 2) -> list:
    """1617.T の実データから JP 営業日リスト（date オブジェクト）を構築"""
    end   = datetime.now() + timedelta(days=1)
    start = end - timedelta(days=years * 365 + 60)
    df = yf.download('1617.T',
                     start=start.strftime('%Y-%m-%d'),
                     end=end.strftime('%Y-%m-%d'),
                     progress=False, auto_adjust=True)
    # 単一銘柄は MultiIndex の場合も flat の場合もある → どちらでも日付は取れる
    return sorted([pd.Timestamp(d).date() for d in df.index])

def prev_jp_day(d, trading_days):
    if isinstance(d, str): d = date.fromisoformat(d)
    cands = [x for x in trading_days if x < d]
    return cands[-1] if cands else None

def next_jp_day(d, trading_days):
    if isinstance(d, str): d = date.fromisoformat(d)
    cands = [x for x in trading_days if x > d]
    return cands[0] if cands else None


# ============================================================
# データ取得
# ============================================================

def fetch_us_close(start: str, end: str) -> pd.DataFrame:
    """
    US 複数銘柄の終値 DataFrame を返す。
    yfinance は MultiIndex([('Close','XLC'), ...]) を返すので
    df['Close'] で ticker 列を持つ DataFrame に変換。
    """
    df = yf.download(US_TICKERS, start=start, end=end,
                     progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()

    # MultiIndex: df['Close'] → columns=['XLC','XLY',...]
    if isinstance(df.columns, pd.MultiIndex):
        close = df['Close']
    else:
        # 旧形式 or すでに flat の場合
        close = df if set(US_TICKERS).issubset(df.columns) else df

    close = close.dropna(how='all')
    close.index = pd.to_datetime(close.index).normalize()
    available = [t for t in US_TICKERS if t in close.columns]
    if not available:
        print(f"   警告: US ティッカー列が見つかりません。columns={close.columns.tolist()[:5]}")
        return pd.DataFrame()
    return close[available]


def fetch_jp_ohlcv(start: str, end: str) -> dict:
    """
    JP ETF を 1 銘柄ずつ取得。
    単一銘柄なので MultiIndex を flat 化してから Open/Close を選択。
    戻り値: {ticker: DataFrame(DatetimeIndex, columns=['Open','Close'])}
    """
    result = {}
    for ticker in JP_TICKER_MAP:
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            # 単一銘柄: MultiIndex([('Close','1617.T'),...]) → flat(['Close',...])
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).normalize()
            if 'Open' in df.columns and 'Close' in df.columns:
                result[ticker] = df[['Open', 'Close']]
        except Exception as e:
            print(f"   警告: {ticker} 取得失敗 ({e})")
    return result


# ============================================================
# JP Close DataFrame 構築
# ============================================================

def build_jp_close_df(jp_ohlcv: dict) -> pd.DataFrame:
    """
    {ticker: DataFrame} から Close だけを集めた DataFrame を構築。
    .rename() を使わず .name = ticker で安全に Series 名を設定。
    """
    series_list = []
    for ticker, odf in jp_ohlcv.items():
        s = odf['Close'].copy()
        s.name = ticker          # ← .rename(ticker) は使わない
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1)


# ============================================================
# コアアルゴリズム: 部分空間正則化 PCA
# ============================================================

def build_B_matrix(us_ret: pd.DataFrame, jp_ret: pd.DataFrame):
    us = us_ret.values   # (L, 11)
    jp = jp_ret.values   # (L, 17)
    n_U = us.shape[1]
    n_J = jp.shape[1]

    z_U = (us - us.mean(0)) / (us.std(0) + 1e-8)
    z_J = (jp - jp.mean(0)) / (jp.std(0) + 1e-8)

    X      = np.hstack([z_J, z_U])     # (L, n_J+n_U)
    C_data = X.T @ X

    C_cross = z_J.T @ z_U              # (n_J, n_U)
    C0 = np.zeros((n_J + n_U, n_J + n_U))
    C0[:n_J, n_J:] = C_cross
    C0[n_J:, :n_J] = C_cross.T

    C_reg = LAMBDA * C0 + (1 - LAMBDA) * C_data

    evals, evecs = np.linalg.eigh(C_reg)
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    V_top = evecs[:, :K]
    V_J   = V_top[:n_J, :]
    V_U   = V_top[n_J:, :]
    return V_J @ V_U.T, evals[:3].tolist()


def compute_signal(B, us_latest_ret, us_mu, us_sigma):
    z_U = (us_latest_ret - us_mu) / (us_sigma + 1e-8)
    return B @ z_U, z_U


# ============================================================
# 検証
# ============================================================

def verify_positions(prev_pos: dict, jp_ohlcv: dict, verify_date: date) -> dict:
    n2t = {v[0]: k for k, v in JP_TICKER_MAP.items()}
    results  = {}
    long_sum = short_sum = 0.0
    vts = to_ts(verify_date)

    for name in prev_pos.get('long_positions', []):
        t = n2t.get(name)
        if t and t in jp_ohlcv and vts in jp_ohlcv[t].index:
            row = jp_ohlcv[t].loc[vts]
            op, cl  = float(row['Open']), float(row['Close'])
            roc     = (cl - op) / op * 100
            contrib = roc / TOP_N
            long_sum += contrib
            results[name] = {'side':'LONG','open':op,'close':cl,
                             'roc_pct':round(roc,4),'contrib':round(contrib,4)}

    for name in prev_pos.get('short_positions', []):
        t = n2t.get(name)
        if t and t in jp_ohlcv and vts in jp_ohlcv[t].index:
            row = jp_ohlcv[t].loc[vts]
            op, cl  = float(row['Open']), float(row['Close'])
            roc     = (cl - op) / op * 100
            contrib = -roc / TOP_N
            short_sum += contrib
            results[name] = {'side':'SHORT','open':op,'close':cl,
                             'roc_pct':round(roc,4),'contrib':round(contrib,4)}

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

    # 前回ポジション読み込み
    prev_pos = None
    if LAST_POS_JSON.exists():
        with open(LAST_POS_JSON) as f:
            prev_pos = json.load(f)
        print(f"✅ 前回ポジション: LONG={prev_pos['long_positions']}")
        print(f"                   SHORT={prev_pos['short_positions']}")

    # [1/7] JP 営業日カレンダー
    print("[1/7] JP 営業日カレンダー構築中...")
    jp_days = build_jp_trading_days()
    print(f"   📅 {len(jp_days)} 営業日（実データ由来）")

    window_end = prev_jp_day(today_jst, jp_days)
    if window_end is None:
        print("ERROR: JP 直前営業日が見つかりません"); sys.exit(1)

    buf = (L + 80) * 2
    fs  = (datetime.now() - timedelta(days=buf)).strftime('%Y-%m-%d')
    fe  = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

    # [2/7] US データ
    print("[2/7] 米国 SPDR ETF データ取得中...")
    us_raw = fetch_us_close(fs, fe)
    if us_raw.empty:
        print("ERROR: US データが空です"); sys.exit(1)
    us_data_date = us_raw.index[-1].date()
    print(f"   米国データ最新日: {us_data_date}  列数: {us_raw.shape[1]}")

    # [3/7] JP データ
    print("[3/7] 日本 TOPIX-17 ETF データ取得中...")
    jp_ohlcv = fetch_jp_ohlcv(fs, fe)
    print(f"   取得銘柄数: {len(jp_ohlcv)}/17")

    jp_close_raw = build_jp_close_df(jp_ohlcv)
    if jp_close_raw.empty:
        print("ERROR: JP Close データが空です"); sys.exit(1)

    jp_close_dates_set = set(jp_close_raw.index.date)
    avail_jp = [d for d in jp_days
                if d in jp_close_dates_set
                and not jp_close_raw.loc[to_ts(d)].isna().any()]
    print(f"   利用可能 JP 日数: {len(avail_jp)}")

    # [4/7] ウィンドウ構築
    print("[4/7] リターン計算...")
    jp_win_dates = [d for d in avail_jp if d <= window_end][-(L+1):]
    jp_close_win = jp_close_raw.loc[[to_ts(d) for d in jp_win_dates]]
    jp_ret_win   = jp_close_win.pct_change().dropna().tail(L)
    jp_ret_win.columns = [JP_TICKER_MAP[c][0] for c in jp_ret_win.columns]
    jp_ret_win = jp_ret_win[[n for n in JP_NAMES_ORDERED if n in jp_ret_win.columns]]

    us_win      = us_raw[us_raw.index.date <= us_data_date].tail(L + 2)
    us_ret_full = us_win.pct_change().dropna()
    us_ret_win  = us_ret_full.tail(L)
    us_latest   = us_ret_full.iloc[-1].values

    print(f"[4/7] ウィンドウ終端:{window_end}  JP:{jp_ret_win.shape}  US:{us_ret_win.shape}")

    # [5-7] B 行列 / シグナル
    print("[5/7] 事前部分空間・C0 構築...")
    print("[6/7] 正則化PCA 固有分解...")
    B, eig3 = build_B_matrix(us_ret_win, jp_ret_win)
    print(f"[7/7] B行列完成  固有値上位3: {[round(e,4) for e in eig3]}")

    us_mu, us_sigma = us_ret_win.mean().values, us_ret_win.std().values
    signal, z_U = compute_signal(B, us_latest, us_mu, us_sigma)

    jp_names    = jp_ret_win.columns.tolist()
    sig_series  = pd.Series(signal, index=jp_names).sort_values(ascending=False)
    long_names  = sig_series.head(TOP_N).index.tolist()
    short_names = sig_series.tail(TOP_N).index.tolist()
    target_date = next_jp_day(today_jst, jp_days) or next_jp_day(window_end, jp_days)

    print(f"✅ シグナル計算完了")
    print(f"   米国データ日付  : {us_data_date}")
    print(f"   日本株売買対象日: {target_date}")
    print(f"   LONG : {long_names}")
    print(f"   SHORT: {short_names}")

    # 前回ポジション検証
    verification = None
    if prev_pos:
        try:
            prev_td = date.fromisoformat(str(prev_pos.get('target_date', '')))
        except Exception:
            prev_td = None
        verify_date = prev_td if (prev_td and prev_td in avail_jp) \
                      else prev_jp_day(today_jst, jp_days)
        if verify_date:
            verification = verify_positions(prev_pos, jp_ohlcv, verify_date)
            R = verification['strategy_return_pct']
            print(f"✅ 検証完了  R={R:+.4f}%  {verification['win_loss']}")

    # US 詳細データ
    us_prev = us_win.iloc[-2].values if len(us_win) >= 2 else np.zeros(len(US_TICKERS))
    us_curr = us_win.iloc[-1].values
    us_detail = {
        t: {
            'prev_close': round(float(us_prev[i]), 4),
            'curr_close': round(float(us_curr[i]), 4),
            'r_cc_pct'  : round(float(us_latest[i]) * 100, 4),
            'z_U'       : round(float(z_U[i]), 6),
        }
        for i, t in enumerate(US_TICKERS) if i < len(us_latest)
    }

    # 全銘柄シグナル
    n2t = {v[0]: k for k, v in JP_TICKER_MAP.items()}
    signals_ranked = {
        nm: {
            'signal'  : round(float(sig_series[nm]), 6),
            'code'    : n2t.get(nm, '').replace('.T', ''),
            'class'   : JP_TICKER_MAP[n2t[nm]][1] if nm in n2t else '',
            'position': 'LONG' if nm in long_names
                        else ('SHORT' if nm in short_names else '-'),
        }
        for nm in sig_series.index
    }

    # JSON 組み立て
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

    # ファイル保存
    with open(LATEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ latest.json 保存完了")

    new_pos = {
        'target_date'       : str(target_date),
        'long_positions'    : long_names,
        'short_positions'   : short_names,
        'us_data_date'      : str(us_data_date),
        'execution_datetime': now_jst.strftime('%Y-%m-%d %H:%M'),
    }
    with open(LAST_POS_JSON, 'w', encoding='utf-8') as f:
        json.dump(new_pos, f, ensure_ascii=False, indent=2)
    print(f"✅ last_position.json 保存完了")
    print(f"\n=== 完了 ===")

if __name__ == '__main__':
    main()
