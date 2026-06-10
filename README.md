# 日米業種リードラグ戦略

論文「部分空間正則化付きPCAを用いた日米業種リードラグ投資戦略」の自動実行システム。

## 仕組み

GitHub Actions が **毎朝 6:30 JST（月〜金）** に自動実行し、  
結果を `output/latest.json` に保存します。  
Claude がこのファイルを自動取得して分析します。

## ファイル構成

```
leadlag-strategy/
├── strategy.py                   # メイン戦略スクリプト
├── .github/workflows/
│   └── daily_run.yml             # 自動実行スケジュール
└── output/
    ├── latest.json               # 最新の分析結果（毎朝自動更新）
    └── last_position.json        # 前回のポジション（自動引継ぎ）
```

## パラメータ

| パラメータ | 値 |
|---|---|
| ウィンドウ長 L | 250 日 |
| 主成分数 K | 3 |
| 正則化係数 λ | 0.9 |
| 分位点 q | 0.3（上位・下位 5 業種） |

## 自動実行スケジュール

| 実行タイミング | 説明 |
|---|---|
| UTC 月〜金 21:30 = JST 火〜土 06:30 | 通常の平日分析 |
| 金曜 UTC 21:30 = JST 土 06:30 | 月曜ポジションの事前生成 |

## 手動実行

GitHub の **Actions タブ → Daily Leadlag Strategy → Run workflow** で手動実行できます。
