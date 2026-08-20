<h1 align="center">KEHAI</h1>

<p align="center"><i>気配 — そこに誰かが居る、という感じ。</i></p>

KEHAI は、リアルタイムの 3D の身体を持つ音声対話 AI コンパニオンです。話しかけると
自分の声で答え、話している間ずっと身体が動きます — 発話内容から生成される
co-speech ジェスチャー、表情、リップシンク。

[ENGLISH README](./README.md) | 日本語 README

---

## 特徴

- **全二重の音声対話。** マイク入力は WebRTC、出力はストリーミング TTS。サーバ側 VAD が
  発話を区切り、ASR は区間と並行して走り、生成は固定の無音タイムアウトではなく
  セマンティックな終端検出で始まります。
- **絵ではなく身体。** VRM（および Live2D）アバターに、事前生成モーションライブラリ、
  感情駆動の talk ループ、瞬き・表情モーフ、髪とスカートの物理。
- **キャラクターは設定ファイル。** ペルソナ・声・言語・アバター・モーションプロファイルが
  `characters/` の YAML 1 枚でまとめて切り替わります。
- **自分のハードで動く。** このリポジトリはアプリ本体（PC-A）。重いモデル — LLM、
  GPT-SoVITS TTS、Whisper STT、モーション生成 — は GPU マシン（PC-B）の
  プライベートゲートウェイ側で動きます。

## 必要なもの

- Python 3.10〜3.12
- Node.js + npm（フロントエンドを再ビルドする場合のみ）
- 環境変数 `OPENAI_API_KEY`
- GPU ゲートウェイへのアクセス（`MOTION_GATEWAY_KEY`、`.motion_gateway_key` から読み込み）

## クイックスタート

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip && pip install -r requirements.txt

bash start_server.sh            # 詳細ログは --verbose
```

**https://127.0.0.1:12394/** を開き、マイクを許可してください。

> **起動は必ず `start_server.sh`。`python run_server.py` は使わないこと。**
> `MOTION_GATEWAY_KEY` / `GESTURE_ENABLED` / `GSV_STREAM` / LLM コンテキスト上限を
> セットし、ポート 13000 を持つトンネル監視プロセスを起動するのはこのスクリプトだけです。
> 素で起動すると LLM だけは答えるのでチャットは生きて見えますが、TTS と STT が全滅し、
> 症状は「話しかけても喋らない」として現れます。

正常起動なら、最初のログ行はこうなります:

```
GESTURE_ENABLED=1  MOTION_GATEWAY_KEY=<set>  BASE=http://127.0.0.1:13000/v1
```

### 喋らないとき

```powershell
pwsh scripts/health-check.ps1
```

読み取り専用で、会話中に実行しても安全です。サーバ → 起動時 env → トンネル →
音声スタック → 直近のターン、の順に辿り、最初に壊れている箇所の対処法を表示します。
サーバ側が健全と出たら原因はページ側です: `?debug=` 付き URL は時計を止めるので何も再生されず、
再ビルドやサーバ再起動のあとは WebRTC のマイクトラックを張り直すためにハードリロードが必要、
そしてマイク許可はオリジン単位なので `https://127.0.0.1` と `https://localhost` は**別物**です。

## 設定

| 内容 | 場所 |
| --- | --- |
| メイン設定 | `conf.yaml` |
| デフォルト / テンプレート | `config_templates/` |
| キャラクター（ペルソナ + 声 + 言語 + アバター） | `characters/` |
| ペルソナプロンプト | `prompts/personas/` |
| アバター | `live2d-models/`、VRM は `frontend/` 以下 |
| モーションライブラリ | `motion-lib-kawaii/` |

デフォルトキャラクターはミク（英語） — AI-CCORE の秘書アンドロイドです。
`characters/rem_ja.yaml` に切り替えれば、ペルソナ・声・言語・GSV の重みごと
レム（日本語）にワンクリックで戻ります。

## 構成

```
run_server.py            エントリポイント（start_server.sh から起動）
src/kehai/               バックエンド: server / websocket handler / service context
  agent/ asr/ tts/ vad/  差し替え可能なエンジン群（各々ファクトリ 1 つ）
  conversations/         ターン制御と TTS ストリーミング
  motion/                モーションライブラリ、リターゲット、SMPL → アバター
  config_manager/        型付き・検証付きの設定モデル
frontend/                ビルド済み Web UI（配信用・手で編集しない）
frontend-src/            Web UI ソース（React + Vite + Cubism WebSDK）
scripts/                 ビルド / ヘルスチェック / トンネル監視
```

## 開発

```bash
ruff check .                      # lint
ruff format .                     # format
pre-commit run --all-files
pwsh scripts/build-frontend.ps1   # Web UI の再ビルド & デプロイ
```

フロントエンドをビルドしたら、ブラウザをハードリロード（Ctrl+Shift+R）してください。

## クレジット

[Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)（MIT）と
その Web フロントエンドをベースにしています。`LICENSE` および
`frontend-src/LICENSE` を参照（フロントエンドは商用利用条項を含む独自ライセンス）。
