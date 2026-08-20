# motion-service — HY-Motion ライブ生成サービス (PC-B)

会話中の MMD アバター用ボディモーションを **Tencent HY-Motion 1.0**（テキスト→3Dモーション）で
生成し、SMPL → MMD(.pmx `yyb_miku_10th`) にリターゲットして **.vmd** で返す常駐サービス。

- 本体アプリ (PC-A `kehai`) とは **LAN の HTTP/WS** で連携。
- 設計の全体像は `../docs/motion-service-plan.md` を参照。
- **これはスキャフォールド**です。GPU/モデル固有の実装は `# TODO(PC-B)` を埋めてください。

## 役割
PC-A はモーション計算をしない（TTSで限界）。本サービスが GPU で生成・リターゲットし、
感情別の **ウォームプール**（事前生成済み .vmd）を維持。PC-A は cue を投げ、即時に
ready クリップを受け取る（生成遅延を会話に持ち込まない設計）。

## セットアップ (PC-B / Remote-SSH `AI_CEC_2026`)
```bash
cd ~/kehai/motion-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) HY-Motion 本体を取得（別リポジトリ）
#    git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0
#    その requirements（torch/cuda, pytorch3d 等）も導入
# 2) 重みを ckpts/ へ（HuggingFace: tencent/HY-Motion-1.0） … .gitignore 済み
# 3) SMPL/SMPL-H ボディモデルを body_models/ へ … .gitignore 済み
# 4) yyb_miku の .pmx を assets/ へ（リターゲットのレストポーズ参照に使用）

# 起動（2GPU = 1GPUに1ワーカー）
CUDA_VISIBLE_DEVICES=0,1 python server.py --host 0.0.0.0 --port 8800
```

## ファイル
| ファイル | 役割 | 状態 |
|---|---|---|
| `server.py` | FastAPI(+WS) エンドポイント | 契約あり / モデル未配線で503 |
| `config.py` | 環境変数・パス・並列数 | done |
| `prompts.py` | 感情→英語プロンプト雛形・対象感情 | done（共有定数） |
| `hy_motion_runner.py` | HY-Motion 常駐推論ラッパ | TODO(PC-B) |
| `smpl_to_mmd.py` | SMPL→MMDボーン リターゲッタ＋対応表 | 対応表done / 数値変換TODO |
| `vmd_writer.py` | ボーンフレーム→.vmd | TODO(PC-B) |
| `pool.py` | 感情別ウォームプール管理 | 骨組みdone / 生成配線TODO |

## API
- `GET  /healthz` → `{status, runner_ready, pool}`。
- `GET  /pool/{emotion}/next` → ready な .vmd（`application/octet-stream`）。ライブ主経路。
- `POST /motion` body `{prompt?, emotion, duration?, seed?}` → 一致即返(200 .vmd) or 受理(202 `{job_id}`)。
- `WS   /events` → `{type:"job_done", job_id, url}` を push。

## Phase 0 スパイク（最初にやる）
1. `python -m hy_motion_runner --selftest "a person waving hello" --duration 4`
   → NPZ/SMPL 出力確認＋**1クリップの実生成時間を計測**（ログ）。
2. `python -m smpl_to_mmd <out.npz> assets/yyb_miku.pmx -o out.vmd`
   → 1本を .vmd 化し、フロントの `__mmdPlayVmd("<url>")` で見た目検証（ボーン/軸/レストポーズ/ルート破棄）。
