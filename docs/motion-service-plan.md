# 初音ミク モーション: Tencent HY-Motion ライブ生成（PC分離・エンドツーエンド）

> リポジトリ内の正本。設計を変えたらこのファイルを更新して push。
> （元の作業プランは個人の Claude plans に保管。内容はこちらと同一。）

## Context（なぜ・何を）

会話中の MMD（`.pmx`, `yyb_miku_10th`）アバターのボディモーションを **Tencent HY-Motion 1.0**
（テキスト→3Dモーション拡散モデル）で生成して人間らしくする。

- モーション生成は **PC-A（このアプリ＝TTSで限界）ではなく、LAN上の PC-B（RTX 3090 ×2 = 各24GB）** で実行。
- 方式: **ライブ生成サービス**／計画: **エンドツーエンド全体**。

### HY-Motion の確定事実（調査済み・設計を縛る制約）
- 入力 = **英語テキスト**（<30語推奨）、出力 = **NPZ(SMPL/SMPL-H)**、**最長5秒**。
- **拡散モデルで生成は数秒〜数十秒（リアルタイム不可）**、VRAM 24–26GB、**HTTP API 無し**（CLI `local_infer.py` / Gradio のみ）。
- **MMDボーンへのリターゲットは自前**。**音声同期ではない**（→ 汎用トーキングclipはどの文にも流用でき、語単位同期は不要かつ不可能）。

参考: HY-Motion GitHub / arXiv 2512.23464 / ComfyUI-HY-Motion1（FBX/GLB/retarget）/ MMD_3D_POSE_Converter / Nuthouse01 PMX-VMD-Tools / GMR。

### 設計の肝（遅延を会話に持ち込まない）
「文ごとに専用モーションを同期生成して待つ」は破綻する。代わりに:
1. PC-B 常駐サービスが **感情別ウォームプール**（事前生成済み .vmd 群）を維持し続ける。
2. 実行時はフロントが **プールから即時取得 → クロスフェード再生**（発話中はトーキングclipをループ/連鎖）。
3. 感情変化・キーワードで該当clipを選択。専用プロンプトは **非同期** で投げ、出来たら次のビートで差し込む（数秒遅れOK）。
4. PC-B が遅い/落ちている時は **手続き型idle（フォールバック）** に滑らかに退避し、会話は止めない。

---

## アーキテクチャ（3層）

```
[PC-A: open_llm_vtuber]                       [PC-B: 2×RTX3090]
 LLM→感情抽出→TTS(既存)                        HY-Motion 常駐サービス(新規 motion-service/)
   │ motion cue(感情/プロンプト) ─ LAN(HTTP/WS) ─▶ 生成→SMPL→MMDリターゲット→.vmd
   │ ◀── .vmd(またはURL) ───────────────────────  感情別ウォームプール/キャッシュ
   ▼
 WebSocket payload に motion フィールド追加
   ▼
[Frontend: mmd-viewer.tsx]
 .vmd を AnimationMixer で再生＋crossFade、helper.update で物理、idleはゲート
 口パク(mmd-lipsync)・表情は独立で不変
```

---

## コンポーネント別 設計

### A. PC-B — `motion-service/`（FastAPI, モデル常駐, 1GPU=1ワーカーで2並列）
1. **推論ラッパ** `hy_motion_runner.py`: HY-Motion を import し常駐。`generate(prompt, duration<=5, seed) -> SMPL npz`。
2. **リターゲッタ** `smpl_to_mmd.py`（最重要・最難関）: SMPL/SMPL-H 関節回転 → yyb_miku の MMDボーン局所 quaternion 列。
   レストポーズ差(T字/A字)・座標系・**ルート移動破棄**。流用: GMR / MMD_3D_POSE_Converter / Nuthouse01。
3. **VMDライタ** `vmd_writer.py`: ボーンキーフレーム → `.vmd`。
4. **プール** `pool.py`: 感情ごとに常時 N 本の ready .vmd を背景生成＋オンデマンド割り込み。
5. **API** `server.py`:
   - `GET  /pool/{emotion}/next` → 即時 ready な汎用talkクリップ .vmd（ライブ主経路）。
   - `POST /motion {prompt?,emotion,duration,seed?}` → 一致なら 200+.vmd、無ければ 202+job_id。
   - `WS   /events` → 非同期 job 完了 push。
   - `GET  /healthz`。

### B. PC-A — backend（`src/open_llm_vtuber/`）
1. **MotionClient** `motion/hy_motion_client.py`: PC-B への async クライアント。タイムアウト/失敗時はフォールバック指示。
2. **プロンプト導出**: 感情→英語プロンプト雛形（`motion-service/prompts.py` と共有）。
3. **会話統合**: 既存感情抽出（`agent/transformers.py`, `live2d_model.resolve_*`）に相乗り。
4. **payload 拡張**: `conversations/types.py:AudioPayload` / `agent/output_types.py:Actions` / `utils/stream_audio.py` に `motion` 追加。
   PC-A が .vmd を取得し `cache/motion/*.vmd` を既存静的ルートで配信。
5. **設定**: `conf.yaml` / `config_manager` に `motion_service:{ enabled, base_url, timeout, … }`。

### C. Frontend（`frontend-src/`）
1. **VMDプレイヤー** `utils/mmd-motion-player.ts` ＋ `mmd-viewer.tsx`:
   自前 `THREE.AnimationMixer`（helperは物理専任）、`MMDLoader.loadAnimation`、`crossFadeTo(0.3)`/`fadeOut`。
   `animate()`: `mixer.update(delta)` → `helper.update(delta)`。**clip再生中は updateIdle のボーン書込みをスキップ**、終了でidle復帰。
2. **cue 受信**: `websocket-handler.tsx`/`use-audio-task.ts` で payload `motion` を読み `window.__mmd` 経由でプレイヤー駆動。
3. **フォールバック**: 手続き型 idle を残し graceful degradation。口パク/表情は不変。

---

## フェーズ
- **Phase 0 — スパイク**: ①PC-B で HY-Motion 実行→NPZ確認→**1クリップ(5s)の実生成時間を実測**。②1本を SMPL→.vmd 自動リターゲット→フロント `__mmdPlayVmd(url)` で yyb_miku に載せ見た目検証。
- **Phase 1 — PC-Bサービス**: 常駐推論(2GPU)＋リターゲッタ＋VMDライタ＋プール＋API（curl→.vmd）。
- **Phase 2 — フロントVMDプレイヤー**: 取得/再生/クロスフェード/物理/idleゲート/復帰。
- **Phase 3 — backend統合**: MotionClient＋プロンプト＋payload `motion`＋conf.yaml＋.vmd配信＋感情cue。
- **Phase 4 — ライブ調整**: プール深さ/クロスフェード/感情マップ/遅延隠し/フォールバック、会話E2E。

## リスク
1. **生成遅延 vs 会話速度（最大）**: 感情別プール＋汎用talkクロスフェード＋非同期差込＋thinking中先行＋2GPU。Phase0で実測。
2. **SMPL→MMDリターゲット忠実度**: レストポーズ/軸/スケール/ルート破棄。Phase0で1本を端から端まで検証。
3. **idle と clip のボーン競合・物理**: clip中 idle ゲート、物理はアニメ後、切替クロスフェード。
4. **指/顔の精度**: SMPL body は手指が乏しい。顔は morph/口パクで別管理。
5. **PC-B 可用性**: タイムアウト＋idleへ fallback。会話は止めない。
6. **VRAM/throughput**: 2×3090で十分。モデル常駐・1GPU1ワーカー。

## 検証
- PC-B単体: `curl http://<PC-B>:8800/pool/happy/next -o out.vmd`。
- 見た目: フロント `__mmdPlayVmd(url)` で yyb_miku に再生。
- フロント: 口パク継続・物理追従・idle↔clip クロスフェード・発話終了で復帰。
- 統合: `pwsh scripts/build-frontend.ps1`→ハードリロード、`python run_server.py`＋PC-B 起動、感情で clip 切替／PC-B 停止で idle 退避。

## PC-B 共有・運用（Git共有 ＋ Remote-SSH）
- コードは Git（`motion-service/` を pull で同期）、実行時データ(.vmd)は HTTP/WS。
- VS Code Remote-SSH（ホスト `AI_CEC_2026`）で PC-B 上を直接編集・常駐起動。
- `ckpts/` と SMPL ボディモデルは Git 管理外（`.gitignore` 済み）、PC-B で直接DL/配置。
- 詳細手順は `motion-service/README.md`。
