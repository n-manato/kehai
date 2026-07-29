# リアルタイム低遅延ロードマップ（ChatGPT Voice ベース）

> 2026-07-22 改訂。旧プラン `~/.claude/plans/chatgpt-voice-…-adaptive-meerkat.md` を
> 現状（gateway統合完了・streaming TTS稼働）と技術的確定事項で置き換えたもの。

## 0. ゴール

マイクボタン1回で ChatGPT Voice のように **低遅延で会話し続ける**。
体感を支配するのは **TTFA（発話終了→初音声）**。勝ち筋は2つ:

1. **ユーザーが話している間に ASR と LLM を先に走らせる**（発話終了後に残る仕事を最小化）
2. **全段を重畳**して初音声を最短化（プリロール撤廃・first-chunk 優先）

**目標 TTFA**: 発話終了→初音声 **0.3–0.7s**（2ターン目以降）。

---

## 1. 現状（確定・DONE）

- **転送**: WebSocket。マイクは発話を**丸ごとバッチ送信**（`mic-audio-data`×N → `mic-audio-end`）。
- **ASR**: `sherpa_onnx` SenseVoice、**CPUバッチ**。発話終了まで開始不可。非ストリーミング。→ **最大のボトルネック**。
- **LLM**: ✅ ローカル vLLM `Qwen3-30B-A3B-AWQ`（PC-B :8807）。gateway 経由。prefix-cache 有効（実測 hit 98%）。**TTFT ~0.16s**。既に速い。
- **TTS**: ✅ **streaming 稼働**。`gsv_stream.py` → gateway `/v1/audio/stream` → GPT-SoVITS :9880 raw PCM。**first-chunk ~1.1s、RTF ~0.5**。文単位・逐次。
- **motion**: GestureLSM（audio→SMPL-X、**whole-clip ~1.5–2.8s**）。文末発火・~5s間引き。**音声のcritical pathには乗らない**。
- **gateway 統合**: ✅ 完了。LLM/TTS/gesture 全て公開 Nexus gateway 経由、**SSH トンネル全廃**。（[gateway-streaming-consolidation メモリ]参照）
- **再生**: frontend streaming-audio-player（プリロール ~1.5s）。
- **ターン**: 半二重。発話ごと `create_task`、`frontend-playback-complete` を待って次ターン。

### 現状のレイテンシ内訳（発話終了→初音声）

| 段 | 現状 | 効かせ方 |
|---|---|---|
| 終端検出 | 固定無音待ち | セマンティック終端で短縮 |
| **ASR** | **バッチ・CPU・発話後に開始** | **ストリーミング化で発話中に隠す（最大効果）** |
| LLM TTFT | ~0.16s（発話後開始） | partial prefill で発話中に隠す |
| 文/句生成 | faster_first_response | 句単位でさらに前倒し |
| TTS first-chunk | ~1.1s | first-clause で前倒し |
| プリロール | ~1.5s | 詰める／撤廃 |

---

## 2. 技術的確定事項（重要な前提）

- **GestureLSM は audio 駆動 + whole-clip**。テキスト生成不可、causal/streaming 不可（毎回 ~1.5–2.8s で全再生成）。
  → **1ワードずつ音声を送る方式は物理的に無理**。真のストリーミング型（LiveGesture/UMo）は weights 未公開。
  → **解: 2層構成**（下記 Phase D）。
- **ASR 環境が CPU 専用**: `torch 2.10.0+cpu` / `onnxruntime` CPU版。PC-A は RTX 4060(8GB, 空き~6.7GB)あり。
  → GPU-ASR は「config 1行」では済まない。**CUDA 環境を隔離 venv で1回だけ構築**し、
    使い捨ての「SenseVoice を GPU化」は飛ばして **ストリーミング GPU ASR に直行**（GPU化と partial を一度に取る）。
- **streaming TTS は完成済み**。以降の TTS 改善は「さらに前倒し」のチューニングのみ。

---

## 3. ロードマップ（ROI 順・フェーズ）

### Phase A — 即効（CUDA不要・今日）

TTFA を数百ms削る安全な小改修。

- **A1. first-clause TTS**: LLM が最初の**句**（読点まで）を吐いた瞬間に合成開始。
  現状 `faster_first_response`（最初の“文”をカンマ分割）を、**最初の句を即 TTS に流す**よう調整。
  変更: `agent/transformers.py` / `utils/sentence_divider.py`（first-response の分割粒度）。
- **A2. preroll 詰め**: gateway ジッタは低い（stream 実測 95chunk 増分）ので frontend のプリロールを下げる。
  変更: `frontend-src/.../utils/streaming-audio-player.ts`（`prerollSec`）+ `task-queue.ts`。
  下げすぎ = underrun（途切れ）なので二分探索でチューニング。**要 frontend rebuild**。

→ 成果: バックエンド再起動＋frontend rebuild だけで初動が体感で軽くなる。

### Phase B — ASR コア（CUDA環境 + ストリーミング ASR）

**最大の効果**。発話中に文字起こしが走るようにする。

- **B1. CUDA 環境（隔離）**: 既存 CPU 環境を壊さないよう **別 venv** に CUDA 対応スタックを構築
  （torch+cu / onnxruntime-gpu / もしくは sherpa-onnx CUDA wheel）。4060(8GB)に収まる小型モデル前提。
- **B2. ストリーミング ASR エンジン**: SenseVoice(非streaming)を置換。候補:
  - sherpa-onnx **streaming zipformer**（GPU、低遅延、partial 出力）
  - NeMo **cache-aware FastConformer**（Parakeet-ja / Nemotron-streaming、JA 強い）
  `asr/` に interface 実装 + `asr_factory.py` 追記。
- **B3. 連続上り + サーバ VAD**: マイクを **20ms フレームで常時ストリーム**（`use-send-audio` を
  「発話後バッチ」→「常時ストリーム」へ）。`conf.yaml: vad_model: silero_vad` を有効化し、
  `websocket_handler._handle_raw_audio_data` + `vad/silero.py` の `StateMachine` を実配線。
- **B4. partial 字幕**: 安定 partial を WS/DataChannel で字幕表示。

→ 成果: ASR が発話中に走り、発話終了時にはほぼ確定テキストがある。

### Phase C — prefill コア + セマンティック終端

ASR の成果を LLM 側に繋ぎ、TTFT も発話中に隠す。

- **C1. partial で LLM prefill 先行**: 安定 partial を vLLM に投げてプロンプトを**発話中に prefill**。
  発話終了後は「最終 delta + decode」だけ残る。ASR/LLM が発話時間に隠れる。
- **C2. セマンティック終端検出**: silero の無音 endpoint に、ASR partial の**文末完結性**（pysbd）＋
  任意の軽量 LLM「ターン完了?」判定を組み合わせ、OpenAI `semantic_vad` を模す。誤割り込み・早期切断を抑制。

→ 成果: 発話終了後の残り仕事が最小化。TTFA が目標域（0.3–0.7s）へ。

### Phase D — モーション2層（ストリーミング音声を活かす現実解）

GestureLSM が streaming 不可なので**2層**にする。

- **D1. 手続き的 talking-motion（下層・0遅延）**: 再生中 PCM の **live-RMS/ビートを直接**
  頭の上下・体の揺れ・腕の動きへ毎フレーム反映。**ストリーミング音声をリアルタイムに motion 化**する本命。
  変更: 各 renderer（mmd/vrm/live2d）の update に RMS 駆動の base motion を追加。
- **D2. GestureLSM 窓・早撃ち（上層・意味的）**: 文末まで待たず、**最初の ~2–3s の PCM が溜まった時点で発火**。
  生成クリップが早く届き、手続きベースの上に重ねる（1窓分の遅れは下層が吸収）。
  変更: `tts_manager` の gesture 発火条件（累積 5s → 先頭窓 2–3s トリガ追加）。

→ 成果: 「即座に動く（手続き）」＋「来たら本格ジェスチャー（GestureLSM）」で破綻なく追従。

### Phase E — LLM decode 高速化（任意）

- **E1. speculative decoding**: 小 draft（Qwen3-0.6B）+ 30B ターゲットで batch=1 decode を 1.5–2x。
  VRAM(PC-B)+ 複雑さ増。効果が欲しければ着手。

### Phase F — full-duplex + サーバ barge-in（大工事）

- 発話ごと `create_task` → クライアント毎の永続 `ConversationSession`（常時 listen→endpoint→generate→speak-while-listening）。
- AI 発話中に上り VAD が発話検知 → **generation-id** を進め旧生成（LLM/TTS/再生）を破棄、履歴を聞こえた分だけ truncate。
- ダッキング（上り検知で AI 再生ゲイン低下、スピーカー自己割り込み抑制）。
- 変更: `conversations/`（新 ConversationSession）、`websocket_handler`、`use-interrupt`。

→ 成果: 「途中で割り込める」自然なターンテイキング。

### Phase G — WebRTC 全面移行（大工事・本命）

- サーバ `aiortc` peer。上り audio track → 48k→16k → Phase B の VAD/ASR。下り用カスタム MediaStreamTrack に TTS PCM。
- **DataChannel** に制御集約（partial字幕/actions/motion/turn/interrupt）。
- クライアント `RTCPeerConnection` + getUserMedia(AEC/NS/AGC)。`ontrack` → WebAudio **AnalyserNode でライブ RMS リップシンク**
  （＝「全ファイルから volumes 事前計算」を撤廃、ストリーミングの構造的ブロッカー解消）。
- **プリロール撤廃**。SRTP/UDP で jitter 最小。

→ 成果: SRTP双方向・ライブ analyser・サーバ barge-in で ChatGPT Voice 級 full-duplex。

---

## 4. 実行順序（推奨）

1. **Phase A**（今日・安全）: A1 first-clause TTS → A2 preroll 詰め
2. **Phase B**（承認のうえ）: CUDA隔離venv → ストリーミングGPU ASR → 連続上り+サーバVAD
3. **Phase C**: partial prefill → セマンティック終端
4. **Phase D**: 手続きモーション下層 → GestureLSM 窓早撃ち
5. **Phase E**（任意）: speculative decoding
6. **Phase F → G**（大工事）: full-duplex barge-in → WebRTC

各フェーズ後に実会話で TTFA を実測（サーバに `endpoint→ASR-final→LLM-TTFT→TTS-TTFA→first-frame` のタイムスタンプ log を追加し突合）。

---

## 5. リスク / 注意

- **CUDA導入（Phase B）**: 既存 CPU 環境（torch+cpu）を壊さないよう**別 venv 隔離**。4060 は 8GB なので ASR モデルは小型に。
- **preroll 下げすぎ（A2）**: underrun（音途切れ）。二分探索でマージン確保。
- **セマンティック終端（C2）**: 早すぎ＝発話途中で切る、遅すぎ＝もっさり。無音長＋文末完結のバランス調整。
- **WebRTC on Windows（G）**: aiortc ネイティブ依存（av/cffi）、要検証。
- **リップシンク再設計（G）**: 「事前 volumes」→「ライブ analyser」。全 renderer で調整。
- **secrets**: `.motion_gateway_key`（Bearer）、`conf.yaml`（gitignored・gateway キー直書き）を露出しない。

---

## 6. 変更ファイル早見

- **A1**: `agent/transformers.py`, `utils/sentence_divider.py`
- **A2**: `frontend-src/.../streaming-audio-player.ts`, `task-queue.ts`（要 rebuild）
- **B**: 新 `asr/streaming_*`, `asr_factory.py`, `conf.yaml`(vad/asr), `vad/silero.py`, `websocket_handler`, frontend `use-send-audio`
- **C**: `conversations/`（prefill フック）, `vad/`（セマンティック終端層）
- **D**: renderer 各 `update`（RMS base motion）, `conversations/tts_manager.py`（gesture 発火条件）
- **E**: PC-B vLLM 起動引数（draft model）
- **F**: 新 `conversations/ConversationSession`, `websocket_handler`, `use-interrupt`
- **G**: 新 `realtime/`（aiortc peer/track/session）, `server.py`/`routes.py`, frontend WebRTC サービス
