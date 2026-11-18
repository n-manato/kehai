# PC-B (aiccore) で KEHAI を開発するときに最初に読むもの

2026-08-06、PC-A (Windows) から移設。PC-A の install は**別インスタンスとして残っている**。

## この箱の制約（設計判断の理由がここに全部ある）

| 事実 | 何が決まるか |
|---|---|
| Nexus gateway が `*:3000` で稼働（同じ箱） | LLM/STT/TTS/motion は全部 **loopback**。SSHトンネルも Cloudflare もホットパスに無い |
| **サウンドカードが無い**（`arecord -l` = no soundcards） | ここでブラウザを開いても声は出ないし聞こえない。音の確認は PC-A 側でしかできない |
| 外から届くのは **:22 だけ**（`:3000` すら不通） | 利用は「ブラウザ PC-A ←ssh -L→ サーバ PC-B」固定 |
| その結果 **WebRTC が張れない** | aiortc の host candidate に届かず ICE 失敗 → フロントが WS 音声へ自動フォールバック。ブラウザ AEC が無いのでヘッドホン前提 |
| system python が 3.8（app は >=3.10） | venv は `/usr/bin/python3.11`。`.venv/bin/python` を使うこと |
| ディスクが常に逼迫（残り20GB台） | torch は **CPU wheel 固定**（silero VAD 専用でGPU不要）。CUDA版を入れると ~4GB 飛ぶ。`pip cache purge` で戻せる |
| GPU は vLLM / GSV / ActionPlan が占有 | KEHAI 本体は GPU を使わない。使おうとしないこと |

## 起動・確認

```bash
cd ~/Documents/DEV/KEHAI
./kehai.sh start|stop|restart|status|log     # status は :3000 gateway も叩く
.venv/bin/python scripts/e2e_probe.py "Say a short greeting." 40
```

`e2e_probe.py` が **PASS/exit 0** を返せば、WS → LLM → ストリーミングTTS → モーションクリップ
までの全経路が生きている。ここには音が無いので、これが唯一のシェルからの合否判定。
基準値（2026-08-06 実測）: TTFA 0.37s / audio 67KB / clip 241 frames。

見た目の確認が要るときは、この箱の Chrome を headless で `http://127.0.0.1:12394/` に
向ければスクリーンショットは撮れる（描画に音は要らない）。ただし対話の音は判定できない。

### LLM は Qwen3.8-27B（2026-10-23 から）

`scripts/pcb-vllm/serve-qwen38.sh` が :8807 で Qwen3.8-27B（4bit、思考オフ、MTP）を出す。venv と重みは
`AICCORE-Nexus/stack/vllm-next/` にあり、そこの `serve-qwen38.sh` はこのファイルへのシンボリックリンク
（Nexus は `stack/` を git 管理していないので、設定はこちらで管理する）。旧 id `qwen3-30b-a3b` を別名で
受けるので、gateway と conf.yaml はそのまま。旧モデル（Qwen3-30B-A3B）は容量のため削除済みで、
戻す手順は `scripts/pcb-stack-up.sh` のコメントにある。

- **e2e_probe の PASS だけを見ないこと。** 音声さえ出れば PASS になるので、LLM がエラーでもエラー文を
  喋って PASS する。`spoken` の行を必ず読む。
- Qwen3.8 の元のテンプレートは、会話の途中の system を 400 で拒否する。KEHAI は毎ターン時刻の system 行を
  入れるので、`scripts/pcb-vllm/chat_template_kehai.jinja`（途中の system を受け入れる版）を外すと
  全ターンが失敗する。
- 時刻の行は履歴にも残す（`basic_memory_agent._to_messages`）。残さないと前のターンのプロンプトが次の先頭に
  ならず、GDN 混合型の prefix cache が毎ターン外れる。
- 接続したとき・履歴を読んだときに、システムプロンプトと履歴を裏で先読みする
  （`warm_prefix`、`LLM_PREFIX_WARM=0` で無効）。実測の TTFA は最初のターンから約1.0〜1.5s。
- `start_server.sh` は `LLM_BASE_FALLBACK=""`。公開 gateway はこの箱自身なので切り替えても無意味なうえ、
  切り替えが有効だと12秒の読み取りタイムアウトが付き、長いプロンプトの読み込みで5分間 Cloudflare 経由になった。
- Rem（システムプロンプト約1.2万トークン）は、キャッシュが空なら読み込みに約10秒かかる。キャラを切り替えると
  先読みが裏で走るので、切り替え直後に話しかけなければ待たされない。
- 先読みはターンと**同じシステムプロンプト**でないと効かない。ターンごとに `apply_client_language` が
  言語の指示を末尾に足すので、履歴を読む前にそれを当てている（`_resume_latest_history`）。
  当てずに先読みすると最初のターンだけ履歴を全部読み直す（2026-10-24: 最初の文まで 2.8s → 0.65s）。

### 話しかけてから返答まで（2026-10-24 に直したところ）

- **割り込みはサーバーの VAD が決める。** 発話の始まり（`<|PAUSE|>`、68dB の関門を通ったもの）で
  サーバーが `{"type":"control","text":"user-speech-start"}` を送り、描画タブはそれで返答を止める。
  描画タブ自身の VAD は関門が無いので、server-VAD モードでは割り込みも「考え中」への切り替えもしない。
  （関門の下の声で彼女が止まり、そのまま誰も答えない、が起きていた）
- 割り込みで再生を止めるのは `streamingControl.stopNow()`（`use-interrupt.ts` から直接）。
  `aiState === 'interrupted'` の effect 頼みだと、'interrupted'→'listening' がまとめて描画され、止まらなかった。
- `[TTFA]` は先読みの返答をゲートが閉じている間は測らない。HIT で音声を溜めていたときは
  `[speculative HIT, ...]` 付きで、解放した時刻で出る。
- 無人の試験は `/render-stream/mic-inject` に 48kHz int16 の kind 3 フレームを実時間で流す
  （GSV の声は ×4。×0.4 だと関門 68dB の下で、「止まらないこと」の試験になる）。
  描画タブ（CDP :9222）で `AudioBufferSourceNode.start/stop` と `AudioContext.close` をフックすると、
  割り込みで音が本当に止まったかを確認できる。試験の発話は履歴に残るので、終わったら
  `truncate_history` で戻してタブを再読み込みする。
- **GSV のパッチ**（`scripts/pcb-gsv/defer-empty-cache.patch`、GPT-SoVITS は Nexus の git 外）。
  リクエストの最後の `empty_cache()`（gc.collect）が最後の音声のあと約460ms ストリームを開けたままにし、
  次の文はその後ろで待たされていた。文の切れ目ごとに無音が入っていた。2秒アイドルで実行に変更。
  GSV を作り直したら `git apply` し直すこと。
- GSV の再起動: `keepalive.sh` は cron に**入っていない**（gateway が呼ぶ）。gateway は :9881 が生きていると
  起こさないので、手で `bash stack/voice/keepalive.sh` を走らせる。そのとき自分のコマンドラインに
  `api_v2.py .*-p 9880` が含まれていると keepalive の `pgrep` がそれを拾い、起動しない。
  GSV を再起動したら KEHAI も再起動する（KEHAI はキャラの重みを適用済みと覚えていて送り直さない）。

### 脳の切替: `./kehai.sh mode {cascade|omni|status}`（2026-11-02 から）

| | cascade（元の構成、戻し先） | omni |
|---|---|---|
| 聞く | Whisper → 文字 | Whisper は**区切り・意味的エンドポイント・先読みの一致・履歴**にだけ使い、**理解は Qwen3-Omni が発話音声そのものから**行う（ASR の取り違えが返答に影響しない） |
| 考える | Qwen3.8-27B（vLLM :8807、TP=2） | **Qwen3-Omni-30B-A3B-Instruct**（Thinker INT4 AutoRound、vLLM-Omni :8808、**GPU1 単独**） |
| 喋る | GPT-SoVITS（Aura） | Stage 1: GPT-SoVITS のまま／Stage 2（`OMNI_STAGE=2`）: Talker（声はプリセット Chelsie/Ethan/Aiden、GPU0） |
| 体 | KAWAII 辞書 + GestureLSM + ActionPlan | KAWAII 辞書のみ（GestureLSM と ActionPlan は停止、GPU1 を空ける） |

- 切替は `scripts/pcb-mode.sh`。止めるもの（vLLM :8807 / ActionPlan、gateway 管理の GestureLSM・GSV は **DB の `enabled`** で）→ 新しい脳を起こす → `.mode` に書く → KEHAI 起動。`.mode` は `start_server.sh`（`KEHAI_MODE`、`LLM_PROVIDER_OVERRIDE=openai_omni_llm`、`LLM_CTX_*`）と `scripts/pcb-stack-up.sh`（@reboot）が読むので、再起動後も同じモードで上がる。**conf.yaml と characters は両モード共通**（`llm_configs.openai_omni_llm` を足しただけ）。
- **omni の起動に失敗したら自動で cascade に戻る**（箱を脳なしで放置しない）。成功判定は `:8808/v1/models`。
- 起動確認は 1 行目 `MODE=omni (llm=openai_omni_llm)` と `./kehai.sh mode status`。無人の合否は `scripts/omni_probe.py <wav>`（マイク経路: 16 kHz フレームを実時間で流し、聞き取り・初音声・モーションを見る。cascade でも同じ wav で比較できる）。`e2e_probe.py`（文字入力）はどちらのモードでも動く。
- vLLM-Omni 側: venv は `AICCORE-Nexus/stack/vllm-omni/.venv`（uv、Python 3.12、vLLM 0.32.0 **+cu129**（Driver 570 は cu130 不可）、vllm-omni 0.32.0）。重み `stack/vllm-omni/models/qwen3-omni-int4-autoround`（26 GB、Thinker だけ INT4、Talker/Code2Wav は bf16 同梱）。起動スクリプトは `scripts/pcb-vllm/serve-qwen3-omni.sh`（Nexus 側はシンボリックリンク、`serve-qwen38.sh` と同じ流儀）。
- **踏んだ罠（2026-11-02、全部スクリプトに織り込み済み）**:
  1. vllm-omni は Qwen3-Omni に MoE バックエンド `triton` を強制し、それは INT4 MoE の expert bias を扱えず起動で落ちる → `moe_backend: marlin` を明示。
  2. 重みだけで **18.0 GiB**。共有カード（GSV 2.1 GiB が同居）ではプロファイラの KV 見積りが外れて OOM → `kv_cache_memory_bytes` を明示（2.0 GB、16k 文脈に必要な 1.5 GiB を確保）。`gpu_memory_utilization 0.90`、`max_num_batched_tokens 4096`。
  3. `pkill -f` の自殺、再び: 切替スクリプトが `serve-qwen3-omni.sh` を名前で殺したら、その名前を**コマンドラインに含むだけ**の呼び出し側シェルまで死んだ。パターンは `^bash \./serve-qwen3-omni\.sh` と `[v]llm serve.*--port 8808` のように**固定の形**に。切替を呼ぶときも、コマンド文字列にそれらの語を書かない。
  4. GSV の生存確認は `/healthz` ではなく **`/docs`**（api_v2.py に /healthz は無い＝404）。
  5. 起動を見張るとき「プロセスが消えた」は、`vllm serve` の行が**一度見えてから**しか信用しない（ラッパーが exec する前の数秒を「死んだ」と誤判定して cascade に戻し、起動中の omni と GPU1 を取り合って OOM にした）。
  6. GSV（2.1 GiB）が GPU1 に同居すると、Thinker は**起動して warm-up 2 回に答えたあと最初の本番 forward で OOM** し、以後すべてのターンがエラー文を喋った（`/v1/models` は生きたまま）。→ omni では **GSV を GPU0 へ**（`stack/voice/.gsv_gpu` を keepalive.sh が読む。cascade は 1 のまま）、成功判定は `/v1/models` ではなく**実際の生成**（`omni_smoke`）。
  7. gateway の worker は音声スタックの生死を **voice_service（:9881）** で見るので、GSV（:9880）だけ殺しても起こし直さない。移すときは `keepalive.sh` を自分で走らせる（冪等）。GSV を起こし直したら KEHAI も再起動（重みの適用済みを覚えている — 起動直後は `[gsv-weights] 502` が出て、次のターンで `loaded` になる）。
- **実測（Stage 1、2026-11-02）**: 起動 2 分強（重み読み込み 38 s、KV 20,336 トークン）。vLLM-Omni 直叩き TTFT 0.12 s（文字）/ 0.13 s（音声入力）。KEHAI 経由の「話し終わり→最初の音」0.57〜1.19 s（先読み HIT 込み、cascade と同じ帯）。Whisper が "why CCORE" と聞き違えた発話でも Thinker は音声から「AI-CCORE の今週の取り組み」と理解して答えた。GPU1 21.6 GB（Thinker 単独）、GPU0 7.8 GB（Chrome 3.9 + Whisper 1.8 + GSV 2.1）。
- **品質の注意**: 7B 級の Thinker はペルソナ追従が Qwen3.8-27B より弱い（自分の knowledge cutoff を語る、存在しない人名を出す例があった）。履歴にエラー文が残ると次の返答に引きずられるので、試験のあとは `truncate_history` で戻す。

### 喋り始めを早くする（2026-11-03、omni モードのみ）

目的は「話し終わり→声」。内訳は **エンドポイントの無音待ち**＋LLM の最初の文＋TTS の最初の音。omni では次の 3 つで
**1.5〜2.1 s → 0.60〜0.63 s**（`scripts/omni_probe.py` で同じ 3 つの wav を計測）。残りはほぼエンドポイントの 0.61 s そのもの。

1. **先読みを音声で即開始**（`SPECULATE_AUDIO_FIRST=1`）: 区切り（0.19 s）で Whisper を待たず、ここまでの音声で返答を始める。
   記憶と履歴には `[spoken — transcript pending]` を置き、Whisper の文字が来たら差し替える（`speculative.set_text`、
   `BasicMemoryAgent.replace_user_text`、`chat_history_manager.replace_message_content`）。文字が junk / 言いかけなら破棄。
   HIT 判定は文字のまま（エンドポイント側は `text_ready` を最大 1.5 s 待つ）。
2. **エンドポイント 0.9 → 0.61 s**（`VAD_END_MISSES=13`、`silero.py` の env 上書き）。cascade では ASR 待ちで先読みが
   間に合わなくなるので cascade は 22 のまま。
3. **Whisper は turbo + greedy**（`ASR_MODEL_OVERRIDE=faster-whisper-large-v3`、`ASR_BEAM_OVERRIDE=1`）。理解は Thinker が
   音から行うので、文字はターン制御と履歴にしか使わない。large-v3 + beam 5 だと TTS と GPU0 を取り合って 0.6 s 超になり、
   HIT しても解放が文字待ちで 0.4〜0.5 s 止まった。warmup も同じ id を温める。
4. **TTS を Qwen3-TTS に**（`OMNI_TTS=qwen3tts`、既定はまだ `gsv`）: vLLM-Omni :8809（GPU0、`scripts/pcb-vllm/serve-qwen3-tts.sh`、
   `stack/qwen3-tts/models`）。キャラの **GSV 参照クリップと転写をそのままゼロショットでクローン**（`service_context.gsv_stream_config`
   が `Qwen3TTSStreamConfig` を返す、`tts/qwen3tts_stream.py` が 24→32 kHz にして同じ契約で流す）。最初の音 0.09〜0.13 s
   （GSV 0.36〜1.16 s）、実時間の 4 倍。`qwen-tts` の Python パッケージは**ストリーミング無し・実時間の 1.76 倍**で使えない。
   サーバーには `--allowed-local-media-path stack/voice/refs` が必要（無いと 500 で全文無音）。
- 罠: **`pkill -9 -f '^VLLM::'` を手で打たない**。Thinker・Qwen3-TTS・（cascade の）vLLM のワーカーは全部この名前で、
  TTS の残骸を掃うつもりで Thinker を殺した（2026-11-03）。`pcb-mode.sh` の `reap_workers` は 3 つのサーバーが全部落ちている
  ときだけ掃く。
- 声の判断材料は `cache/omni_tts_test/`（サーバー停止で消える）に同じ文の Qwen3-TTS 版と GSV 版を置く。

### Qwen3-TTS を Aura で学習した（2026-11-03、ゼロショットは文ごとに声が揺れたため）

- 公式レシピ（`Qwen3-TTS/finetuning`、単話者 SFT）をこの箱向けに写した `stack/qwen3-tts/finetune/sft_12hz_pcb.py`:
  8-bit AdamW（bitsandbytes）・sdpa・勾配チェックポイント・最新チェックポイント 1 つだけ保持。データは GSV と同じ
  `aura_gsv.list`（828 クリップ 89.7 分）→ `finetune/aura/train_raw.jsonl`（`ref_audio` は全行 `aura_ref1_24k.wav`、**24 kHz 必須**）→
  `prepare_data_pcb.py`（コーデック化、バッチ 2。既定バッチは GPU0 で OOM）→ 学習 batch 1 × 累積 8、lr 1e-5、4 エポック、
  話者名 `aura`。**GPU0 でピーク 11.5 GiB、20 分**（Chrome / Whisper / GSV と同居、PROTO は止めずに済んだ）。損失 14.6 → 6.3。
- 成果物 `finetune/aura/output/checkpoint-epoch-3`（`tts_model_type: custom_voice`、`spk_id: {aura: 3000}`、3.6 GB）。
  `copytree` は safetensors を写さないので `speech_tokenizer/model.safetensors` を Base からリンクしないと配信の codec 段が落ちる。
- 配信: `models/current` → そのチェックポイント、`models/current.speaker` = `aura`。`serve-qwen3-tts.sh` は `models/current` があればそれを出し、
  `start_server.sh` は `current.speaker` があれば omni の既定を `OMNI_TTS=qwen3tts` + `QWEN3TTS_SPEAKER=aura` にする
  （`tts/qwen3tts_stream.py` は speaker があると `task_type: CustomVoice` + `voice`、参照クリップ無し）。
- 実測: 最初の音 **0.05〜0.07 s**（ゼロショット 0.11〜0.13、GSV 0.36〜1.16）。
- **2026-11-03 22:30 から既定は GSV に戻してある**（文ごとに声が揺れる: f0 104〜237 Hz。「話しかけられているかを omni が見る」の節末尾）。Qwen3-TTS にするのは `OMNI_TTS=qwen3tts ./kehai.sh restart`。別の声を学ぶときは `train_raw.jsonl` の
  `ref_audio` を 24 kHz に、`--speaker_name` を変え、`models/current` を張り替える。学習中は GPU0 を 11.5 GiB 使うので、
  その間だけ `:8809` を止める（`kill <pid>`、名前で pkill しない）。
- 制約: 英語・日本語とも Thinker は喋れるが Stage 1 の声は GSV（Aura）のまま。文脈は 16k（Rem の 1.2 万トークンのペルソナは入るが余裕が薄い）。ツール（time / ddg-search）は hermes パーサ経由。初音声は cascade と同じ帯（E2E でも速くはならない、調査結果は `~/.claude/plans/speach-to-speach-mossy-pancake.md`）。

### GSV と Qwen3-TTS の速さと声の安定（2026-11-03 深夜、ユーザー「GSV で同じ速さが狙えるなら GSV、無理なら Qwen3-TTS」）

- **速さは Qwen3-TTS**。最初の音: Qwen3-TTS 0.05〜0.06 s、GSV 本番 0.50〜0.61 s（単体 0.21 s。`streaming_mode=true` は**文全体を合成してから**
  最初のチャンクを出す: "Hello." 0.21 / 7 語 0.78 / 20 語 1.57 s）。同じ 3 問の話し終わり→声: Qwen3-TTS 0.60 / 0.61 / 0.67、GSV 0.64 / 1.16 / 1.73
  （3 問目は Thinker の最初の文が 1.15 s かかった分も含む）。GSV の `streaming_mode=3`（AR デコーダからチャンク単位）なら最初のチャンクは
  文の長さに関係なく 0.18〜0.22 s（本番では +0.05〜0.1）。それでも Qwen3-TTS より 0.15〜0.2 s 遅い。`GSV_STREAMING_MODE=3` で試せる
  （`tts/gsv_stream.py`、既定は従来の true。GSV の文書では「品質は下がる」、耳で判断）。
- **声の安定は GSV**。同じ 8 文の文ごとの中央 f0: GSV 150〜183 Hz（幅 33）。Qwen3-TTS（checkpoint-epoch-3、CustomVoice aura）は
  seed 1234 で 104〜237（"Hello." が 104 Hz）。seed で大きく変わるが**汎化しない**（seed 11 は A 組 31 Hz / 別の B 組 178 Hz）。
  B 組で最良の seed 5 / 42 でも 57 / 63 Hz。文の長さも揺れ（"Good evening." が 16 s に伸びる seed あり）、seed 21 / 77 は **680 codec
  トークンの上限まで EOS を出さず**（`Qwen3TTSCodecLimitError`、音声が途中で切れる）。
  効かないもの: `temperature` / `top_k`（speech エンドポイントは無視、出力が同一）、`instructions`（f0 幅 57 Hz になるが 7 語が 7 s に伸びる）、
  参照クリップ付き ICL（CustomVoice checkpoint は `task_type=Base` を 400 で拒否）。
- **結論**: 速さのルールでは Qwen3-TTS だが、今の checkpoint は声が本番品質でない → **学習を足すまで既定は GSV**。
  合否試験は同じ 16 文（A 組 8 + B 組 8）で f0 幅 ≤ 50 Hz・runaway 0（スクリプトはセッション記録、要点は「40 ms 窓の自己相関で中央 f0」）。
  比較 wav は scratchpad `voice/`（セッション限り）。

### Qwen3-TTS の学習↔試験の周回（2026-11-03 深夜〜、ユーザー「使えるようになるまで試行錯誤して」）

- **合否**: `scripts/qwen3tts_gate.py`（:8809 の CustomVoice を 16 文で測る。A/B 各組で f0 幅 ≤ 50 Hz・runaway 0・4 語以上の文は 6〜30 字/秒。
  `--gsv` で同じ 16 文を GSV に通して比較、`--extra '{"temperature":0.5}'` で `extra_params` を試せる）。epoch-3 の結果: A 133 Hz / B 88 Hz で FAIL、
  温度を下げても直らず（0.5/top_k 20 で runaway、0.3 で 5 字/秒に間延び）。
- **checkpoint を 1 つ試す**: `scripts/pcb-vllm/qwen3tts-gate-ckpt.sh <checkpoint dir> [gate 引数]`（:8809 に配信 → gate → PID で停止。
  speech_tokenizer の safetensors リンクも張る）。学習中は GPU0 に配信を同居させられない（学習 11.5 GiB + 配信 8.8 GiB + Chrome/Whisper/GSV）。
- **学習**: `stack/qwen3-tts/finetune/sft_12hz_pcb.py` に `--resume_from <ckpt>`（talker の重みだけ差し替え、speaker encoder は Base）・
  `--epoch_offset`・`--save_every` を足した。例（r1 = epoch 4〜6、epoch-3 から再開）:
  `CUDA_VISIBLE_DEVICES=0 .venv-train/bin/python finetune/sft_12hz_pcb.py --init_model_path models/Qwen3-TTS-12Hz-1.7B-Base --train_jsonl finetune/aura/train_with_codes.jsonl --output_model_path finetune/aura/output_r1 --batch_size 1 --grad_accum 8 --lr 1e-5 --num_epochs 3 --speaker_name aura --resume_from finetune/aura/output/checkpoint-epoch-3 --epoch_offset 4 --save_every 3 --keep_last 1`
  **`python -u` で起動すること**（素の python だとログがバッファされ、終わるまで何も見えない）。1 エポック約 5 分、保存は 3.6 GB/個、ディスクは 13 GB 空き。
  `models/current` が指す checkpoint は消さない（配信が壊れる）。
- 学習前に :8809 を止める: `kill $(pgrep -f '[v]llm serve.*--port 8809')`。終わったら `./kehai.sh up`（stack-up が :8809 を起こす）か gate-ckpt.sh。
- データ: 828 クリップ 89.7 分（32 kHz、1.5〜25 s）、codec は 12.5 個/秒で整合、ref は `aura_ref1_24k.wav`。epoch ごとの損失 7.85 → 6.89 → 6.28 → 5.85（まだ下がり途中）。
- **周回の記録（2026-11-04 未明）**:
  - r1（epoch 4〜6、元データのみ）: "Hello." 104 → 199 Hz に改善、しかし "Yes." 333 Hz・"Okay." 暴走。**学習データに 12 字未満の文が無い**のが原因
    → GSV で短文 131 本を合成（`finetune/make_short_set.py` → `train_short_raw.jsonl`、×2 で混ぜる）。
  - r2（epoch 7〜9、短文入り）: 声の高さは揃った（A 組 幅 17 Hz）が **16 文中 8 文が暴走**（EOS を出さず上限まで大声で喋り続ける、無音ではない）。
  - 検証 40 行（`val_with_codes.jsonl`、`--val_jsonl`）: 損失は epoch 4 で底（5.48）、以後上昇 = 過学習。ただし暴走は平均損失に出ない。
    `--eval_only` で **EOS の確率**を測る: epoch-3 は真の終端で p_end=0.44（温度 0.9 の抽選では半分止まらない）、他の位置 p_false=0.022。
  - 推論側の対策は全部不発: 温度/top_k（`extra_params` で効くが暴走が増える）、seed（汎化しない）、SFT talker + 参照クリップの hybrid
    （`finetune/make_hybrid.py`、暴走が増えるだけ）。**素の Base + 22 s 参照（`aura_qwen3tts.wav`+.txt）が最良**: 幅 60 / 32 Hz、暴走 0、
    最初の音 0.11 s（学習なしで使える次善。`models/current` を Base に向け `current.speaker` を消せば `service_context` が ICL で使う）。
  - r4（進行中）: epoch-3 から 6 エポック、短文入り、**`--eos_weight 2.0`**（終端位置の CE を加算）。epoch 6 と 9 を保存して gate。
  - 本番の保険: `tts/qwen3tts_stream.py` が `max_new_tokens = 15 + 2.3×文字数` を送る（暴走しても 2〜3 s で切れる）。
  - r4（EOS 損失 ×2）: 検証では p_end=0.999 になったのに暴走は 7/16。暴走の中身を Whisper で書き起こすと「still here… **I don't know**」—
    学習データの短文（"I don't know." は短文セットにある）をつなげて喋っている = **丸暗記**。教師強制の EOS 確率では自由生成の暴走は測れない。
  - 参照クリップの作り直し（`refs/aura_calm.wav`+.txt、f0 中央 178 付近・抑揚の小さい 3 クリップ 17.5 s）: Base ICL で B 組は通る（32〜42 Hz）が
    A 組の "Hello." が 270 Hz。ゼロショットは短い感嘆文で揺れる。
  - r5: **LoRA**（`--lora_r 16 --lora_alpha 32`、talker と code_predictor の q/k/v/o/gate/up/down 計 231 本、19.2M パラメータ、lr 2e-4、
    短文 ×1 の 919 行、4 エポック、ピーク 4.1 GiB）。保存時に重みへ merge するので配信は従来どおり。
    → 結果: 声の高さは揃う（幅 15〜35 Hz）が**中身が言葉にならない**（Whisper で書き起こせない、87〜92% が有音のまま 15〜45 s）。
    Base は CustomVoice 形式（参照無し・話者ベクトルのみ）を知らず（損失 14.8 から始まる）、低ランクでは形式を学び切れない。
  - r6（epoch-3 に LoRA、lr 5e-5、3 エポック）: 幅 53〜65 Hz・暴走 4/16（短文 "Hello." "Okay." "Yes."）。検証損失は最良（5.34）。
  - 全変種に共通: **CustomVoice では短い文で止まらない**。Base の ICL（参照の文字と音声が前に付く）は一度も暴走しない。
  - r7（進行中、Base から 4 エポック、1749 行）: (1) Thinker が書いた多様な 915 行を Base ICL で Aura 声に合成し f0/速さで選別した 838 行
    （`finetune/make_synth_set.py`、`refs/aura_qwen3tts.wav` 参照）を足して丸暗記を薄める、(2) 全クリップに**無音の前後付け**（前 0〜0.3 s、
    後ろ 0 または 0.2〜0.9 s、`finetune/augment_silence.py`）で「話し終わって無音が続いたら EOS」を学ばせる。
    罠: 増強した jsonl の `ref_audio` は必ず `aura_ref1_24k.wav`（dataset.py が 24 kHz 以外を assert で落とす。`train_raw.jsonl` は 32 kHz の
    ref1 を指しているので、そこから作ると落ちる）。
  - r7 の結果: 声の幅 39/25 Hz と最良だが **16 文中 15 文が暴走**（無音ではなく有音のまま上限まで）。検証の p_end は 0.09 に低下。
  - **原因の特定（2026-11-04 02:30）**: `finetune/swap_parts.py` で r7e3 の **サブトーカー（`talker.code_predictor`、codebook 1〜15）だけ Base に戻す**と
    16/16 止まる（epoch-3 でも同じ）。学習したサブトーカーの出力が talker に戻って文から外れる。サブトーカーの抽選を冷やしても
    （`generation_config.json` の subtalker_temperature 0.3）変わらないので、抽選のノイズではなくサブトーカー自体が壊れている
    （検証の sub 損失が 7.1 ≈ 一様分布のまま動かない）。学習側の疑い: code_predictor への gradient checkpointing か 8-bit AdamW。
    → 学習スクリプトに `--sub_lr`（0 = 凍結）と `--sub_loss_weight` を追加。
  - Base のサブトーカーに戻した talker の声の幅: epoch-3 63/86 Hz、r7e3 74/114 Hz（短い感嘆文が 240〜276 Hz に上がる）。
  - r8（talker のみ、Base から）: epoch-1 で **16/16 止まる**、幅 61/114（"Good evening." 276）。epoch-3 の保存はディスク不足で失敗
    （`No space left on device` — checkpoint 3.6 GB、常に 7 GB 以上空けておく）。
  - r9（r8 epoch-1 から +4 エポック、talker のみ）: 検証損失は epoch 2〜3 が底（4.12）、epoch-3 が最良: 幅 A 51 / B 99 / C 99 Hz、
    **暴走 0**（seed 6 種 × 温度 2 種 × 24 文 = 192 回で 0）。残る揺れは**短い挨拶だけ**（"Hello." "Good evening." "Good morning." "Oh, hi!"
    が 240〜306 Hz）で、seed・温度では消えない。長い文は 180〜200 Hz。`models/current` → `output_r9/checkpoint-epoch-3`。
  - r10（挨拶 170×3 ＋短文 121×2 ＋一般 600、r9e3 から lr 5e-6 で 2 エポック）: 変わらず（"Good evening." 258、"Good morning." 254〜276）。
    librosa.pyin でも同じ（Qwen3 "Good evening." 275 Hz、GSV の教師クリップは 159、Aura ref 169）= 推定器の誤りではなく本当に高い。
    長い文は 186〜205 Hz。**挨拶の短文だけ明るく出る癖は talker の学習では直らない** → サブトーカー側（r7 では学習したサブトーカーが
    f0 を揃えた）を極小の学習率で動かす r11（talker 凍結、`--freeze_talker --sub_lr 2e-6 --no_ckpt_sub`、1 エポック）。
  - r11（サブトーカーのみ lr 2e-6）: 変化なし。r12（サブトーカーのみ lr 1e-5、checkpointing 無し・torch AdamW）: **止まる性質は壊れない**
    （壊していたのは code_predictor への gradient checkpointing か 8-bit AdamW）が、声はむしろ高く散る（"See you later." 369 Hz）。
  - **結論・採用（2026-11-04 06:00）**: `models/current` → **`output_r9/checkpoint-epoch-3`**（talker のみ学習、サブトーカーは Base）。
    暴走 0、長い文 186〜205 Hz（pyin）、短い挨拶単独は 240〜300 Hz のまま → 本番は **`tts_manager` が 16 字未満の文を次の文につなぐ**
    （`QWEN3TTS_MERGE_SHORT`、Qwen3-TTS のときだけ。つないだ 8 文は 176〜216 Hz、"Hi there!" だけ 264）。返答の最後に残った短文は
    `flush()` で単独で喋る。`start_server.sh` の omni 既定は Qwen3-TTS に戻した（`OMNI_TTS=gsv` で GSV）。
    最終判断は耳（PROTO で聞く）。戻すのは `OMNI_TTS=gsv ./kehai.sh restart`。
  - 片付け: 学習用 wav（aug_main 366 MB、aug_synth 80 MB、synth 64 MB、short/greet）と jsonl は残してある。checkpoint は r9e3 のみ（3.6 GB）。
  - **学習データの内訳**（r9e3 が見たもの）: Aura の GSV 用クリップ 791 本 90.9 分 ＋ GSV 合成の短文 121 本 3.5 分 ＋ Base＋参照で合成した
    多様文 837 本 28.4 分 ＝ 1749 行 122.8 分（別に検証 40 行）。全クリップに無音の前後付け。
  - **再生の途切れ（06:00 に修正）**: vLLM-Omni の既定 `initial_codec_chunk_frames: 1`（80 ms）→ `codec_chunk_frames: 25`（2 s）だと、
    最初の 80 ms のあと次のチャンクまで 430 ms 空き、描画タブの再生が 0.08 s 地点で 230〜350 ms 止まっていた（`[audio-health] gaps=1/…@0.08s`）。
    `stack/qwen3-tts/qwen3_tts.deploy.yaml`（パッケージの yaml の写し、3 + 6 フレーム）を `serve-qwen3-tts.sh` が優先して読む。
    実測: 0.05 s に 80 ms、0.10 s に +160 ms、以後 480 ms ずつ約 100 ms 間隔。
  - 短文つなぎの待ち時間は `QWEN3TTS_MERGE_HOLD_S`（既定 0.25 s）で上限。次の文が来なければ単独で喋る。ログ `short sentence merged … after N ms`。

### 話し終わり→声を詰めた（2026-11-04 06:20、omni モード）

計測は `scripts/omni_latency_bench.py --wavs q1 q2 q3 --cam 写真 --runs 2`（プローブの「最後の発話フレーム→最初の audio-stream-chunk」と、
ログの区切り→先読み開始、ASR 追いつき、エンドポイント→最初の PCM をまとめる。終わったら履歴を `truncate_history` で戻す）。

| 構成 | 話し終わり→音声（中央値、6 ターン） | 内訳 |
|---|---|---|
| 従来（`VAD_END_MISSES=13`、ASR を待って解放） | **0.62 s**（0.59〜0.83） | 区切り 0.19 + エンドポイント 0.41、ASR は区切りから 0.40 s、エンドポイント時点で音声は用意済み（3〜8 ms） |
| **今**（`VAD_END_MISSES=9` + `SPEC_FAST_RELEASE=1`） | **0.54 s**（0.46〜0.64、2 周目 0.46〜0.55） | 区切り 0.19 + エンドポイント 0.28、解放は ASR を待たない。残りは Thinker の最初の文（区切りから約 0.5 s）待ち |

- **fast release**（`websocket_handler._finalize_vad_turn` 冒頭）: 先読み（音声で開始済み）がエンドポイントまで生きていれば、ASR の文字を待たずに
  解放する。文字は後から来て、パネル表示（`user-input-transcription`）・履歴の差し替え・`last_vad_turn` を埋める。文字が雑音
  （junk / 幻聴の定型句 / CJK）なら **撤回**（`speculative.discard` で返答を止めて記憶と履歴を戻す）。0.7 s 未満の発話は従来経路（短い物音が幻聴の元）。
  カメラ判定「彼女宛てでない」なら解放しない。
- エンドポイントは 13 → **9 miss**（区切りから 0.28 s、最後の声から約 0.47 s）。q2 の文中の間（"What time is it? / Right now in Omaha?"）は切れない。
- **相槌は Qwen3-TTS では既定オフ**（`BACKCHANNEL`、start_server.sh）: fast release で毎ターン送られ、0.56 s の「Mm-hm.」が返答の頭と重なった。
- Thinker 側: vLLM metrics（`:8808/metrics`）で TTFT は 0.1〜0.25 s が 634/817、prefix cache 命中 92%。`_trim_to_context` は格子で切るので
  長い履歴でも prefix は保たれる。プローブは接続のたびに新しい文脈なので TTFT が 0.5 s 超に出ることがある（本番の連続会話はもっと速い）。
- **耳までの残り**: 描画タブの preroll 50 ms（`websocket-handler.tsx:680`、変えるにはフロント再ビルド）＋ 描画→ビューワの hop ＋ ビューワの
  ジッタ緩衝（床 40 ms、starve ごとに +40、今は 4 s の平穏ごとに −10 ms に短縮。`render_view.html` はページ再読み込みで反映）＋ 出力デバイス 124〜140 ms（PROTO 側、こちらから変えられない）。
- 今の床: 区切り 0.19 s ＋ Thinker の TTFT 0.1〜0.25 ＋ 最初の文 0.1〜0.15 ＋ TTS 0.05 ≈ 0.45〜0.55 s。これより下は「区切りの前に喋り出す」しかない。

**06:55 の追加（独立レビュー 4 読者＋批評の結果を実装）**

| 経路 | 話し終わり→最初の PCM（6 ターン中央値） |
|---|---|
| 素のクライアント（プローブ → /client-ws） | **0.50 s**（0.46〜0.53） |
| **PROTO 経路**（ビューワのマイク 48 kHz フレーム → mic-inject → サーバー VAD 直結） | **0.54 s**（0.53〜0.56）。批評の推定では直結前は ≈0.70 s |

- **計測の打刻バグ**: fast release でゲートが開いたあとの最初の PCM を誰も打刻せず、`[TTFA]` は 2 文目の時刻を出していた（06:13〜06:19 の
  「endpoint→first_pcm 594〜785 ms」は幻）。`tts_manager` でゲートが開いていれば打刻（note `first PCM after release`）、`[TTFA]` に
  `wall endpoint@ first_pcm@` を追加。本当の値は 0〜44 ms（音声はエンドポイントでほぼ用意済み）。
- **VAD_VIEWER_DIRECT=1**（omni 既定）: サーバー VAD を描画タブ往復のコピー（81〜111 ms 遅れ、相互相関で実測済み）ではなく、ビューワの
  kind-3 フレームそのもので回す（`render_stream.hub.mic_sink` → `websocket_handler._viewer_mic_pcm`、63 タップ FIR + 3 倍間引き、512 サンプル詰め）。
  描画タブの `raw-audio-data` は生存確認だけ。`mic-inject` の kind 3 も同じ sink に入るので無人試験が同じ経路になる。
- **描画タブの preroll 0**: フロントに `localStorage.streamPreroll` の knob があった（`websocket-handler.tsx:678`）。CDP で `'0'` を入れて再読み込み済み
  （再ビルド不要）。戻すのは `localStorage.removeItem('streamPreroll')`。
- 文字待ち 1.5 s のタイムアウト時は先読みを捨てずに HIT 扱い（以前は冷たい再生成 1.7〜2.4 s）。
- 区切りの look は、その発話が既に「彼女宛て」と判定済みなら撮らない（エンドポイント直前の look で settle が 0.3 s 待つ経路を消す）。
- **無人計測**: `scripts/omni_latency_bench.py --via-viewer`（プローブが 48 kHz kind-3 を mic-inject に流し `speech_end_wall` を出し、
  ログの `first_pcm@` と突き合わせる）。via-viewer の試験ターンは**描画タブのセッション（白いロボットの履歴）に入る**ので、両方の履歴を truncate する。
- 批評が挙げた残り（未実装）: ① サーバーから viewer へ PCM 直送（描画タブ往復 40〜80 ms、要フロント再ビルド＋割り込み flush）、
  ② PROTO 側 Chrome の入出力バッファ（device 124〜140 ms、こちらから不可）、③ ターン初文の 16 字保留を飛ばす（−34〜55 ms、声の明るさと引き換え）、
  ④ 合流（rapid follow-up）が omni では後半の音声しか聞いていない（前の発話の ring 位置から切り出す）、⑤ トークン推定器が 3 倍過大で
  履歴窓が 1.9k トークンしか無い（`usage.prompt_tokens` で校正）、⑥ MCP サーバーの初回 spawn 478 ms とツール filler の発火時点。

**07:40 残り候補の実装（ユーザー「すべてやって」）**
- **①サーバー→ビューワ PCM 直送**（`VIEWER_PCM_DIRECT=1`、omni 既定）: `render_stream.ViewerMirror` が render client のソケットを包み
  （routes.py）、`audio-stream-chunk`（PCM; この client には Opus を出さない `pcm_only`）と相槌の WAV を 32→48 kHz にしてそのまま
  viewer へ。先読みの GatedSocket はこの包みの外側なので解放時にだけ届く。描画タブは従来どおり再生して口を動かすが、タップされた
  音声は `source_ws` で捨てる（二重再生なし）。割り込み（`user-speech-start` 送信、`interrupt-signal` 受信）で viewer に `{"t":"flush"}`、
  Play ワークレットはキューを空にする。実測: `[render] direct PCM: first chunk to the viewer @…` が `[TTFA] first_pcm@` の **1〜2 ms 後**
  （以前は描画タブの再生＋タップ経由で 40〜80 ms ＋ preroll）。
- **③ターン初文の保留を飛ばす**（`QWEN3TTS_MERGE_FIRST=1` で元に戻す）: 初文はゲートが開いた瞬間に出す。単独の「Hello.」は明るい声に戻る。
- **④合流の音声**: TURN_END で前の発話の ring 開始位置を `last_turn_start` に残し、rapid follow-up の合流では `_audio_b64_from` でそこから
  切り出す（Thinker が両方を聞く）。合流になるターンは fast release しない（従来経路で合流）。
- **ビューワの自動開始**: `render_view.html` は一度タップされると `localStorage.autostart=1` を立て、以後は再読み込み後に自分で開始する
  （AudioContext が suspended のままなら開始ボタンが残る）。**今回の再読み込みで PROTO は開始画面で止まっている**（07:33〜）。
  一度タップすれば以後は不要。`POST /render-stream/reload` の前にこの点を知っておくこと。
- **②PROTO 側の Chrome バッファ（この箱から不可、未検証）**: viewer の統計 `device 124〜140 ms` は PROTO の出力デバイス遅延。Linux/X11 の
  Chrome なら起動時に `PULSE_LATENCY_MSEC=30`（PulseAudio）または `PIPEWIRE_LATENCY=512/48000`（PipeWire）を環境に入れると入出力とも
  縮む見込み（出力 −60〜100、入力 −20〜40）。小さすぎると取り込みが途切れる（VAD のミス増）ので PROTO で聞いて決める。あわせて
  `--autoplay-policy=no-user-gesture-required` を付けると再読み込み後のタップが要らない。
- **⑤トークン推定の校正**（適用済み）: `openai_compatible_llm.AsyncLLM._cal`（初期 1.0、各返答の `usage.prompt_tokens / 推定` の EMA、α 0.3、
  [0.25, 1.0] に clamp）。`_trim_to_context(..., cal)` と `warm_prefix` が同じ係数で窓を決め、agent の `_mem_tokens` も掛ける。
  ログ `[context] token estimate calibrated: xN.NN (real … vs est …)`、trim の警告に `cal x…` が付く。英語ペルソナは chars/1.3 が 3 倍過大
  （実測 4.4 字/トークン）なので、数ターンで窓が実トークンで ~1.9k → ~5〜6k に広がる（その間 prefix cache が数回外れる）。
  係数は接続ごと（AsyncLLM インスタンス）に学び直す。
- **⑥MCP**（適用済み）: `ToolAdapter` のツール一覧はプロセスで 1 回だけ取得（接続ごと・キャラ切替ごとの二重 spawn をやめた）。
  `MCPClient` はサーバーごとに専用タスク（`_serve`）が stdio/session を入って出る（以前は別タスクで閉じて anyio の
  「cancel scope を別タスクで exit」が出ていた）。接続直後に `start_servers()` で裏で spawn（`MCP_PRESPAWN=0` で無効）。
  ついでに `handle_disconnect` が pop 後に `.get` していて `context.close()` が一度も走っていなかったバグを修正（MCP 子プロセスが切断で残っていた）。
  ツール filler は最初の tool-call delta（function.name）で発火（126〜285 ms 前倒し）。
- **直送の罠（07:52、ユーザー「ちゃんと喋らなくなった」）**: 直送をそのまま push すると、Qwen3-TTS の 480 ms チャンクが 100 ms ごとに届き、
  viewer のジッタ緩衝（`render_view.html`: target+40 ms 超の「溜まり」を切る、1 s 上限）が音を捨てて文が途切れた（統計 `play 480/40 ms`）。
  → `hub._pace_direct` が 20 ms フレームを実時間で送る（`VIEWER_PCM_LEAD_S`、既定 0.06 s だけ先行、1 s 無音で時計を仕切り直し、flush で破棄）。
  修正後 `play 100〜120/40 ms、starve 0`。
  先行は 0.04 s に（0.06 だと緩衝が target+40 を超えて 20 ms ほど切られる）。fast release 後の撤回は「空 / 幻聴の定型句 / 文字種違い /
  相槌語だけ」に限定（語数の規則は使わない: 一語の答え "Technology." が一音節出てから撤回されていた）。
- **最終実測（07:48、全部入り、各 6 ターン中央値）**: 素のクライアント **0.48 s**（0.45〜0.58、エンドポイント→最初の PCM 1 ms）、
  PROTO 経路 **0.53 s**（0.49〜0.58）。朝の開始時 0.62 / 推定 0.70 から。校正は 2 ターンで x0.70 → x0.63。
- **08:12 区切りを短く**（`VAD_PAUSE_MISSES=4` = 128 ms、`silero.py` の env 上書き。conf.yaml は 6）: 素のクライアント **0.43 s**（0.39〜0.53）、
  PROTO 経路 **0.47 s**（0.43〜0.53）。Thinker 単体の TTFT は音声 1〜3.5 s でも 70〜140 ms（実測）なので、区切りが早いほど最初の文が早い。
  代償: 発話内の間で先読みが捨てられる回数が増える（12 HIT に対し 8 drop、q2/q3 は文中に間がある）。GPU の無駄打ちで、速さには響かない。
  ビューワのジッタ床 40 → 20 ms（`render_view.html`、直送は 20 ms フレームを時計で送るので）: `play 80/20、starve 0`。
- **Thinker に渡したものの保存**: `OMNI_AUDIO_DUMP=logs/omni_audio`（omni 既定）に各ターンの wav と文脈末尾 8 件の json、新しい 60 ターン。
  「意味が分からない／英語じゃない」の切り分けはここから（wav を Whisper に通す／単体で Thinker に投げる／文脈付きで投げる）。

### Iris・Whisper 精度・文脈 24k・cascade を Qwen3-TTS で（2026-11-04 10:20）

- **新キャラ Iris**（`characters/iris_android.yaml`、`prompts/personas/iris_en.txt`、conf_uid `iris_android_001`、体は female_android と同じ）。
  白いアンドロイド＋Aura の声から作った AI Maker Space の案内役: 船のコンピュータのように落ち着いて正確、乾いた控えめなユーモア、来訪者の作るものに興味。
  3 案を書かせて規則（ジェスチャのキー、感情の絵文字、「not English / makes no sense」を言わない、聞き取れないときは具体的に 1 問だけ聞き返す、
  道具の実況をしない、時刻はシステム行から）で検査して合成。新しい conf_uid なので履歴は空から（ミク/Übel の「退屈」履歴を引き継がない）。
  - 描画タブは `localStorage.lastConfigFile`（`iris_android.yaml`）で復元する。戻すときは CDP で `female_android.yaml` を入れて再読み込み。
  - 待機モーションの回し方は conf_uid ごと（`motion-lib-catalog/active.json` の `by_conf` / `engaged_by_conf`）。female_android_001 のものを写した。
  - Iris の人格は 1 文目の前に `[g:…]` や 🙂 を置くので、消したあとの「Hello .」の空白を `transformers._close_tag_gap` で詰める（字幕・履歴）。
- **omni の Whisper を large-v3（full）＋beam 5 に**（`start_server.sh` の `ASR_MODEL_OVERRIDE` / `ASR_BEAM_OVERRIDE`）。返事は fast release で文字を待たない
  ので速さは変わらない。文字は区切りから 0.7〜1.1 s で届く（turbo は 0.3〜0.5）。0.7 s 未満の短い発話だけはこれを待つ。
- **omni の Thinker 文脈 16k → 24k**（`serve-qwen3-omni.sh`: KV 2.0 → 3.0 GB = 30,512 トークン、`max_model_len 24576`、util 0.95。
  `start_server.sh`: `LLM_CTX_BUDGET 20000` / `LLM_CTX_MAX 24576`）。GPU1 22.7 / 24 GB。
- **cascade を Qwen3-TTS の声で**（`./kehai.sh mode cascade`、`.cascade-tts` に記録、`CASCADE_TTS=gsv` で元の GSV 構成）:
  Qwen3.8-27B（TP=2）＋Qwen3-TTS を **GPU1** に（`serve-qwen3-tts.sh` の `QWEN3TTS_GPU`）。入らないので ActionPlan・GestureLSM・GSV は止め、体は omni と同じ
  （KAWAII 辞書）。`pcb-stack-up.sh` も `.cascade-tts` を読んで再起動後に同じ配置にする。`start_server.sh` の cascade 分岐で `OMNI_TTS=qwen3tts`
  （名前はそのまま、service_context が両モードで読む）、`VAD_VIEWER_DIRECT` / `VIEWER_PCM_DIRECT` も有効。GPU0 16.3 / GPU1 20.0 GB。
  - **実測（PROTO 経路、6 ターン中央値）: 話し終わり→ビューワの声 1.18 s**（1.04〜1.39）。omni は 0.45〜0.54 s。内訳: Whisper が返事の前に入る
    （区切り後 0.45〜0.5 s）、Qwen3.8-27B の最初のトークン約 0.3 s、最初の文 0.13 s、TTS 0.065 s。
  - cascade ではカメラの「話しかけられているか」判定（Omni の Thinker）が無い → 聞こえた声に全部答える。
  - Qwen3.8 は道具を使う前の思考を声に出した（「Let me check the current time in the America/Chicago timezone」）→ 人格に「道具の実況をしない」を追加して解消。
- 戻し方: `./kehai.sh mode omni`（Omni＋Iris のまま、速い方）。
- **10:55 に omni へ戻した**（ユーザー「omni に戻して、Whisper は精度の高いもの、Iris はこのまま」）。
  - `mode omni` は、cascade の配置で GPU1 に居る Qwen3-TTS を先に止めてから Thinker を起こし、TTS は GPU0 で起こし直す
    （`pcb-mode.sh`。止めないと Thinker が GPU1 に入らない）。
  - **Qwen3-TTS の初回合成は 4.5 s**（起動後の最初の要求で Triton の抽選カーネルを JIT コンパイルする。次の要求は 0.35 s）。
    切替直後の最初の返事が 4.2 s 黙った。`start_server.sh` の warmup で同じ形の要求を 1 回投げる（`[warmup] Qwen3-TTS is warm (5.0 s)`）。
  - 実測（PROTO 経路、Iris、3 問×2 周、最後の発話フレームから ms の中央値）: 区切り 168 / 最初のトークン 298 / 最初の文 426 /
    最初の PCM 492 / エンドポイント 441 / ビューワ **494**（初回のターンも 451）。ミクより最初の文が長い分（"The current time in Omaha is 10:45 AM."）、
    最初の文までが 60 ms ほど遅い。
- **11:20 計測の注意: プローブの数字は実際の部屋より 60〜100 ms 速く出る。** 試験の wav は声の直後が完全な無音（int16 で 0）で、
  VAD の dB の移動平均が 1 フレームで −inf になり、区切りもエンドポイントも即座に数え始める。末尾を雑音に置き換えた比較
  （3 問×2 周、ビューワまでの中央値）: 無音 494 ms / 25 dB 554 ms / 38 dB 593 ms。区切り 168 / 206 / 232、エンドポイント 441 / 480 / 510。
  試験用の写し（末尾を雑音にする `NOISE_DB`）はセッションの scratchpad のみ。実会話の目安は 0.55〜0.6 s。
- **11:20 コード点検で見つけた未修正の不具合**（4 区間の読み手＋点検役、主なものだけ。どれも未修正）:
  1. 割り込み後の合流が、large-v3 では文字が届くまでの 0.55〜0.9 s 効かない。fast release が `last_vad_turn` を (解放時刻, "") にし、
     文字が届くと (到着時刻, text) で上書きするので、その間の割り込みは「割り込み時刻 > last_ts」を満たさない（`websocket_handler.py` 1338, 1691, 1877）。
     08:24 の修正は turbo（文字が速い）で試験していた。
  2. Thinker は貪欲デコード。vLLM-Omni はリクエストが明示した項目だけを上書きし、stage の既定 `top_k 1`・seed 42 が残る
     （`qwen3_omni_moe_thinking.yaml`、`serving_chat.py` `_apply_request_overrides`）。conf の temperature 0.7 は効いていない。
  3. 毎リクエストの全メッセージ（base64 の発話音声を含む）が DEBUG のファイルログに書かれ、30 日残る（`openai_compatible_llm.py` 324、`run_server.py` のログ設定）。
  4. fast release した返答を雑音として撤回しても、ビューワへ直送済みの音声は止まらない（`viewer_flush` を呼ばない）。
  5. 繰り返しの返答を外す規則が質問だけ外して時刻の system 行を残し、system 行が連続して溜まる（`basic_memory_agent.py` 320, 673）。
  6. `SPEC_FAST_RELEASE_MIN_S=0.7` は PROTO では効かない（音声長に開始前バッファ・終わりの無音・0.15 s の先行が入り、最短でも約 0.8 s）。
  7. `_VIEWER_LEAD_S=0.15` は VAD 直結では不要になり、ASR と Thinker の音声が 150 ms 早く始まる。再切り出しはイベントループ上。
  8. 短文つなぎで後ろの文の身振りが消える（`tts_manager.py` 336、`Actions` に `__bool__` が無く常に真）。
  9. Whisper の `initial_prompt` に Iris が無い（Miku はある）。
- 同日 11:00: Whisper サイドカーが turbo（09:49 以降未使用）も抱えていたので `/unload` して large-v3 だけ読み直した（2.7 → 1.8 GB）。
  サイドカーは一度使ったモデルを手放さない。`/health` の `loaded.stt` は常に false（キー名の照合の誤り）。

### 午後の「バグり散らかしてる」の根本原因と修正（2026-11-04 17:40）

症状: 14:53 のまともな会話のあと、15 時台から崩れ、16 時台の返事 263 件中 251 件が「I am not sure what you mean. Could you clarify
your question?」。部屋の人どうしの会話（発表、電話、雑談）にも全部答え、割り込まれて断片をつなぎ、また「分からない」。

原因は 3 つの重なり（切り分け: Thinker に渡した実際の音声を、文脈なしで投げると 12/12 普通に答え、実際の文脈付きだと 12/12「分からない」）:
1. **文脈の壊れ方**: 17:11 の要求は 693 件で、利用者の発言 36・時刻の行 346・割り込みの印 160・「分からない」系の返事 135。
   繰り返しの規則が質問だけ消して時刻の行を残し、直後の割り込み処理が消えた返事を「…」付きで書き戻していた。さらに `_trim_to_context` が
   古い側（まともな 14:53 の会話）から捨てて、新しいゴミを残した。
2. **Thinker が貪欲デコード**（top_k 1、seed 42。vLLM-Omni の stage 既定）。温度 0.7 は効かず、同じ入力に毎回同じ文。
3. **話しかけの判定が効いていない**: カメラの判定は同じ場面（奥の机で背を向けて話す 2 人）に 0 点も 9 点も付け、YuNet は近くに顔が無いと
   「答える」に倒れ、ビューワの AGC で遠くの声も 68 dB の関門を越える。

修正:
- **記憶をターン単位で整える**（`basic_memory_agent.tidy_memory`、`is_non_answer`）: 「分からない」系の返事と繰り返しの返事は、質問・時刻の行・
  割り込みの印ごと外す。質問の無い時刻の行・返事の無い印・質問を失った返事も外し、続いた質問はまとめる。会話中（`_add_message`）、
  割り込み（`handle_interrupt`、外したターンには何も書かない）、履歴の読み込み、各要求の前、先読みの記録時に当てる。履歴ファイルはそのまま。
  今日の壊れた文脈 691 件が 23 件の正しい形に、履歴からの読み込みは 53 件（「分からない」0 件）になる。
- **先読みの巻き戻しを「その先読みが足したものだけ外す」に**（`speculative.checkpoint` の `mem_ref`、id で追う）。件数で切り戻す方式は、
  ほかの書き込みや整理で件数が変わると別のターンを消していた。
- **抽選を明示**（`conf.yaml` の `openai_omni_llm`: temperature 0.7・top_p 0.8・top_k 20。`OpenAIOmniLLMConfig.top_k` を足し、extra_body で送る）。
  起動ログ `[omni] ... sampling temperature=0.7 top_p=0.8 top_k=20`。
- **耳の判定**（`omni_addressee.AudioJudge`、`OMNI_ADDRESSEE_AUDIO`、閾値 `OMNI_ADDRESSEE_AUDIO_MIN`=5）: 発話の音声だけを Thinker に聞かせて
  「ロボット宛てか」を 0〜9 で答えさせる（max_tokens 1、45〜220 ms）。区切りで先読みと並行に始め、話し終わりで 5 点未満なら声を出さずに捨てる
  （ログ `[addressee] not to her (ears 0/9) — dropped at the endpoint`）。彼女が話している最中だけは先に判定する（先読みが返事を打ち切るため）。
  続きの発話（合流）は、前の発話からつないだ音声全体で判定する。判定が取れないときだけ従来のカメラに戻る。区切りごとのカメラの look はやめた（onset の look は割り込み用に残す）。
  - 評価: 今日の部屋の 60 件（全部雑談か雑音）は全部 0 点、PROTO のマイクに寄せた質問 30 件（3.9 kHz で切る・残響・部屋の雑音・同じ音量）は 28 件 9 点。
    合成音声のままだと帯域の違いで分けているだけかもしれないので、必ずマイクに寄せて試すこと（部屋の録音は 4.2 kHz より上がほぼ空）。
  - 「彼女の直前の返事」を文字で添えると、雑談を 60 件中 28〜51 件通すので添えない。「自分宛てでなければ [silent]」を返事の生成に混ぜる案は 0/60 で効かない
    （人格の「聞き取れなければ 1 問聞き返す」が勝つ）。
  - **短い返事**（"Yes." "Why?" "Thank you."）は単体だと 0 点。彼女の返事の終わりから 6 秒以内（`ADDRESSEE_FOLLOWUP_S`）で、YuNet で近くの人が
    こちらを向いていれば通す（`_followup`、ログ `taken as hers`）。「直前の返事が ? で終わった」だけで通す版は、奥の雑談を答えとして拾い連鎖したのでやめた。
- 撤回したときにビューワの音も止める（`viewer_flush`）。fast release のあと文字が届いても `last_vad_turn` の時刻は解放時のまま（割り込み後の合流が効く）。
- デバッグログは要求の全文（base64 の音声込み、1 行 160 KB、4 分で 10 MB ローテート、来訪者の声が 30 日残る）をやめ、件数・役割の並び・直近 12 件（音声は伏せる）に。1 行 1.2 KB。
- Whisper の語彙ヒントに Iris と AI Maker Space を足した。

実測（PROTO 経路、部屋程度の雑音 25 dB を足した 3 問×2 周、ビューワまでの中央値）: 570 ms（判定を入れる前 554 ms）。

残り・知っておくこと:
- **割り込みはまだカメラの onset の look で決まる**。彼女が答えている最中に奥で雑談が始まると、カメラが 9 点を付けた場合は止まる（止まったあと、その雑談には答えない）。
  耳で決めるには発話の頭 0.5 s ほどを待つ必要があり、割り込みが遅くなるので未対応。
- 短い返事は、話し手がカメラに映らない位置（近すぎ・画面の下）だと拾えない。「Yes, tell me more」のように中身を足せば耳で通る。
- 耳の誤りの例: 部屋の「All right.」が 9 点で通った（17:39）。
- 履歴ファイルは今日の雑談込みのまま（1007 件）。読み込み時に整える。まっさらにしたいなら新しい会話を作る。
- ディスクは残り 8.6 GB（100%）。logs は 738 MB。

### 「前に人がいなくても答える・隣の TV に反応する・周りの音を拾いすぎる」（2026-11-06 13:30）

（この節の「言葉の判定」は 2026-11-08 に削除した。一般化しなかったため。次の節「宛先は『誰が話しているか』で決める」を参照）

- **原因**: 11-05 16:00〜11-06 12:50 に答えた 45 ターンのうち 41 が「耳は 0 点だが、近くの人がこちらを向いている」の例外（11-04 18:55 に足した `_followup`）で
  通っていた。PROTO の隣に TV があるので、TV を見ている人・横の机の人（カメラ下端、2.0 m）は PROTO の方を向く → TV の声（「Thank you for tuning in…」
  「We'll be right back」「see you again in the next episode」）に答え、「Are you referring to …?」と聞き返し続けた。TV の声の大きさは 68〜77 dB で人の声（70〜80 dB）と
  変わらない（ビューワの AGC が遠くの音も持ち上げる）ので、音量の関門では分けられない。さらに、彼女が話している最中の TV の声で毎回止まっていた
  （発話の始まりでカメラの look が「宛て」と見れば止める作り。12:41〜12:44 の 3 分で 14 回）。
- **答える条件**（`websocket_handler._addressee_policy`）: 耳（`omni_addressee.AUDIO`）と「前に人」（YuNet、近く 4 m 以内でこちらを向いた顔）が
  一致すればそのとおり、食い違ったら**言葉**（`omni_addressee.WORDS`）が決める。カメラが無いときは従来どおり耳だけ。言葉の判定が取れないときは従来の規則（答える側）。

  | 耳 | 前に人 | 以前 | 今 |
  |---|---|---|---|
  | 5 点以上 | いる | 答える | 答える |
  | 5 点以上 | いない | 答える | 言葉が「宛て」なら答える |
  | 5 点未満 | いる | **必ず答える** | 言葉が「宛て」なら答える |
  | 5 点未満 | いない | 答えない | 答えない |
- **言葉の判定**: Thinker の書き取り（下）を、手本 21 個付きの A/B（A = 来訪者が Iris に、B = TV・動画・人どうし・意味のない断片・直前に彼女が話していない相槌）で判定。
  彼女の返事の終わりから 6 s 以内（`ADDRESSEE_FOLLOWUP_S`）または話している最中は、彼女の直前の文を添える（「Yes.」「Thank you.」は返事として通る）。45 ms。
  評価（本物の来訪者の発話と試験の質問 31 件、同じ日の TV・部屋の発話 60 件、手本とは別の文）: **31 件中 30 件を通し、60 件中 9 件を通す**（9 件は TV の登場人物の質問と
  「No.」「What is it?」）。0〜9 点で答えさせる版は同じ規則でも 0 か 9 に振れて 15〜22/31 しか通らなかった。試験の道具はセッションの scratchpad のみ（`intent/eval.py`）。
  区切りで耳と並べて始める（`spec.words`）ので、話し終わりの判定は待たない。`OMNI_ADDRESSEE_WORDS=0` で無効。
- **彼女が話している最中の割り込みは、言葉を聞いてから**（`BARGE_CONFIRM=1`、`_speech_onset_omni` が保留 → `_barge_probe` が発話の始まりから
  `BARGE_PROBE_S`=0.7 s の音声で耳＋言葉を判定、区切り・話し終わりでも判定）。「話している最中」は、ビューワへの直送の声が流れている（`render_hub.speaking`）か、
  解放済みの返事が作られている間（ゲートの閉じた先読みは数えない）。実測: 質問で割り込むと 0.54〜0.58 s で止まる（以前はカメラの look で約 0.25 s）、TV の文では
  止まらず話し続ける（`[addressee] no barge-in: she kept talking`）。`BARGE_CONFIRM=0` で従来どおり始まりで止める。
- **書き取り（`[omni] heard`）の誤りを直した**: 「Transcribe the speech exactly」と頼むと、Thinker が依頼に**答えて**いた（試験の 8 本中 3 本:
  「Iris, tell me a short story…」→「Once upon a time…」、「What is machine learning?」→ その説明。本番でも「I'm sorry, I can't understand…」）。この文は 11-04 から
  返事に「(What I said, as heard: …)」として添えていたので、返事も惑わせていた可能性がある。→ 書き出しを `The speaker said: "` で始めさせて（assistant の prefill、
  `continue_final_message`、`"` で止める）**8 本中 8 本正しく、440 ms → 96〜220 ms**。書き取りは `omni_addressee.HEARD` が 1 発話 1 回だけ行い、返事
  （`openai_omni_llm.transcribe`）と言葉の判定で共有する。
- 試験（`omni_probe.py --via-viewer --cam`、TV らしい文は Qwen3-TTS で合成）: 前に人＋TV の文 3 本 → 3 本とも捨てる、前に人＋質問 → 答える、無人＋TV → 捨てる、
  無人＋質問 → 答える（耳 9 点＋言葉「宛て」）、話している最中に TV → 話し続ける、話している最中に質問 → 0.58 s で止まって答える。
  罠: プローブを 2 本同時に動かすと mic-inject の音が交互に混ざって断片になる。割り込みの試験は 1 本のプローブに 2 つの wav を渡す（`--window 2.5`）。
- 残り: ① 来訪者の発話の**途中**に TV の声が入ると（同じ発話の中）、書き取りにも返事にも混ざる。② TV の登場人物の質問（「What's up with you guys?」）は、
  前に人がいると通ることがある（評価で 60 件中 9 件）。③ 誰も映っていなくても、耳と言葉の両方が「宛て」なら答える（カメラの外・真下にいる人のため）。
  ④ AGC（自動音量調整）を切ると、近くの声と遠くの TV の差が開く見込み。ただし関門 68 dB は AGC ありで決めた値なので、PROTO のマイクパネル（自動音量調整・
  メーター・関門）で、その場で話しながら決める必要がある。
- **11-07「人間の声を最後まで聞き取れてない」の修正**（11-06 13:31〜13:34 のユーザーの試験、PROTO の正面 1.5〜2.5 m、TV あり）:
  - 原因 1: 言葉の判定がユーザー本人の発話を捨てた（「Oh, I mean, you know, go back to the market…」「Tell me about it.」「How do you even show? …」）。
    PROTO のマイク（4 kHz）と訛りで書き取りが崩れる（「Can you tell me about one place?」が Thinker では「Now, see, they tell me about one place.」、4 回続けて B）。
    耳はユーザーの声にほぼ毎回 0 点なので、毎ターン言葉の判定に頼っていた。
  - 原因 2: 文の途中の間（0.5 s 超）で話し終わりになり、「So I」「Tell me about」が断片のまま判定され、手本の「意味のない断片は B」で捨てられた。
  - 原因 3: 途中で止まった文を待って後半とつなぐ処理（`pending_vad_transcript`）は前からあったが、omni では**つないだ文字は Thinker に渡らず後半の音声だけ**を
    聞かせていた（omni は文字を送らない）。さらに後半の始まりを VAD が検知するまで約 0.9 s かかり（68 dB の平滑化した関門）、0.8 s の待ちが先に切れていた。
  - 直したこと: (1) 言葉の判定は Thinker と Whisper の**両方の書き取り**で行い、どちらかが A なら A（`_addressee_policy` の `alt_text`）。
    ユーザーの発話 11/15 → 12/15、TV・部屋 0/15 のまま。(2) **会話中**（彼女の返事の終わりから 6 s 以内、近くの人がこちらを向いている）は寛容な手本
    （`omni_addressee._WORDS_SYSTEM_ENGAGED`、訛り・崩れた書き取り・断片も A）: ユーザー 14/15。ただし 11-04〜06 の TV・部屋 60 件中 19 件も通すので、この窓だけ。
    (3) fast release の時点で言葉が B でも**すぐ捨てない**（Whisper の文字と未完の待ちを通す通常の経路へ）。すぐ捨てるのは「耳 5 点未満で前に人もいない」ときだけ。
    (4) 書き取りが文の途中で終わっていたら（`_transcript_looks_incomplete` に「you know」「I」「about」「the」などで終わる英語を追加、末尾のピリオドも見ない。
    Thinker と Whisper のどちらかが未完なら）fast release せず、**1.5 s 待つ**（`VAD_UNFINISHED_HOLD_S`、その他の待ちは従来の 0.8 / 0.3 s）。
    待ちに入る前に話が再開していたら、すぐ次の発話につなぐ。(5) つないだときは**音声も前半の頭から切り出す**（`prev_turn_start`）。文字の間に空白を入れる。
  - 試験: 「Can you tell me about」＋0.9 / 1.2 s の間＋「the South Market, please?」→ 前半で答えず、両方を聞いて 1 回で答える（話し終わり→最初の声 0.54〜0.59 s）。
    TV の文（前に人あり・なし）は捨てる、話している最中の TV では止まらない、質問の割り込みは 0.64 s で止まる、は変わらず。
  - 試したがやめたもの: 話者の声の特徴で「さっきの人か」を見分ける（sherpa-onnx の WeSpeaker ResNet34 / 3D-Speaker CAM++）。この部屋の音では TV・部屋の声も
    ユーザーに近い値になり（WeSpeaker 0.66〜0.89）、2 s 未満の短い発話はユーザー本人でも 0.12〜0.97 とばらつく（CAM++）。
  - 残り: 「Excuse me.」のような完結した短い呼びかけのあとの間では、先にそれに答え始めてから続きで差し替える（従来どおり）。返事の多くが「Are you asking about …?」の
    聞き返しになるのは人格の「聞き取れなければ 1 問聞き返す」の効き過ぎで、未対処。

### 宛先は「何を言ったか」でなく「誰が話しているか」で決める（2026-11-08、ユーザー「PROTO に話しかける人はいろんな人がいる。この会話に最適化しても根本的解決をしない限り意味がない」）

- **上の 11-06/07 の言葉の判定は一般化しなかった**（削除済み）。手本はこの部屋の TV とこのユーザーの文から作り、評価も同じ日のデータだった。部屋と無関係の公開データで測ると「Iris 宛て」と判定した割合は次のとおり（scratchpad の評価、各 150 件）。
  - 音声アシスタントへの一般的な依頼（MASSIVE en-US test、18 分野）: **41 %**
  - 人どうしの会話（DailyDialog）: 31 %
  - YouTube・ポッドキャスト: 0 %
  - 「会話中は寛容」版: 77 % / 31 % / 61 %
  - 「ライブか放送か」で分ける版も分かれなかった（依頼を 95 % 通すと YouTube も 85 % 通る）。
- **誰が話しているか**: `src/kehai/active_speaker.py` を使う。モデルは LR-ASD（MIT、0.84 M パラメータ、`models/asd/` に重み・ライセンス・視覚側の ONNX）。前の顔の唇が声に合わせて動いたかを判定する。
  - `models/` は git の管理外。重みは github.com/airmeet/LR-ASD の `weight/finetuning_TalkSet.model`。`lrasd_visual.onnx` はその視覚側を `torch.onnx.export`（opset 17、時間方向は可変長）で書き出したもの。無ければ torch で動く（約 2 倍遅い）。
  - TalkNet の公開デモ動画での値: 話している人 +1.1〜+3.1、聞いている人 −0.5〜−1.3。
  - 顔が小さいと、聞いている人の値も上がる（36〜46 px で 2/10、25〜30 px で 7/9 が正）。そこで判定は顔 **56 px 以上**（`ASD_MIN_FACE_PX`）に限る。+0.6 以上なら「話している」（`ASD_SPEAK`）、−0.3 以下なら「黙っている」（`ASD_SILENT`）、間は判定しない。
- **ビューワが 4K で撮り、近くの顔だけ 112×112 に切り出して送る**（`render_view.html` の `sendFaces`）。
  - 送信形式は kind 6: `<u8 6><u16 track id><u16 顔 px>` + JPEG。顔の枠はサーバーが `{"t":"asd-faces"}` で送る。
  - 640×360 の画では 2 m 先の顔が約 20 px しかない。カメラは約 100° なので、顔の大きさは 640 幅で約 40/d px、1920 幅で約 120/d、3840 幅で約 240/d px（d は距離 m）。
  - 4K で 20 fps を切ったら 1080p に落とし、`localStorage.camW` に記憶する（`?camw=` で強制）。
  - ログ: `[render] viewer camera runs at WxH N fps`。統計行に `N lip crops/s`。
  - **PROTO では未検証**（ビューワは 11-06 13:33 から未接続）。
- **答える条件**（`websocket_handler._addressee_policy`）。言葉は使わない。

  | 前の顔 | 結果 |
  |---|---|
  | 唇が声に合わせて動き、その顔がこちらを向いている | 答える |
  | 唇が声に合わせて動いたが、顔がこちらを向いていない | 答えない |
  | 顔は見えたが、誰も話していない（TV・電話・画面外の人） | 答えない |
  | 判定できない（顔が小さい・顔が無い・カメラが無い） | 前に人がこちらを向いていれば答える。それ以外は耳で決める |

- **計算時間**: 視覚側は ONNX（起動時に torch と照合し、合わなければ torch で動かす）。
  - 発話中に 0.5 s ごとに先に進めておき（`active_speaker.prepare`）、区切りで判定する。
  - 1 人なら約 0.13 s、3 人なら 0.3〜0.45 s。区切りから話し終わりまでの 0.28 s を超えた分だけ、返事が遅れる。
  - 文脈なしに区切って計算すると、聞いている人の値が +0.16 から +0.72 に上がった。だから前後の画像がそろった部分だけを確定させる。
- **文が途中か**: 英単語のリスト（`_EN_DANGLING_RE`）を削除した。代わりに Thinker が文法で判定する（`omni_addressee.COMPLETE`、手本は一般的な文）。
  - 公開データで測った（DailyDialog と MASSIVE の文を、ランダムな語の位置で切ったもの 200 件と完結した文 200 件）。
  - 途中で切れた文の検出: リストは 66/200、言語モデルは 148/200。Whisper 風に末尾へ「.」「?」を付けても 142/200・139/200。
  - 完結した文を誤って待たせたのは 6/200。
  - 前半とつないだ後は、新しい半分だけを判定する。
- **続きの発話**: 彼女が受けたターンの終わりから 1.5 s 以内（`TURN_CONTINUE_S`）に話し始めたら、中身を見ずに同じ人の続きとみなす。彼女が話している最中なら即座に止まる。落とすのは「前の顔が話していない」と見えたときだけ。
- **聞き返しの溜まり**: 履歴にある彼女の返事 157 件のうち 60 件が「Are you referring to / asking about …?」で、それがお手本になっていた。最新の 1 件以外は文脈から外す（`basic_memory_agent.is_echo_question`、tidy_memory）。文脈は 316 → 145 件になった。
- **試験**: `scripts/asd_probe.py` で動画の絵と音を mic-inject に流す。`--crop` で聞き手だけを映せば TV の状態になる。`--before q.wav --gap S` で割り込みを試せる。

  | 試験（TalkNet デモ動画） | 結果 |
  |---|---|
  | 話し手が映っている（内容は YouTube 冒頭のあいさつ、耳は 0/9） | 答えた |
  | 聞き手だけ映っていて、声は画面外 | 答えなかった（−1.3） |
  | 話し手だけ映っている | 答えた（+1.6、判定 0.12 s） |
  | 区切りのある文（間 0.9 s / 1.2 s） | 待ってつなぎ、1 回だけ答えた |

- **残り・注意**:
  1. **Qwen3-TTS（:8809）は 11-07 23:30 に外部から終了された**（正常終了の記録。誰が止めたかは不明）。00:10 から GPU0 で trellis-service が動いている。声が出ないので、GPU0 の使用者を確かめてから `./kehai.sh up` で戻す。
  2. 彼女が話している最中の割り込みは、TTS が無いため新しい判定で試験できていない。
  3. **PROTO のマイクは今も 4 kHz で切れている**。11-06 の実録音で 4.2 kHz 以上と以下の比が 2e-7〜2e-3（合成音声は 5e-3〜1e-2）。入力には LADSPA の SC4（コンプレッサ）と Simple amplifier の連鎖が見える（caps のログ）。誰の声でも子音が消えるので、PROTO で `pactl list short sources` と `pactl list short modules` を確認する必要がある。
  4. 静止画を使う `omni_probe.py --cam` は 2.5 fps なので ASD の対象外になり、従来の「前に人」の規則で決まる。
  5. Smart Turn（声の抑揚で文の終わりを判定）は使っていない。PROTO の実録音の「Tell me about」を 0.93（完結）と判定したため。

### 唇の判定は PROTO の実機で使えなかった（2026-11-09 00:20〜01:00、ユーザーが PROTO で試験）

- **PROTO 本体とマイクは変えられない**（ユーザー）。ビューワのページとサーバーだけで直す。
- 実機の試験（ユーザーが正面 2.6〜3.1 m からこちらを向いて話す、顔の切り出し 160〜220 px、4K カメラ 29 fps）:
  - 唇の判定（`active_speaker`）は、話しているユーザーを −0.15〜−2.5 の「黙っている」と判定した。最初の版はこれで返事を捨て、「全然しゃべらない」状態になった。
  - 切り出しの画像は正常だった（眉から顎まで写り、口も動いていた。確認後に画像は削除）。
  - ビューワがカメラの撮影時刻（rVFC の `captureTime`）を付けて送るようにし、さらに音声を −0.3〜+1.0 s ずらしても、点数は負のまま（`ASD_LAG_SWEEP=1` のログ）。時刻のずれが原因ではない。
  - 疑われる原因: PROTO の音（4 kHz で切れていて、Chrome のノイズ除去・AGC もかかっている）が学習データ（AVA・VoxCeleb）と違う。
  - 4K の取り込みと毎コマの切り出しで PROTO が重くなった（ユーザー「しゃべり始めると全体的に重い」）。
- **今の状態**: `ASD=0`（start_server.sh の既定）、ビューワのカメラは 640×360 に戻した（`?camw=3840` で 4K の切り出しを試せる）。唇の判定を有効にしても既定は記録だけで、返事を捨てたり待ったりしない（`ASD_DROP=1` のときだけ捨てる）。宛先は「前に人がこちらを向いているか」と耳の判定で決まる。TV の前でこちらを向いている人がいると、TV の声に答えることがある。この問題は、マイクを変えられない限りソフトだけでは解けていない。
- **会場向けの宛先の決め方**（ユーザーの選択「近くで止まった人だけ」）: カメラがある限り、宛先は `face_detect.Attention.engaged` で決める（耳も言葉も使わない、`ADDRESSEE_ENGAGED=0` で従来）。
  - 条件: 発話の 1 s 前（`ADDRESSEE_DWELL_S`）から話し終わりまでのコマの 60 %（`ADDRESSEE_DWELL_FRAC`）に、`ADDRESSEE_NEAR_M`（3.3 m）以内の顔が写っていて、その顔がこちらを向いている。
  - 近づきすぎてカメラの下に外れた人は、直前 3 s に近くにいて、その後どの顔も写っていなければ同じ人とみなす。
  - **距離は較正していない**。PROTO で話しかけたユーザーは 2.6〜3.1 m と推定され、11-06 には TV を見る横の人が 2.0 m と推定された。その場で `[addressee]` のログの `nearest X m` を見て、`ADDRESSEE_NEAR_M` を決める必要がある。
- **騒音に合わせた話し始めの関門**（`vad/silero.py` の `_adaptive_gate`）: 直近約 8 s の音量の下位 20 % を騒音の床とし、関門を「床 + `VAD_ADAPT_MARGIN_DB`（10）」にする。設定の関門（68）より下げず、`VAD_ADAPT_MAX_DB`（74）より上げない。上がったときはログ `[vad] noisy room: onset gate N dB (floor M dB)`（30 s に 1 回）。`VAD_ADAPT_MARGIN_DB=0` で無効。
- **改行コードの罠**: `silero.py`、`websocket_handler.py`、`basic_memory_agent.py` は HEAD では CRLF。Python の open() で読み書きすると LF になって、差分が全行に広がる。silero.py は CRLF に戻した（あとの 2 つは 11-08 より前から LF になっていた）。
- **読み込み直しの不具合**: ページを読み込み直したとき、古いページの接続が後から閉じると映像の取り込みが止まり、0 fps のままになっていた。ビューワが接続中なのに 10 s 何も作られていなければ、取り込みを起こし直すようにした（統計のループ。ログは `viewer connected but no capture running`）。
  サーバーを再起動したとき、描画タブより先にビューワがつながると、描画タブに届かず取り込みのモードが cdp のまま止まり、0 fps になった（02:42）。同じループで描画タブへの接続もやり直すようにした（ログ `viewer connected before the render tab was reachable — will retry`）。
- **映像だけの唇の判定（`lip_motion.py`、`LIP=1`、記録だけ）**: LR-ASD の映像専用の判定部分（lossV）を使う。ビューワのカメラは 1920×1080。
  - 結果: デモ動画では話し手が +0.5〜+1.7、聞き手が −0.2〜−0.9 だった。PROTO で正面から話したユーザーは 8 回で −1.22〜+0.07 になり、そのままでは使えない。
  - 切り出し枠の揺れ・遅れ・顔の小ささは原因ではない。デモ動画で、枠を 8 % 揺らしても、100 ms 遅らせても、顔を 31 px にしても、話し手は正のままだった。
  - 値を崩すのは画質。白っぽい画やコントラストの低い画では、話し手も −0.4〜+0.1 になった。コマごとのノイズ（sd 8）では全体が負側にずれる（ただし話し手と聞き手の順は残る）。
  - 明るさとコントラストを学習時の分布（平均 106、sd 43）にそろえると、白っぽい画もコントラストの低い画も元の値に戻る。
  - `[lip]` の行には、そろえた後の値（`normalised`）と、切り出しの `crop mean / std / noise` も出す。デモ動画では mean 117 / std 33 / noise 5.3。
- **距離の較正（02:55）**: ユーザーが PROTO の正面 **1.5 m** で話したとき、推定は 2.1〜4.0 m だった。サーバーの画角の仮定を 60° → **100°** に変えた（`FACE_HFOV_DEG`、すべての距離が 0.48 倍になる。90° の段階では 1.5 m が中央値 1.7〜2.0 m と出た）。閾値も実際のメートルに合わせた: `ADDRESSEE_NEAR_M` 3.3 → **2.0**、`ADDRESSEE_MAX_M` 4.0 → **2.5**。フロントの `headCamFovDeg`（60）は別の値で、今回は触っていない。
  - 「近い」は、その発話の間の一番近い顔の距離の**中央値**で判定する（1 コマごとの値は ±30 % 揺れ、1 コマずつ判定していた 02:35 は 7 回中 4 回を捨てた）。「立ち止まった」は、顔が写っているコマの割合（`DWELL_FRAC` 0.6）で判定する。ログは `nearest median X m, p10-p90`。
- **長い待ちは、両方の書き取りが「途中」のときだけ（02:56）**: Thinker の書き取りが言い淀み（"Ah, which ah, which job can you ah, can you explain more?"）を「途中」と判定し、Whisper（"Which of them can you explain more?"）は「完結」と判定していた。それでも 1.5 s 待ったので、返事が 1.8 s 遅れた。両方が「途中」なら 1.5 s 待つ。食い違ったら通常の 0.8 s だけ待つ。
- **聞き返しばかりになる（03:00）**: 「Are you referring to X? Could you please clarify …?」が 3 回続いた。2 つの修正を入れた。
  - 文脈から外す判定（`is_echo_question`）が、2 文目の「Could you please clarify」を拾えていなかった。「clarify / rephrase / specify」「seems incomplete」「didn't catch」などを含む聞き返しだけの返事も対象にした。
  - Iris の人格の指示を変えた。来訪者の多くは英語が母語ではない、崩れた英語や半分の文でも、一番ありそうな意味で中身を答える、「Are you asking / referring」で始めない、clarify を頼まない。ほとんど何も聞き取れなかったときだけ 1 回聞き返してよい（続けては不可）。手本の聞き返しの例も、崩れた質問に答える例に替えた。
  - 評価（Thinker に文字で投げる、各 2 回）: 聞き返しだけの返事は次のとおり。
    - MASSIVE の質問を機械的に崩したもの（語を 3 割落として "uh" を入れる）: 80 回中 17〜18 → 4〜6
    - 崩していない質問: 80 回中 6 → 4
    - 今日のユーザーの発話: 60 回中 6〜9 → 4
    - 数字の作り話（質問に無い数字）は変わらず、80 回中 4〜6
- **聞き返しの作り直し（03:20、`ASKBACK_GUARD=1`）**: 人格の指示を変えても、03:10 の会話で「Are you asking for a recommendation in Barro Anime? If so, could you please clarify your request?」「I did not understand the question. Could you please rephrase?」が出た。
  - 返事の最初の 5 語（または最初の文）を声に出す前に見る。「Are you asking / referring」「I did not understand」「I am unsure」「Could you clarify」などで始まっていたら、生成を止める（`askback_opening`）。
  - その場合は、来訪者の発話の末尾に「(Answer what I most likely meant, directly. Do not ask me what I meant.)」を付けて、1 回だけ作り直す。
  - 遅れは作り直したターンだけで、+0.3〜0.5 s。普段のターンは最初の 5 語を待つだけなので、ほぼ遅れない。
  - 評価（Thinker に文字で投げる、聞き返した返事を 3 回ずつ作り直す）: 作り直してもまた聞き返した割合。
    - 崩した MASSIVE の質問: 69 回中 29 回
    - 今日のユーザーの発話: 24 回中 3 回
    - 同じ指示を末尾の system 行で渡すと、それぞれ 69 回中 22 回、24 回中 17 回だった。実際の発話にはほとんど効かないので、来訪者の発話の側に付ける方式にした。
  - 03:24 に一度効かなかった。最初の 5 語（"I am not sure what"）で判定を打ち切っていて、"you" が来る前に「聞き返しではない」と決めていたため。直したこと: 聞き返しの型になりうる語（I / Are / Could / The / Iris …）で始まる返事は、最初の文が終わるか 8 語になるまで見る。それ以外の語で始まる返事は、最初の 1 語で声に出す。
- **作り直しても聞き返したとき（03:35）**: 03:31〜03:33 に作り直しは 10 回働いたが、作り直した返事の約半分がまた「Are you asking about X? If so, <答え>」の形だった。この形のときは、最初の確認の文と「If so,」を落として、答えだけを話す（`strip_askback`）。確認の文だけの返事は、そのまま話す（聞き返しは 1 回まで許す）。
- **できないことを請け負わない（03:40）**: Iris の人格に「できることは話す・前に人がいるかを見る・時刻・ウェブ検索だけ。アラーム・予定・メール・再生・アプリへの接続・機器の操作はできない。頼まれたらそう言い、代わりにできることを示す。やった・やるとは言わない」を足した。MASSIVE の操作の依頼 40 件 × 2 回で、「Setting a reminder」「I will remind you」のような請け負いは 4/80 → 0/80。
- **同じ出だしの繰り返しを作り直す（03:40）**: 「I am Iris, the android guide for this maker space.」が 4 回続いた（03:31）。返事の出だし 6 語が直近 8 回の返事のどれかの出だしと同じなら、声に出す前に止めて、「(Answer this, not what I asked before. Do not repeat what you already told me.)」を付けて作り直す（`repeat_opening`、聞き返しの作り直しと同じ 1 回の枠）。
- **途中で止まった発話の後半は 0.8 s だけ待つ（03:28）**: 「Nature me about that team.」のあと、後半の「Prospective.」が 1 語だけで「途中」と判定され、さらに 1.5 s 待って返事が 2.1 s 遅れた。いったん待ってつないだ後半には、長い待ち（1.5 s）を使わない。
  - 作り直した返事の中には、中身の薄い推測もある（"The process involving those three things is working well."）。聞き返しの連鎖と引き換え。
- **書き取りの暴走（03:02）**: 「Mm-hmm」のような小さな声で、Thinker が書き取りの書き出しを繰り返した（"The speaker said: 'The speaker said: …"）。それが記憶と「文が途中か」の判定に入っていた。書き出しの繰り返しや、同じ 3 語が 3 回以上続く書き取りは捨てて、Whisper の文字を使う（`omni_addressee._degenerate`）。
- 唇の判定（02:54、正面 1.5 m、5 回）: 話しているユーザーは −0.63〜−0.06 で、明るさとコントラストをそろえても変わらなかった。切り出しの明るさは mean 106〜120・std 32〜48 で正常。noise（目の領域のコマ間の変化）は 9〜17 で、デモ動画の 5.3 の 2〜3 倍。デモ動画に sd 8 のノイズを足すと、同じ noise 11〜13 になり、値が全体に負側へずれる。原因はカメラのノイズと見られる。次に試すなら、同じ人の黙っているときの値を基準にした比較（ノイズでも話し手と聞き手の順は残るため）。
- **書き取りは外せない**: 4 kHz に切った公開データの質問 40 問 × 3 回で、返事が質問を正しく理解した割合は、音声だけ 101/120、音声＋書き取り 117/120。
  - 書き取りを、区切りまでの書き取りの続きから書かせる方式も試した。速くはなった（95 → 60 ms）が、単語の誤りが 7.1 % → 16.8 % に増えたのでやめた。
  - したがって PROTO での「話し終わり→声」は約 0.6 s が下限。実機では 0.19〜0.34 s（話し終わり判定から最初の PCM まで）の回が多く、文の途中で間を空けた回は 1.7〜2.0 s かかった（後半を待ってつなぐため）。

### 速さ・聞き取り・体感の全体改善（2026-11-08 02:40、ユーザー「他社の構成を参考に、聞き取りと応答速度、体感をよくする改善策を」）

- **計測の誤りを直した**: `omni_probe.py` は毎回まったく同じ音声を流していたため、2 周目は Thinker の書き取り（音声で引くキャッシュ）と vLLM の prefix cache に当たり、0.1〜0.2 s 速く出ていた。今は毎回 −90 dBFS の雑音を足して別の音声にしている（`PROBE_SAME_AUDIO=1` で従来どおり）。実際に近い値は、11-08 の変更前で **0.64 s**（0.50〜0.77、3 問）。
- **遅れの正体は、返事が書き取りを待つこと**（`basic_memory_agent` → `omni_addressee.HEARD`、90〜240 ms）。書き取りを切ると 0.47 s になる。
  - 書き取りは 11-04 に、4 kHz で切れた PROTO の音で Thinker が聞き違えたので入れたもの。全帯域の音では逆に足を引っ張ることもあった。書き取りが「Omaha」を「Owa / Ohio」と聞き違え、返事が「The current time in Ohio is not available」になった。同じ音声を直接聞かせると正しく時刻を答えた。
  - **返事が書き取りを待つのは、音声が 4 kHz で切れているときだけにした**（`omni_addressee.reply_needs_heard`）。判定には話者や内容を使わず、4.1〜4.6 kHz と 3.4〜3.9 kHz のエネルギー差（dB）を使う。
    - 実際の PROTO の録音: −29〜−30 dB
    - PROTO を模した音: −17〜−23 dB
    - 全帯域の声（合成音声・YouTube）: −7.5〜+8.5 dB
    - 閾値 −12 dB（`OMNI_HEARD_CLIFF_DB`、`off` で常に待つ）。ログは `[omni] audio band edge … dB at 4 kHz`。
  - 全帯域の音では **0.51 s**（0.45〜0.60）。**PROTO の今のマイクでは従来どおり待つ**ので、PROTO では速くならない。マイクが直れば自動で速くなる。
  - 書き取りそのものは「文が途中か」の判定のために並行して続けている（GPU1 を取り合う分で約 0.04 s）。
- **耳（音声だけの宛先判定）は必要なときだけ呼ぶ**: 前に人がこちらを向いていれば呼ばない（`_addressee_policy` の `ears_audio`）。区切りごとに Thinker への音声要求が 1 本減る。
- **聞いている合図（うなずき）**: 相手の区切りと話し終わりで、サーバーが描画タブへ `{"type":"control","text":"listening-nod"}` を送り（`_maybe_listening_nod`）、頭が 4° 下がって 0.45 s で戻る（`ATTEND.listenDeg` / `listenDur`、`__attend({listenDeg:0})` で無効）。
  - 送らない条件: 発話 0.6 s 未満、彼女が話している最中、2.5 s 以内（`LISTEN_NOD_GAP_S`）、前の人が横を向いている・誰もいない。`LISTEN_NOD=0` で止まる。
  - 描画タブで頭の角度を測って確認済み（合図あり 4.00°、無効時 0.01°）。返事の声より 0.3〜0.5 s 早く反応が見える。フロントは再ビルド済み（`main-flHD38jR.js`）。
- **声と頭の生死**: `./kehai.sh status` が omni モードで :8808（Thinker）と :8809（Qwen3-TTS）を確認する。11-07 23:30 に TTS が外部から止められ、約 2 時間、誰も気づかないまま黙っていた。01:32 に別のセッションが再起動済み。
- **GPU の掃除**（ユーザー「KEHAI で使用していないものはすべてアンロード」）: GPT-SoVITS（:9880、2 GB）を停止し、gateway DB の `gpt-sovits` を `enabled=0` にした（バックアップ `prisma/dev.db.bak-unload-20261108`）。PC-A の KEHAI や公開 gateway で GSV を使う日は 1 に戻す。
  - GPU0 に残っているのは KEHAI が使うものだけ: 描画 Chrome・Whisper large-v3・GestureLSM・Qwen3-TTS・MPS。GPU1 は Thinker。
  - GPU1 にはほかに OS のデスクトップ（Xorg と gnome-shell、約 170 MB）が載っている。止めるには sudo が要るので残した。
- **PROTO のマイク診断**: `scripts/proto/proto-audio-check.sh` を **PROTO 本体で**実行する（読み取り専用）。入力の構成（仮想入力・LADSPA の連鎖）を一覧し、入力ごとに 5 s 録って「4 kHz で切れているか」を測る（parec と python3 の標準ライブラリだけで動く）。
  - 全帯域の入力が見つかったら、ビューワの URL に `?micid=<入力名の一部>` を付ける（`render_view.html` に追加）か、`pactl set-default-source` で既定にする。
- 未着手（判断が要るもの）:
  - 方向の分かる USB マイクアレイ（エコー除去・ビームフォーミング・音源方向。Furhat はこれで話者を分けている）
  - PROTO 側 Chrome の音声バッファを縮める（`PULSE_LATENCY_MSEC=30` 等、出力 −60〜100 ms の見込み）
  - 聞き取った文の字幕
  - 近づいて立ち止まった人への声かけ
  - 声が出ないときの画面表示

### 「固まるようになった・返事が遅くなった」→ 取り込みを GPU 側に（2026-11-05 15:30、ユーザー「60 FPS を維持できるように」）

60 fps 化（下の節）のあと、話し始め・話している最中に絵が 0.1〜0.35 s 止まり、TTS は PIECEWISE で遅くなり（最初の音 64〜120 ms、
返事の頭で声が 20〜40 ms 途切れる）、それを 60 fps のための代償だと思っていた。**本当の原因は ffmpeg の x11grab**。

- 切り分け（描画タブで WebGL のタイマークエリ、別の Xorg :59 と試験 Chrome、`burnq.py` = 行列積を N 個積んでから同期）:
  描画そのものの GPU 時間は 1 コマ 3.6 ms（Whisper 中も最大 8 ms）で、遅くなっていない。止まるのは **GPU に仕事を積む CUDA プロセス**
  （Whisper の encoder/decoder、TTS のグラフ、GestureLSM）が動くときだけ（1 回ずつ同期する計算なら止まらない）。さらに **:59 で取り込みを止めると
  同じ負荷でも 60 fps のまま、x11grab を動かすと 15 fps**（`-c:v rawvideo` でも同じ＝NVENC ではなく X の読み出し）。x11grab の XShmGetImage は
  X サーバーが GPU から画面を読み出し、その読み出しは GPU に積まれた CUDA の仕事が捌けるまで待つ。X サーバーは単一スレッドなので、その間
  Chrome の Present も待たされる（100〜700 ms）。
- 効かなかったもの: Whisper を MPS の外に出す、CUDA_LAUNCH_BLOCKING（止まりは消えるが Whisper 3.2 倍遅い）、Vulkan のキュー優先度を上げる
  レイヤー（VK_EXT_global_priority、ドライバーが一般ユーザーに HIGH を `NOT_PERMITTED` で拒否）、CPU 負荷・メモリ確保の繰り返し・nvidia-smi は無関係。
- **直したもの: `scripts/pcb-render-x/glgrab/kehai-glgrab`**（C、`build.sh`、無ければ `start_server.sh` がビルド）。描画窓の composite pixmap を
  GL テクスチャとして受け（GLX_EXT_texture_from_pixmap、NVIDIA は FBO に付けられないので四角形で描き写す）、**NVENC が GL テクスチャを直接読む**
  （NV_ENC_DEVICE_TYPE_OPENGL）。画素は CPU を通らず、待つのはこのプロセスだけで X サーバーは待たない。設定は従来の ffmpeg と同じ
  （P1・ULL・CBR 30 Mbps・GOP 60・AUD・IDR ごとに SPS/PPS・RGB 入力の VUI = BT.601 限定範囲）。色と向きは x11grab と一致（平均差 0.6）。
  違いは CBR の埋め草が無いこと（静かな場面は 13〜24 Mbps）。`render_stream.py` が `RENDER_CAPTURE=auto` で使い、
  落ちたらビューワ接続中は自動で起こし直す（1 分に 5 回まで）。`RENDER_CAPTURE=x11grab` で従来に戻る（そのときは TTS も PIECEWISE に戻すこと）。
  - 診断: `kill -USR1 <kehai-glgrab の pid>` で今エンコードしている絵を `/tmp/kehai-glgrab-<pid>.ppm`（1/4 サイズ）に書く（「PROTO が黒い？」の確認）。
    ログ `[render] [glgrab] last 60 s: N frames, ... late (GPU wait max N ms)`。
  - 罠: **1 つの窓を GL に結べる取り込みは 1 つだけ**（2 つ目は BadAlloc で黒を出す）。本番の取り込み中に試験で 2 つ目を起動しない。
- 結果（PROTO 経路の実際の返事 3〜4 回）:

  | | 描画 fps | 50 ms 超のコマ | 最悪 | 声の途切れ | TTS 最初の音 |
  |---|---|---|---|---|---|
  | x11grab ＋ PIECEWISE（14:30） | 56.8〜58.8 | 9〜16 | 117〜250 ms | 1〜3 回 | 64〜120 ms |
  | **GPU 取り込み ＋ フルグラフ TTS ＋ 元の画質（MSAA 8・TAA 等倍）** | **60.0** | **0** | 33 ms | **0** | **54〜62 ms** |

  Whisper 単独中も 60.2 fps・止まり 0（Whisper は 0.55 → 0.63〜0.69 s、描画に公平に GPU が回る分。omni では返事の経路外）。
  待機中の GPU0 使用率 62〜77 % → 39 %（X の 4K 読み出し 2 GB/s が消えた）。話し終わり→最初の PCM は 0.24〜0.64 s（文中に間のある発話の合流だけ 1.2 s、以前から）。
- 調査中に下げた画質（MSAA 4・TAA 0.75 倍）は**元に戻した**（描画タブの `localStorage.vrmSpace1`）。TTS は `serve-qwen3-tts.sh` の既定をフルグラフに戻した。

### PROTO を 4K 60 fps に（2026-11-05 14:05、ユーザー「そもそも 60 FPS にしたい」「60 FPS で動くようになるまでやって」）

（15:30 に上の節で更新: 3 の PIECEWISE はもう不要、取り込みは x11grab ではなく kehai-glgrab）

それまでは 30 fps 止まり。原因は描画用 Chrome が Xvfb（CPU の仮想画面）に描いていたこと: 毎コマ 4K を GPU から読み戻して書き込む
`SwapBuffers` が 29 ms/コマ（何を描いても 4K で 15〜30 fps、1080×1920 なら 60）。今の構成（全部そろって初めて話している間も 60）:

1. **描画用 X サーバーを GPU0 に**: Xvfb :57 の代わりに Xorg :58（`/etc/X11/kehai-render-x.conf`、元は `scripts/pcb-render-x/`。画面出力なし
   `UseDisplayDevice none`・入力なし・vt8 を共有して物理モニター（GDM の :0、同じ GPU0 に DELL がつながっている）には触らない）。
   Xorg.wrap に `allowed_users=anybody / needs_root_rights=yes`（ユーザーが sudo で設定）。root で動くが実 uid は自分なので kill できる。
   `scripts/render-host.sh` は設定ファイルがあれば自動で gpu（`RENDER_X=xvfb` で旧構成）、使った画面を `.render-display` に書き、`start_server.sh` が `RENDER_DISPLAY` にする。
2. **Chrome は `--use-angle=vulkan --enable-features=EnableDrDc --enable-angle-features=asyncCommandQueue`**（render-host.sh）。native GL は
   話している間 1〜5 fps、`VulkanFromANGLE,DefaultANGLEVulkan` を付けると読み戻し経路に戻り 83 ms/コマ。**`DRI_PRIME=pci-0000_1a_00_0` 必須**:
   無い試験用 Chrome は GPU1（Thinker、空き 1.3 GB）に乗り、メモリ確保に失敗した（Thinker は無事だった）。
3. **Qwen3-TTS の talker を PIECEWISE CUDA グラフに**（`serve-qwen3-tts.sh`、GPU0 の既定、`QWEN3TTS_EAGER`）。フルグラフの連続リプレイ中は
   GPU の時分割で描画に順番が回らず 1〜5 fps（`FinishPaintCurrentFrame` が 1 回 0.5〜1.1 s）。TTS だけ流した描画 fps / TTS の速さ:

   | talker | 描画 | TTS |
   |---|---|---|
   | フルグラフ（元） | 1〜43 | 実時間の 4〜5 倍 |
   | フルグラフ＋ステップごとに 12 / 30 / 50 ms 休む（`KEHAI_TTS_STEP_SLEEP_MS`、`tts_pace/sitecustomize.py`） | 12〜24 / 39〜48 / 51〜57 | 3.8 / 2.1 / 1.4 倍 |
   | eager（グラフなし） | 51〜61 | 1.4 倍（声が文の頭で尽き、動きも遅れる — 不採用） |
   | **PIECEWISE ＋ 下の 4・5 ＋ Chrome フラグ** | **60** | **2.1〜2.2 倍** |
4. **CUDA MPS（GPU0）**: `scripts/pcb-mps.sh {start|stop|status|env}`、`/tmp/mps-gpu0`。TTS の 2 段・Whisper・GestureLSM が 1 つのコンテキストに
   （計算プロセス 3 つ同時で描画は数秒止まった → MPS で 27〜32 fps＝1 つ分）。`pcb-stack-up.sh` が最初に起動、各起動スクリプトが
   `eval "$(pcb-mps.sh env)"` で参加（Nexus の `model-service/run.sh`・`gesture-service/run.sh` も変更、写し `*.bak-mps-20261105`）。
   確認は `nvidia-smi` で種別 `M+C`。罠: ソケットのパスが長いと（108 字超）クライアントは**黙って MPS なしで動く**。どれか 1 つが GPU で落ちると全員巻き込まれる。
5. **環境遮蔽（GTAO）を描画タブだけ半分の解像度**（vrm-viewer.tsx、`localStorage.gtaoScale`）。画面写真で見分けがつかない。
6. **配信の保護**（render_stream.py、render_view.html、run_server.py）: 書き込み待ちが 256 KB（`RENDER_VIDEO_BUF`）を超えたら映像をキーフレームまで捨てる
   （声を映像の後ろで待たせない・溜めて流し込まない）、1 回の送信が 3 s（`RENDER_SEND_TIMEOUT_S`）終わらなければ切断（ビューワは自分で再接続）、
   WebSocket の permessage-deflate を無効（H.264 をイベントループで圧縮していた）、`gc.freeze()`（起動時の 218 万個を GC の対象外に、340 ms の停止が消えた）。
   ビューワは 4 s 何も来なければ自分で張り直し、統計に `lag`（メインスレッドの遅れ）と `skipped`。
7. GPU0 の計算タイムスライスを Short に（ユーザーの sudo、`nvidia-smi compute-policy -i 0 --set-timeslice 1`、再起動で Default に戻る）。単独の効果は確認できていない。

結果（長い返事 3 つ）: 描画 65 s 中 最低 53・中央値 60（55 未満は 1 秒）、PROTO へ 60 fps・捨てたコマ 0・PROTO の描画 59〜60、声の遅れ 0・尽き 0、
動き 96〜100 %、話し終わり→声 0.30〜0.63 s。TTS は 2 倍なので文の頭も動きのタイミングも足りる。

- 戻し方: `scripts/render-host.sh stop && RENDER_X=xvfb scripts/render-host.sh start && ./kehai.sh restart`（30 fps）。TTS をフルグラフに戻すのは
  `QWEN3TTS_EAGER=0` で :8809 を再起動。MPS をやめるのは `scripts/pcb-mps.sh stop` のあと 3 サービスを再起動。
- 罠: ゲートウェイの GestureLSM は同時 1 件で、処理中に gesture_service を kill すると枠が残り 15 分 429 になる。試験の Chrome は必ず `DRI_PRIME` を付け、
  本番の :58 には窓を開かない（取り込まれて PROTO に映る）— 別の Xorg :59 を同じ設定で立てる。`render-host.sh stop` は X が消えるまで待つようにした
  （待たずに start して「Missing X server」で起動失敗した）。`pgrep -f gesture_service.py` は自分のシェルに当たる（`[g]esture_service.py`）。
- 残り（14:05 時点）: Whisper large-v3（話し終わりの書き起こし）が単独で描画を 2 秒ほど 20〜50 fps に下げる。→ 15:30 に解消（上の節: 原因は x11grab）。

### 「映像と音声がぷつぷつする」（2026-11-05 11:35）

**音声の原因: GestureLSM のクリップが届くたびにイベントループが 110〜190 ms 止まっていた。** PROTO のビューワへの声は 20 ms ずつ
実時間で送っていて（`render_stream._pace_direct`）、ビューワの手持ちは約 60 ms（先行 40 ms＋ジッタ床 20 ms）。クリップの後処理
（`gesture_client` の顔の向きの補正 `_face_front` と MMD 用 .vmd への変換）が純 Python で全フレームを回し、その間ループが止まり、声のフレームが
50〜190 ms 遅れてビューワの手持ちが尽きた（starve）。映像の中継も同じループなので、同じ瞬間に映像も引っかかった。
- 計測: `LOOP_LAG=1 LOOP_LAG_STACK_MS=40 ./kehai.sh restart` で、止まった瞬間のメインスレッドと（新たに）他の忙しいスレッドのスタック。
  `[render] direct PCM late N ms (woke … late, send+lock … ms)` と `[render] direct PCM ran dry` は常時出る（遅れて手持ちが尽きたときだけ）。
- 今日以前から（starve は話している 1 分あたり 5.0 回）、今日の 8 秒クリップで悪化（6.1 回）。スレッドに逃がすのは効かない（GIL を握ったままの
  純 Python と C の json.dumps はループを止め続ける）。
- 直した: (1) .vmd は **最初に `/motion/clip/{id}` が来たときに作る**（`motion_cache._vmd_from_joints`、`MOTION_VMD_EAGER=1` で従来どおり）。
  VRM（Iris）は .vmd を一度も取りに来ないので 60〜170 ms がまるごと消える。(2) `_face_front` をベクトル化（21 → 3 ms、結果は旧と 0.05 mm 以内）、
  出力を 0.1 mm に丸める（float16 由来なので情報は落ちない。JSON の作成 12.5 → 6.7 ms、大きさ 46 %）。
- 結果（長い返事 3 つ）: 遅れた声のフレーム 30〜40 → **0**、starve 0、ループの最大停止 105〜188 → 23〜40 ms、クリップ後処理 110〜270 → 3〜12 ms。
- **残り（映像）**: 返事の出だし約 5 秒、描画タブが 30 → 22〜23 fps に落ちる（GPU0 が 100 %）。音声合成（Qwen3-TTS が実時間の 4 倍で返事全体を
  一気に作る）が主で、GestureLSM の生成が後に続く。描画（Chrome、2160×3840）と同じ GPU0 にいるため。対策は描画の解像度を下げる・TTS を
  実時間に近い速さに抑える（総量は同じで分散するだけ）・GPU の配置を変える、のどれも代償があり未実施。
- 待機中でも KEHAI は CPU 約 1.3 コア（メインスレッド 39 %、他のスレッド 60 %）。中身は未調査（py-spy は ptrace 制限で外から付けられない）。
- **映像のエンコーダを GPU1 に移す試験（11:45、不採用・GPU0 に戻した）**: `RENDER_NVENC_GPU=1` で ffmpeg（x11grab → h264_nvenc）を GPU1 へ。
  返事中の GPU0 の使用は ffmpeg 22 %・描画 19 %（常時）＋TTS 45 %・GestureLSM 25 %（返事中）で、ffmpeg が常時最大だったため。
  結果: GPU0 の待機時の使用は 35〜49 → 23〜30 % に下がり、Thinker は問題なし（GPU1 23,962 / 24,576 MiB で増えず、エラー 0、話し終わり→声 419〜772 ms）。
  **しかし描画は遅くなった**: 待機時 30.4 fps → 25.9 fps（50 ms のコマ 149 → 250 / 30 s）、返事中の最低は 23 fps のまま。描画の律速は GPU0 の計算ではなく
  **4K のコマの往復**（Chrome が GPU0 から読み戻して Xvfb :57 に書く → ffmpeg が 60 fps で取って GPU に上げ直す。待機時の CPU: Chrome GPU プロセス 73 %、
  Xvfb 71 %、ffmpeg 46 %、GPU0 の送出 1.8 GB/s）。次の候補は取り込みを 30 fps に（描画は 30 fps しか作っていない。往復が半分になる。代わりに
  取り込みと描画の位相ずれでコマの重複・抜けが出うる）か、解像度を下げる。

### 話している最中の停止・出だしの空白・話し終わりの動き・レムの立ち姿（2026-11-05 10:50、修正済み）

ユーザー「1 と 2 の根本原因を見つけて対策して、喋り終えた後も少しの間モーションが付いていないか確認、レムの立ち姿を使うプログラムは Iris に一切いらない」。
測り方: 描画タブ（CDP :9222）で `window.__vrmMotion` を毎フレーム読む（`statsSnapshot()` の区分 clip / hold / proc / idle、`elapsed`、`_holdT`、
`_talking`、正規化骨の 1 フレームの回転量）＋ `endTurn` をフックして話し終わり後の骨の動き。試験は `omni_probe.py --via-viewer --cam` で長い返事 4・挨拶 1。

| | 修正前 | 修正後（5 ターン、話している 72 s） |
|---|---|---|
| 話している間の停止 | 5 回 × 0.6〜0.8 s ＋ 出だし 0.46 s / 1.75 s | 0（1 フレームの proc が 4 回だけ）、clip 99〜100 % |
| 話し終わり後の身振り | 1.3 s 身振りのまま → KAWAII の待機へ大きく移る（8〜17°/フレーム、0.75 s） | 0.3 s で止まり、1.2 s で立ち姿へ（最大 1.4〜4.8°/骨/フレーム＝話している間と同じ） |

1. **文の途中で止まる → 原因は GestureLSM の窓**: 128 フレーム（30 fps）の窓を 16 重ねて作るので、音声 n フレームから出る動きは
   16 + 112·⌊(n−16)/112⌋ フレームで、端数は捨てられる。`gesture_client` が音声を 6 s に詰めていた＝窓 1 つ＝**4.27 s が上限**（`dur` を何秒頼んでも）。
   さらに早撃ち（文の頭 0.35 / 1.2 s の音声＋無音）だったので、クリップの大半は「黙って落ち着く」動きだった。
   → `gesture_client._window_cover_seconds`: 頼んだ長さを覆う窓の数まで音声を詰める（4.5 s → 8.07 s の音声）。`tts_manager`: LSM-only のキャラは早撃ちをやめ、
   **文全体の音声ができてから**その文の長さ＋0.6 s を頼む（2 文目以降は前の文の再生中に作り終わる）。LSM-only は毎文 due（間隔 4 s の間引きをしない）。
2. **出だしに動きが無い → 原因は WS 経路が END フレームで適用していた**: 1 文目の音声を作り終えるまで動きが付かない。→ `websocket-handler.tsx` の WS 経路でも
   **START フレームの動きを再生原点で適用**（WebRTC 経路と同じ）、END は重複を飛ばして生成クリップ（upgrade）だけ。
   - 返事の **1 文目は「GestureLSM が続いている」判定で飛ばさない**（割り込まれた前の返事のクリップが残っていると、新しい返事の出だしを 3.7 s 止めていた）。
   - 作り置きの出だし（BRIDGE、2.7 s）は話している間は**待たずに巻き戻す**（`bridge: true`、`holdWrapAfter` 0.8 s を使わない）。長い 1 文目は生成クリップが
     約 3.5 s 後に届くので、その間 0.7 s 止まっていた。
3. **話し終わり後**: `endTurn` で最後のクリップが自由再生のまま 1.3 s 身振りを続け、その後 KAWAII の待機に移っていた。→ ロボット（`robotIdle`）は
   身振りを `robotGestureStop` 0.3 s で止めつつ、`robotWindDown` 1.2 s の smoothstep で立ち姿へクロスフェード（壁時計。描画の dt は 1/30 で切られ
   29 fps では 1.5 倍遅れる）。`endTurn` は返事ごとに 2 回呼ばれる（再生終了と conversation-chain-end）ので、2 回目で計時をやり直さない
   （やり直すと前腕が 1 フレームで 24° 戻った）。調整は `__vrmMotion.robotWindDown` / `robotGestureStop`。
4. **レムの立ち姿を Iris から外した**:
   - KAWAII の待機クリップ: `active.json` の `by_conf` / `engaged_by_conf` の iris_android_001 を空に（空＝待機クリップなし）。待機中はロボットの立ち姿（サーボの首）。
   - GestureLSM の姿勢補正 `POSTURE` は KAWAII の待機（レム／ミクの人の立ち方: 背中 10° 前傾、首 3° 左に傾げる、肩をすくめる）を基準にしていた。
     ロボットは `_postureRef` で**自分のリグの休み姿勢**（`restDir` から: 背骨 −6.2°、首 +8°、傾き 0、肩 0）と腕 7° 外を基準にする。
   - **作り置きの出だし（BRIDGE）も Iris の声で作り直した**（11:10）。旧 `opener.json` は日本語の一文「ええ、そのとおりだと思います。」から
     作った 2.7 s（旧時代の声）。Iris 用は `motion-lib-bridge/iris.json`（3.8 s、114 フレーム）、`characters/iris_android.yaml` の
     `gesture_bridge: iris` で選ぶ（空なら共通の opener。他のキャラは従来どおり）。`motion_lib` はフォルダの `*.json` を全部読み、
     起動ログ `bridge 'iris' loaded`、返事ごとに `LSM-ONLY bridge 'iris' (114f)`。
     作り方: 本番と同じ声（Qwen3-TTS CustomVoice aura、文末の引き伸ばし込み）で Iris らしい出だし 10 文 → GestureLSM（チェーン seed なし、毎回違う）
     各 10 本 = 100 本 → 品質の関門（動きの量 p30〜p85、ぎくしゃく ≤ p70、腕を下ろした姿勢からの距離 ≤ p60、巻き戻しの継ぎ目 ≤ p60、
     体のねじれ ≤ 20°、手の前出し ≤ p85、手首が胴に入らない）→ 通った 6 本のメドイド。選んだのは "Understood. Let me explain the idea in
     a few simple steps." の 5 本目。旧と同じ尺度で: ねじれ 39° → 15°、手の前出し 0.45 → 0.28 m、巻き戻しの継ぎ目 0.083 → 0.034 m、
     出だしの腕 0.31 → 0.21 m、動きの量 0.0099 → 0.0070（落ち着いた）。指標の定義は `iris.meta.json`。作り直すときの道具（gen.py / select.py）は
     この日のセッションの scratchpad のみ（中身は上の手順どおり）。
- 戻し方: 待機を戻す＝`active.json` の iris_android_001 に元の `by_conf`（idle45_wavehandslightly, idle08_comeupwithanidea, idle40_crosslegs,
  idle02_lookleftandright）と `engaged_by_conf`（idle01_breathing, idle09_waiting, chan_idle02_female）を入れて再起動。

### アイリスは GestureLSM だけで動く（2026-11-05 09:55、ユーザー「KAWAII は Iris では使わなくていい、リアルタイムモーション生成はどこ行った？」）

- omni では 11-02 から GestureLSM（声から体の動きをその場で作る）を止めていた（GPU1 を Thinker に空けるため、当時のユーザー判断）。
  そのため話している間の動きは KAWAII の talk ループ（talk_normal / talk_chatty / talk_explain）と wave だけだった。
- **今の構成**: `characters/iris_android.yaml` に `gesture_lsm_only: true`（KAWAII の talk・一回きりの動きを使わない。各文の頭は作り置きの
  BRIDGE、生成した動きが届いたら重ねる）。omni の `MOTION_LSM_UPGRADE` 既定を 1 に（`start_server.sh`）、`pcb-mode.sh` の omni は
  `gesturelsm-1` を enabled=1 に。起動時に GestureLSM を温める（On Demand なので最初の要求で起動、`[warmup] gesture model is warm`）。
- **GPU**: GestureLSM は **GPU0**（1.2 GB）。`AICCORE-Nexus/gesture-service/run.sh` の既定が「GPU0」のコメントのまま実際は `GESTURE_GPU:-1` だったので 0 に直した
  （写し `run.sh.bak-20261105`）。GPU1 の Thinker の横に載せると OOM の恐れ（11-02 の GSV と同じ）。ゲートウェイ DB の写し `prisma/dev.db.bak-mode-20261105`。
- 実測: 長い返事で動き 96%（14.4 s）、停止 0.6 s、GestureLSM 3 本適用、KAWAII 0。生成は 1 文 0.9〜1.9 s、順番待ちで声の開始から 2〜3 s 後に届く
  （その間は BRIDGE）。TTS の最初の音 74 ms（変化なし）。GPU0 14.8 GB。
- 待機中（話していないとき）は **10:50 から KAWAII の idle を使わない**（上の節）。cascade の Qwen3-TTS 構成は
  `MOTION_LSM_UPGRADE=0` のまま。Iris の人格の `[g:…]` タグは GestureLSM だけの今は動きに使われない（文から消えるだけ）。

### 「talking motion が話してる途中で消える」（2026-11-04 19:40）

描画タブの `[motion-usage]` で、長い返事は動き 21〜31%・最後の姿勢で停止 7〜13 秒・適用 1 本。動きが付くのは毎ターン最初の 2〜3 秒
（話し始めのつなぎ BRIDGE）だけだった。原因は 2 つの重なり:
1. BRIDGE は GestureLSM 製なので `gesturelsm: true` の印付き（`motion_lib.bridge_clip`）。描画タブは GestureLSM の動きが流れている間、
   KAWAII の talk ループを流さない（`gestureLSMActive()`、文体の切り替わりを防ぐため）。
2. その判定の「+1.5 s で抜ける」安全弁が効かない: 再生ヘッドは最後の姿勢で止まると進まない（`update()` の free-running 分岐）ので、
   `elapsed < clipLen + 1.5` が永遠に真。→ 2 文目以降の talk はすべて捨てられた。omni は `MOTION_LSM_UPGRADE=0` で後続の GestureLSM も来ないので毎ターン固まる。
   長い返事が増えたので目立った（cascade の 10 時台は 69%）。
修正: (a) BRIDGE の印は `MOTION_LSM_UPGRADE=1` のときだけ、(b) `gestureLSMActive()` に `this._holdT < 1.5`（止まっている時間で抜ける）。
描画タブを `bash scripts/build-frontend.sh` で作り直した（`main-cjaLmFQd.js`、前回のビルド以降にソースの変更は無かった）。
実測: 長い返事で動き 96%（13.4 s）、停止 0、適用 3 本。

### 「うまく会話が成立しない」（2026-11-04 19:20）

原因は 4 つ（英語の訛りだけではない）:
1. **PROTO のマイクが 4 kHz で切れている**（ユーザーの 60 発話すべてで 4.2 kHz 以上がほぼ空、比 6e-6、普通は 1e-2）。子音が消え、Whisper と
   Thinker の聞き取りが 3〜4 割で食い違う。PROTO の OS の音声入力の設定（10-23 の調査、`pactl` で見る）。こちらからは直せない。
2. **Thinker が音声から直接答えると聞き違える**: 同じ音声を「書き取って」と頼むと "I want to get information about nearby supermarket or market"
   と正しいのに、人格の下で直接答えさせると大学の停電の話をした。→ **返事の前に Thinker に書き取らせ（`AsyncOmniLLM.transcribe`、100〜230 ms、
   `OMNI_HEARD=0` で無効、`OMNI_HEARD_LANG`）、その文を音声に添えて渡す**。記憶にも Whisper の代わりにこの文を残す（履歴とパネルは Whisper のまま）。
   ログ `[omni] heard in N ms: '...'`。9 件中 8 件の返事が噛み合うように。最初の声が 0.1〜0.2 s 遅くなる。
3. **文脈の手本**: 私の 18:43 の試験（"why CCORE…" → "…Sticiowa…"）を履歴から消したが**記憶を読み直さなかった**ため、Thinker がそれを手本に
   同じ文を返し、本物の会話として履歴に保存され、以後も手本になった。午後の雑談の決まり文句も同様。今日の履歴（文脈あり）では天気の質問で検索を
   1 回も呼ばず、文脈なしでは毎回呼んだ。→ 今日の Iris の履歴を `chat_history/_archive/iris_android_001/` へ移し、会話を新しく始めた（戻すなら元の場所へ）。
   **試験のあとは必ず履歴を戻してから描画タブを再読み込みするか KEHAI を再起動する**（履歴ファイルだけ戻しても動いている記憶には残る）。
4. **予告だけで道具を呼ばない**: "I can check that for you." で終わり検索しない。→ (a) 予告だけの返事・「I am not aware of …」系の断りだけの返事・
   言い回し違いの繰り返しを文脈に残さない（`is_announce_only`、`is_non_answer` の拡張、`_near_repeat`）、(b) 道具を渡したのに予告だけで終わった回は
   1 度だけ「今すぐ道具を呼んで」と促してもう 1 周（`_TOOL_NUDGE`、ログ `[tool-nudge]`）。天気（"partly cloudy with a high near 52°F"）と
   近くのスーパー（Walmart, Aldi…）が実際に検索して答えるように。

### 「話すときと話さないときがある」（2026-11-04 18:55）

原因は**耳の判定（音声だけ）の見逃し**。18:47〜18:49 にユーザーが PROTO の正面（推定 2.8〜3.3 m、こちら向き）から話した 13 発話のうち、
"Can you describe what today's news is all about?" "Can you search about today's weather?" "Thank you." "What a good job." などが
ears 0/9 で捨てられた。合成音声の評価（28/30 通過）は本物の PROTO マイクの声には当てはまらなかった。
**修正**: `websocket_handler._followup` を「返事の後 6 秒以内」に限らず、**発話の間に YuNet が近く（4 m 以内）でこちらを向いた顔を見ていれば、
耳が 0 でも答える**に（ログ `ears 0/9, taken as hers: someone near is facing her`）。耳が決めるのは、近くに向いた顔が無いときだけ。
彼女が話している最中に先読みを始めるかの判断も同じ。`face_detect.Attention.MAX_M` 3.5 → 4.0（正面の人が 3.6 m と推定されて外れていた）。
根拠: 判定画像に YuNet を当て直すと、午後の雑談（16:00〜17:28、奥の机 5〜7 m）では近くでこちらを向いた顔は 0。
試験: 捨てられた本物の録音を、こちら向きの写真と流すと答え、奥の机の画像と流すと捨てる。
**未確認**: 話し始めの関門（68 dB、平滑化した音量）の下で消えた発話があったか。記録が無かったので、関門の下の話し声
（Silero が 0.6 以上で 0.32 s 以上）を `[vad-level] speech UNDER the onset gate: N ms, loudest X dB` で残すようにした。
これが出ていたら関門（`VAD_ONSET_DB`）を下げる。ユーザーの発話の山は 70〜80 dB。

### 「文末が早口」（2026-11-04 18:45）

原因は **Qwen3-TTS の Aura 学習版（r9e3）が文の後半で減速しないこと**。Aura 本人の録音と同じ 16 文を合成して比べると、
後半÷前半の話す速さ（Whisper の単語時刻、単語自体の長さだけで割る）は本人 0.92・TTS 1.15（16 組中 15 組で TTS が速い）。
Whisper を使わない音量の山の数え方でも本人 0.91・TTS 1.05〜1.25。文全体でも TTS が 4〜12% 速い。学習に混ぜた Base＋参照の合成データ
（`aug_synth` 837 本）は 1.43 と強く加速していて、それを受け継いだと見られる。文の長さとは相関しない（-0.06）。
効かなかったもの: 抽選温度 0.7 / 0.5（1.14 / 1.21）、繰り返しペナルティ 1.05→1.0（`extra_params` は temperature / top_p / top_k しか
抽選に反映しないので deploy yaml を変えた試験用サーバーで確認、1.31）、合成データを外した 912 行で r9e3 から 2 エポック（r13、1.18、
声も高くなり B 組 116 Hz、不採用・削除済み）。文の後ろに "..." を足すのも効かない。
**直し方**: `tts/qwen3tts_stream.py` の `_Slowdown`（自前のストリーミング WSOLA、10 ms 刻み・20 ms 窓・±5 ms で波形を合わせる、音の高さは不変）。
文の長さを文字数 / `QWEN3TTS_CPS`（18）で見積もり、`QWEN3TTS_SLOW_FROM`（0.4）から文末 `QWEN3TTS_SLOW_END`（1.4 倍）まで徐々に引き伸ばす。
結果: 後半÷前半 0.98（音量の山 1.00）、全体 4.38 音節/秒（本人 4.42）、Whisper の誤り率 3.6% → 2.5%。1 秒分に 1.9 ms。
最初の声は 10 ms 遅れる（最初の窓の立ち上がり）。文の終わりは 10 ms でなだらかに消える。`QWEN3TTS_SLOW_END=1` で無効。
測る道具（セッションの scratchpad のみ）: 本人の録音（`stack/voice/aura_gsv.list`）と同じ文を合成し、Whisper small の単語時刻で前半・後半の速さを比べる。

同時に直したもの:
- **ビューワが声を捨てていた**: 1 秒間の溜まりの最小値が 目標＋40 ms を超えると先頭から捨てる処理が、話している最中の 20〜40 ms を消していた
  （統計 `play 80〜100/20` の時。午後の約 2% の秒）。→ 捨てるのは無音のコマだけ（`render_view.html` の `quiet()`）。統計行に `N ms trimmed`。
- **溜まりが増える原因**: 送り出し（`render_stream._pace_direct`）が一瞬空になって再開するたびに、ビューワに残っている分を無視して
  先行 40 ms を足し直していた → 残量を差し引いて再開する。
- **映像が戻らない**: ビューワのページを読み込み直したあと ffmpeg が終わっていたのに `hub.nvenc` が残り、再接続しても映像 0 fps のままだった
  → `start_video` が止まった ffmpeg を作り直す。ffmpeg の終了コードを必ず記録（`[render] NVENC capture ended (ffmpeg exit N)`）。
  終わった理由はまだ不明（18:40 の読み込み直しの直後、ログ無し）。
- **読み込み直しの罠**: `POST /render-stream/reload` のあと、PROTO の Chrome は音の再生を許さず（AudioContext が suspended）、
  開始画面「タップして音声を有効にします」で止まる。この間はマイクの取り込みも止まるので彼女は聞こえない。`localStorage.autostart` では足りない。
  PROTO の Chrome に `--autoplay-policy=no-user-gesture-required` を付けるまでは、読み込み直しのたびに誰かがタップする必要がある。

### 使っていないものを GPU から降ろした（2026-11-04 09:45、ユーザー「使ってないなら GPU に載せてほしくない」）

- **GPT-SoVITS（:9880、GPU0 2.2 GiB）を omni で停止**: 返事の声が Qwen3-TTS のとき、`./kehai.sh mode omni` が gateway DB の `gpt-sovits` を
  `enabled=0` にして止める（`pcb-mode.sh` の `stop_gsv`、`omni_tts` は start_server.sh と同じ規則）。停止前の確認: GSV のログで最後の実際の発声は
  当日朝の学習データ作り、その後は KEHAI 起動時の「起動しました」だけ（28 回）。PC-A の KEHAI や公開 gateway の TTS も GSV を使うので、そちらを使う日は戻す。
  - GSV の声に戻す: `OMNI_TTS=gsv ./kehai.sh mode omni`（`move_gsv 0` で GPU0 に起こす）。cascade に戻すと `move_gsv 1` で GPU1 に起きる。
  - KEHAI 側: `service_context.sync_gsv_weights` は声が GSV のときだけ重みを送る（止まった GSV に毎ターン最大 25 s 待つ経路を塞いだ）。
    `start_server.sh` の GSV 起動確認は声が GSV のときだけ。
- omni では GestureLSM と ActionPlan の起動確認もしない（どちらも意図的に止めていて「DOWN — run ./kehai.sh up」と誤解を招いていた）。
- **YuNet（顔検出）は GPU を使っていない**（ONNX Runtime の CPU、`face_detect.py`）。視線追従に使用中。
- 今の GPU: GPU0 = 描画 Chrome 3.9 / Whisper 2.7 / Qwen3-TTS 5.7+1.4 / ffmpeg 0.6（計 13.8 GB）、GPU1 = Thinker 22.1 GB。

### 区切り・閾値以外で速くした（2026-11-04 09:30、ユーザー「このような修正以外でスピードを速める方法を模索して」）

計測道具: `scripts/omni_timeline.py --cam 写真 q1 q2 q3 --runs 2`（PROTO 経路で 1 ターンの各段を「最後の発話フレームから何 ms」で並べる。
ログの `@HH:MM:SS.mmm` 打刻: `[spec] started on the pause`、`[timing] llm request sent / llm headers / llm first token`、
`[timing] first sentence to TTS`、`[timing] tts request sent`、`[timing] first PCM`、`[TTFA] … endpoint@`、`[render] direct PCM: first chunk`）。
任意: `LOOP_LAG=1`（イベントループの遅れ p50/p95/max と、100 ms 超の停止時にメインスレッドのスタック）、`LLM_CHUNK_TIMING=1`（最初の 4 チャンクの到着）、
`LLM_REQ_DUMP=<dir>`（Thinker への要求をそのまま JSON に、外で再生して比べる）。

| PROTO 経路の中央値（ms） | 区切り | 要求 | 最初のトークン | 最初の文 | 最初の PCM | エンドポイント | ビューワ |
|---|---|---|---|---|---|---|---|
| この回の開始時 | 158 | – | 318 | 442 | 514 | 452 | **515** |
| 最終（SMART_TURN=0） | 176 | 188 | 305 | 365 | 451 | 449 | **453** |

効いたもの:
1. **Thinker の prefill を CUDA graph に**（`scripts/pcb-vllm/serve-qwen3-omni.sh` の `compilation_config.cudagraph_capture_sizes` を 512 まで、
   `OMNI_GRAPH_SIZES` で変更）。vLLM は graph の大きさを `min(2×max_num_seqs, 512)` = **4 トークン**で決めるので、5 トークン以上の prefill
   （今ターンの音声・時刻行・新しい発話）は graph なしで ~50 ms 余計にかかっていた。キャッシュ済み前置き＋新しい文字の最初のトークン 97 → 59〜65 ms。
   graph 0.29 GiB 増（GPU1 21.7 / 24.5 GB）。
2. **返答のたびに次のターンの前置きを先読み**（`BasicMemoryAgent.schedule_post_reply_warm`、`LLM_POST_REPLY_WARM=0` で無効）。
3. **Qwen3-TTS のチャンクを 1→2→4→6 フレームに**（`stack/qwen3-tts/qwen3_tts.deploy.yaml` の `codec_chunk_ramp: [1, 2, 4]`）。最初の音 44〜59 ms、
   再生の貯金は最小でも +28 ms で途切れない（ramp の末尾 4 と steady 6 の差の警告は出るが実害なし）。
4. **イベントループの停止**: 文区切りの言語判定（langdetect）を CJK/キリル/アラビア文字が無いときは飛ばす（1 文 3〜10 ms、初回 421 ms の停止）。
   ビューワのマイク用フィルタの初期状態を `lfilter_zi`（初回 401 ms 停止、しかも 0 倍していた）→ 0 配列に。
効かなかった・やめたもの（実測）:
- **音声を区切りごとの複数 item に分けて prefix cache**: 1 item あたりの固定費が大きく、伸びない（119/113 vs 102 ms）。
- **Thinker の最初のトークンの内訳**: 本当に初めての音声だと 105〜113 ms（文字だけなら 67〜76）。音声の分は GPU の encoder ~20 ms＋API 側の前処理 ~13 ms＋
  アップロード。vLLM の encoder 用 CUDA graph / compile は ViT・Qwen2.5-VL 向けで、Qwen3-Omni の音声 encoder は対象外（やるなら vllm-omni の改造）。
  「KEHAI 内で 40 ms 失っている」と一度見えたのは、再生で同じ音声を送り直して cache に当たっていたため（音声の末尾 2 バイトを変えて測り直して判明）。
- **Whisper と TTS の GPU0 競合**: なし（TTS の最初の音 49 vs 53 ms）。
- **Thinker の取り合い**: なし（vLLM の running は要求後 100〜150 ms で 1、待ちは 0、prefix cache 命中 98%）。
- **ストリーミング文字入力の TTS**（vLLM-Omni `/v1/audio/speech/stream`）: 文・節ごとに分けて要求するだけで、KEHAI の文単位と同じ。最初の文は短い（中央値 6 語）ので利得なし。
- **発話終了モデル（Pipecat Smart Turn v3.2、`src/kehai/vad/smart_turn.py`、`models/smart_turn/`、BSD-2）**: 区切りで P(完了) ≥ 0.8 なら
  エンドポイントを待たず解放。特徴量は NumPy 実装（transformers の WhisperFeatureExtractor と 4e-6 以内で一致）。**9 ターン×2 の A/B で逆に遅い**
  （ビューワ 505 vs 453 ms）: KEHAI のプロセス内で走ると、nice+10・1 スレッドでも Thinker のストリームを読む処理と取り合う。合成音声の試験文では
  完了文を「未完」と出すこともある（q2 0.07）。**既定 `SMART_TURN=0`**。やるなら別プロセスにして実音声で閾値を決める。
- 箱の観察: `whoopsie`（Ubuntu のクラッシュ報告デーモン）が CPU を 1 コア使い続けている（32 日稼働、要 sudo）。

### 「意味が分からない／英語じゃない」の正体（2026-11-04 08:15、ユーザー「性格ではなくバグ」）

音声は無罪: `logs/omni_audio/` の各ターンの wav を Whisper に通すと正しく聞き取れ、同じ wav を文脈なしで Thinker に投げると普通に答える
（"What time…" → 時刻、"How are you" → "Not much doing today"）。**文脈が原因**で、再現もできた: 同じ wav に下の形の履歴を付けると
3 回とも「I don't know who you're talking about. This conversation is pointless.」になる。

その形はこうしてできる（連鎖）:
1. 短い相づち（"That's great."）に Thinker が「Huh? That doesn't make sense. Try again.」と非回答を返す（7B 級の癖、ここは残る）。
2. **同じ返答の繰り返しを文脈から外す規則**（履歴読み込み時の dedupe と、今朝入れた会話中の guard）がその非回答を落とす → その user 発話が「未回答」になる。
3. `_add_message` は未回答の user 発話に次の user 発話を**改行でつなぐ**（cascade の「続けて話した」用）→ 文脈が
   `user: "Yeah, that's pretty good.\nThat's great."` / `assistant: "You're not making much sense."` という**二つ質問→意味不明と返す**見本になる。
4. omni では当ターンは音声だけなので、Thinker は見本どおり「意味が分からない」と返し続ける（自己強化）。

修正: 繰り返しの非回答を文脈から外すときは**その質問も一緒に外す**（`set_memory_from_history` と `_add_message` の両方）。
ペアごと消えるので交互の形が保たれ、つなぎも起きない。履歴ファイルには両方残る。
**もう一つの原因（08:24）: 早い解放が人の「間」に割り込む → 割り込みで止まる → 後半だけが新しいターンとして答えられる。**
エンドポイント 0.47 s だと、文中で 0.5 s 黙った人に彼女が答え始める。人は話し続け、割り込み（barge-in）で彼女は止まるが、
これまでは後半（"right now in Omaha?"、ASR では "Bye, now, now, now." のような断片）が**単独のターン**として答えられ、
文脈なしで「意味が分からない」が返っていた（ダンプの wav は Whisper でも断片）。
修正: 割り込み（`interrupt-signal`）の時刻を `interrupted_at` に残し、その後 `_TURN_MERGE_S` 以内のターンは**前のターンの続き**として
合流（文字も音声も前のターンの開始位置 `prev_turn_start` から切り出す）。合流ターンは fast release しない。
試験（`q2_split.wav` = "What time is it? [0.83 s の間] Right now in Omaha?"、via-viewer）: 合流ターンの音声は Whisper で
"What time is it? Right now in Omaha?…" と両方入り、返答は "It's 8:26 AM in Omaha."（修正前は "I'm not sure what you mean by that."）。
履歴の文字は後半の ASR 断片のまま（"What time is it? Bye, now, now, now."）— Thinker は音声で聞くので返答には影響しない。

調査道具: `OMNI_AUDIO_DUMP`（wav + 文脈末尾）、再生は HANDOFF のこの節のとおり（`prompts/personas/miku_en.txt` を system に、wav を audio_url で）。
（ダンプの音量が大きいのはプローブの `load_16k` が正規化しているため。サーバー側 `_wav_b64_16k` は clip とスケールだけ。）

### 話しかけられているかを omni が見る（2026-11-03、omni モードのみ）

- **何をするか**: `src/kehai/omni_addressee.py`。発話の始まり（onset）と区切りごと（segment）に PROTO のカメラ 1 枚
  （640×360 ≈ 245 トークン）を Thinker に見せ、「話者はロボットに向かって話しているか」を **0〜9 の 1 桁**で答えさせる
  （max_tokens 1、70〜150 ms、固定システムプロンプトは prefix cache）。**返答の経路には乗せない**: エンドポイントは
  届いている判定を読むだけで、飛行中なら最大 0.3 s 待つ。実測: 話し終わり→声 **0.60 / 0.61 / 0.67 s**（導入前 0.65 / 0.61 / 0.73）、
  返答生成の途中に 1 回割り込ませて生成速度 113 → 110 tok/s。
- **厳格**（ユーザー指定「PROTO に話しかけている人がいたら答える」）: 新しいフレームがあって点が `OMNI_ADDRESSEE_MIN`（5）未満なら
  「彼女宛てでない」として捨てる。**誰も映っていない（0）も捨てる**。カメラ無し・フレームが 1.5 s より古い・凍結（`render_stream` の判定）・
  :8808 失敗 → 判定なし → 従来どおり（YuNet の `ADDRESSEE`、こちらは既定 0 のまま）。カメラ無しのブラウザから話す分には影響しない。
- **割り込み**（`user-speech-start`）も判定を待ってから送る（上限 0.25 s）。他人と話している人の声では彼女は止まらない。
- **判定したフレーム**は `logs/addressee/<時刻>_s<点>_<onset|segment|endpoint>.jpg`（新しい 300 枚）。誤判定はここで見る。
  `OMNI_ADDRESSEE_DUMP=""` で止める。`OMNI_ADDRESSEE=0` で無効。起動 1 行目に `OMNI_ADDRESSEE=1(min 5)` が出る。
- **プロンプトの要点**: 「2 人以上が互いに向き合っている＝0」を明記しないと、向き合う 2 人を 7 と答えた。確認済み: 空の部屋 0、
  カメラに向かう人 9、向き合う 2 人 0、アニメ静止画 0。vLLM-Omni の Thinker は `logprobs` を返さない（500 "Did not output logprobs"）
  ので、確信度は 1 桁の点数で取る。
- **無人試験**: `scripts/omni_probe.py q.wav --cam picture.jpg [--expect-ignored]`。kind 5 のフレームを `mic-inject` に流す
  （注入中はビューワのカメラもマイクも脇に置かれる）。空の部屋と向き合う 2 人は `--expect-ignored` で PASS。
- **視線追従**（顔の位置、15 fps）は **YuNet のまま**。1 枚 0.1 s の Thinker では追従にならない。
- 同日の発見: 今日の起動は `ADDRESSEE=0`（YuNet の判定はオフ）だった。また omni の `ASR_MODEL_OVERRIDE`（turbo + greedy）は
  短い物音を「Thank you.」と書き起こし（67 ターン中 29）、Thinker がそれに "I don't understand what you're saying." と答えていた
  （cascade の日は 1〜6%）。conf.yaml が `large-v3-full` + beam 5 に固定していた理由そのもの。**未対処**（ユーザーの選択待ち:
  戻す／turbo のまま定型句を雑音扱い／sidecar に `no_speech_prob` を足す）。カメラ判定で誰も映っていない間は捨てられる。

- **2026-11-03 深夜、バグ確認で直した 3 つ**（ユーザー「バグが多そう」「話し始めが他の人の声」）:
  1. 判定は**描画タブのセッションだけ**に当てる（`websocket_handler._omni_judged`: `render_clients` か、試験の `mic-inject` 注入中）。
     判定がプロセス全体だったので、カメラ無しのノート PC のブラウザから話すと、PROTO の空の部屋を見て消されていた。
     `omni_probe.py` は素のクライアントだが、`--cam` の注入中は判定が当たる（注入フレームがカメラの代わり）。
  2. 発話の始まりでは**この発話の look** を最大 0.25 s 待つ（`omni_addressee.settle`、t0 以降の判定が無ければ飛行中を待つ）。
     直前の他人の 0 点（LEAD 1 s 以内）を流用して割り込みを送らず、ユーザーに被せて喋り続けていた。
  3. **相槌（`utils/backchannel.py`）を返答と同じストリーミング backend で作る**（`tts_manager.stream_backend`、GSV なら重み swap を待つ）。
     それまでは `openai_tts` → gateway `/v1/audio/speech`、つまり gateway 固定の参照クリップ（`refs/f02_025.wav`、日本語プロンプト）で
     GSV を回していたので、omni では Qwen3-TTS（aura）の返答の前に**別人の声の「Mm-hm.」**が 0.3 s 入っていた。cascade でも接続時の
     prime が重み swap を追い越し、前のキャラの声で焼かれることがあった（20:20:24 に生成、:26 に swap）。
     起動ログ `[backchannel] 3 clip(s) ready for <conf> via qwen3tts:...` で backend を確認できる。
- **ただし「話し始めが別人の声」の本当の原因は 3 ではなかった**（22:30、ユーザー「声がおかしい」で再調査）。ユーザーのターンでは相槌は
  送られていなかった（先読み HIT で音声が揃っていたため）。原因は **Qwen3-TTS の Aura 学習版そのもの**: 同じ 8 文を両方で合成して
  文ごとの中央 f0 を測ると Qwen3-TTS は **104〜237 Hz**（"Hello." が 104 Hz ＝ 男性の高さ、"Still here…" が 237 Hz）、GSV は 150〜183 Hz。
  短い最初の文で声が別人になる。→ `start_server.sh` の omni 既定を **`OMNI_TTS=gsv` に戻した**（学習済み speaker があっても）。
  話し終わり→声は 0.60 → 約 0.9〜1.2 s 帯に戻る。Qwen3-TTS に戻すのは `OMNI_TTS=qwen3tts`（声が安定する checkpoint ができてから）。
  比較の wav は scratchpad `voice/{qwen3,gsv}_<n>.wav`（セッション限り）。計測は同じ 8 文を `gsv_stream` / `qwen3tts_stream` に通して
  40 ms 窓の自己相関で f0 を取るだけ（スクリプトは HANDOFF のこの節の日付のセッション記録にある）。
- 直していない（挙動として知っておく）: `QWEN3TTS_SPEAKER` は env なので omni で Qwen3-TTS を使うと**全キャラが aura の声**。
  YuNet の `ADDRESSEE`（既定 0）もプロセス全体の判定のまま。
## 触ってはいけない / 間違えやすい

1. **`bash start_server.sh` 以外で起動しない**。素の `run_server.py` は env が入らず、
   LLM だけ動いて TTS/STT/gesture が全滅する。チャットが動いて見えるので誤診しやすい。
   起動1行目が `GESTURE_ENABLED=1 MOTION_GATEWAY_KEY=<set> BASE=http://127.0.0.1:3000/v1`
   であることを必ず確認。
2. **`conf.yaml` と `start_server.sh` を PC-A と同期しない**。この2つは意図的に分岐している
   （:3000 loopback / `enable_https: false` / tunnel supervisor 削除 / `.venv/bin/python`）。
3. **`pkill -f run_server.py` は自分を殺す**。ssh 経由だとコマンドライン自体に文字列が
   含まれるため。`pkill -f '[r]un_server.py'` を使うか `./kehai.sh stop`。
4. **`enable_https: false` を戻さない**。`clean_cache()` が終了時に `cache/` を丸ごと消すので
   https にすると証明書が毎起動再生成され、毎回警告を踏む（自動化ブラウザは開けない）。
   トンネルが SSH で暗号化済み、`127.0.0.1` は http でも secure context なのでマイクは使える。
5. **モーションの 202 は正常なことがある**。クリップは消費・破棄されるので、ターン終了後に
   取りに行くと健全でも 202/404 になる。id が来た瞬間に取ること（`e2e_probe.py` がそうしている）。
6. **`requirements.txt` は不完全**。`aiortc` / `av` / `silero-vad` は宣言が無いのに起動時に
   import される。環境を作り直すときは手で足す（`pyproject.toml` に入れてしまうのが本筋）。
7. **音声の ref パスは `~/Documents/DEV/AICCORE-Nexus/stack/voice/refs/`**。
   旧 `~/Documents/DEV/voice/` は存在しない。GSV は `streaming_mode=true` だと
   参照ファイル欠損で **HTTP 200 + 0バイト**を返し、ログに何も出さずに無音になる。
   切り分けは `streaming_mode=false` で叩くと 400 + `... not exists` が見える。

## 周辺サービス（別プロセス、KEHAI の外）

`~/Documents/DEV/AICCORE-Nexus` — gateway(:3000) / GSV(:9880) / vLLM(:8807) /
ActionPlan(:8893) / GestureLSM(:8802)。ActionPlan は On-Demand ではないので
再起動後は手動起動が要る（`start_server.sh` の warmup がその旨を出力する）。

### 箱の再起動後（2026-10-21 に踏んだ）

crontab の `@reboot` が `./kehai.sh up`（= `scripts/pcb-stack-up.sh`）を走らせ、
gateway → ActionPlan → vLLM → KEHAI → 描画ホスト（`scripts/render-host.sh start`、2026-11-12 に追加。
それまでは自動で起きず、PROTO がつながっても絵も声も出なかった）の順に起こす（結果は `logs/boot.log`）。
- **大学側の IP が変わった日は HTTPS の証明書が作り直される**（`ssl_utils`、ログ `SSL certificate does not cover …
  — regenerating`。2026-11-12 は 10.66.182.130 → 10.66.134.90）。PROTO の Chrome が前に受け入れた証明書とは別物になるので、
  PROTO で `https://192.168.50.1:12443/render-view` を開くと警告画面で止まる。「詳細設定 → 192.168.50.1 にアクセスする」で進み、1 回タップする。
  こちらからは PROTO に SSH できない（鍵が無い）ので、PROTO 側の操作は人が行う。TTS/gesture は
gateway に任せる。手で戻すときも `./kehai.sh up` だけでよい（上がっているものは触らない）。
自前で起動するときの罠:

1. vLLM は **venv の bin を PATH に入れること**。無いと flashinfer の JIT が `ninja` を見つけられず
   エンジン初期化で落ちる。
2. ActionPlan は **GPU1**（`CUDA_VISIBLE_DEVICES=1`）。省くと GPU0 に載り、GPU0 固定の
   gesturelsm が gateway の VRAM スケジューラに 507 `gpu_full` で弾かれる。
3. TTS / gesture は gateway の DB（`prisma/dev.db` の `ModelRuntimeConfig`）で `enabled=1` で
   なければならない。`enabled=0` だと起動されないうえ、手で起こしても worker が kill する
   （GSV が出力ゼロで即死するのはこれ）。

### 隣の音源を拾う問題 → 名前で会話を始める（2026-11-09 04:25、ユーザーの選択）

- **症状**: 人が PROTO の前に立つと、周りの音をすべてその人の発話として答えた。隣のロボットや TV の声（「Welcome to 20 Minutes of Telltale…」「Thank you for joining us today…」）が、ユーザーの発話につながったり、ユーザーが黙っていても答えたりした。
- **試したが分けられなかったもの**:
  - 音量: 隣の音源のピークは 70〜75 dB で、人の声と同じ。自動音量調整を切っても 3〜5 dB 下がるだけだった。
  - 声の特徴（`speaker_probe.py`、記録だけ）: 公開の話者モデル 2 種（WeSpeaker、CAM++、VoxCeleb で学習）。別の音源どうしの類似度は WeSpeaker で 0.68〜0.89 と、全部似た値になった。CAM++ は 0.16〜0.98 とばらついた。
  - 唇: 上の節のとおり、PROTO のカメラ画像では判定できなかった。
- **入れたもの**:
  1. **名前で会話を始める**（`WAKE_NAME=1`、`websocket_handler._session_gate`）。今のカメラの規則（2.0 m 以内に立ち止まってこちらを向いている）に通ったうえで、Whisper か Thinker の書き取りに名前（Iris。書き取りの崩れ方も含めた正規表現 `_NAME_RE`）があれば会話を開く。開いている間は、名前なしで続けられる。
     - 会話が終わる条件は 3 つ。
       - 彼女の返事のあと `WAKE_FOLLOW_S`（10 s）以内に、彼女宛ての発話がない。
       - 3 s 以上、2.0 m 以内に誰もいない（`Attention.left_between`）。
       - bye / that's all などの別れの言葉。この場合は、その発話には答えてから閉じる。
     - 名前のない発話: fast release の時点では Thinker の書き取りだけを見る。名前がなくてもすぐには捨てず、Whisper の文字を待つ通常の経路で、両方の書き取りを見て判定する。
     - 前に人がいて名前なしで話したときは、案内の一言（"If you are talking to me, please start with my name, Iris."）を 60 s に 1 回まで流す（`_maybe_invite`、相槌と同じ経路で、返事と同じ声）。
     - ログ: `[wake] fast release held: …`、`[addressee] not to her (…; her name was not said, and no conversation is open)`、`[wake] invited …`。
  2. **発話を、話し始めた人の音量に合わせて区切る**（`silero._sustain`、`VAD_REL_DROP_DB`、既定 15 dB）。発話の最初の 0.6 s の音量（75 パーセンタイル）より 15 dB 以上小さい声は、その発話を続けることも再開することもない。ログは `[vad] N ms of speech more than 15 dB under the turn's own level …`。これまでは、45 dB 以上の声が鳴り続けると発話が終わらず、TV の声がつながっていた。
  3. **ビューワのマイクの自動音量調整を切った**（`mic_tuning.json` の `auto_gain: false`）。マイクを開く処理が 2 つ同時に走っていて、先に開いた方（自動音量調整あり）の設定が残っていた。開く処理を 1 回ずつ順番に走らせ、古いマイクを閉じてから新しいマイクを開くように直した（`render_view.html` の `openMic`）。`[render] viewer mic:` の行に、頼んだ設定（`asked`）と実際の制約（`constraints`）を出す。
- **残る弱点**: 会話が開いている間（彼女が話している最中と、返事のあと 10 s）は、隣の声も通る。名前の書き取りは、PROTO の 4 kHz のマイクでは崩れることがある（その場合は、もう一度名前を呼んでもらう）。

### 名前で始める方式はやめた → 話しかけ判定を強める（2026-11-09 04:40、ユーザー「あまりよくなかった」）

- **名前（Iris）は PROTO のマイクで書き取れなかった**: 「Iris, …」が「I didn't」「I do」「I know」「I read」になり、9 回話して 1 回も会話が開かなかった。音声を Thinker に直接聞かせて「Iris と言っているか」を答えさせても 9 回中 1 回。`WAKE_NAME` の既定を 0 にした（コードは残してある）。
- 自動音量調整を切ると、声が小さく入って Whisper が何でも「Thank you.」と書いたので、`auto_gain: true` に戻した。
- **唇の値は、長い発話なら差がある（これまでのログの集計）**: 映像だけの判定（`lip_motion`）の値は次のとおり。
  - ユーザーが文を話した回（02:53〜03:33、約 45 件）: おおむね −0.5〜+0.4、正のコマ 15〜60 %。
  - ユーザーが黙って前に立ち、隣のロボットだけが話した回（04:16〜04:17、6 件）: −0.70〜−1.37、正のコマ 0〜10 %。
  - ただし 1 秒ほどの短い発話（04:27、15 件）は −0.24〜−1.09、正のコマ 0〜7 % で、黙っている回と重なる。
  - まだ判定には使っていない。`[lip]` の行に、声が大きいコマだけの値（`voiced`）と、同じ顔の発話前 3 s との差（`rise`）を足した。ラベルのはっきりした試験でこの 2 つが分かれるかを見てから使う。
- **こちらを向いている割合の条件を上げた**（`ADDRESSEE_ENGAGED_FACING`、5 % → 25 %）。前に立って友人や隣のロボットの方を向いている人を落とすため。今日、正しく答えた 48 回は 33〜100 %（あいさつ 1 回だけ 20 %）。

### 話しかけ判定の新案 A・B・C の結果（2026-11-09 14:15）

公開のデモ動画（TalkNet demo 001、話し手と聞き手が写っている）で先に試した。道具は scratchpad の `abc/`。

- **A（顔の連続写真 8 枚＋音声を Thinker に渡し、「この顔が話しているか」を 0〜9 で答えさせる）: 使えない。** 話し手は 10 区間中 8 回が 9 点、聞き手も 10 区間中 5〜9 回が 9 点だった。写真だけを渡すと、話し手も 10 区間中 3〜4 回しか当たらない。Thinker は、コマの間の口の動きを見ていない。写真 8 枚を 1 枚に並べても同じだった。
- **B（口の動きと声の大きさの連動）: 使えない。**
  - 口の領域の画素の変化と声の大きさの相関: 話し手 +0.16、聞き手 +0.02〜+0.14 で、区間ごとの勝ち負けは半々だった。
  - 顔の特徴点（MediaPipe Face Landmarker）で測った口の開きの揺れ: 話し手と聞き手で同じ大きさだった（聞き手も笑ったり口を開けたりする）。切り出した顔では検出の失敗も多かった（499 コマ中 170 コマ）。
  - 同じ動画で差が出るのは、これまでの LR-ASD の映像専用の値だけ（話し手 +0.5〜+1.7、聞き手 −0.2〜−0.9）。映像の候補はこれに絞る（`[lip]` の `voiced` と `rise`、記録だけ）。
- **C（会話のタイミング、`turn_contingency.py`、`TURN_CONTINGENCY=1`、記録だけ）**: 発話ごとに `[contingency]` の行を出す。
  - `gap`: 彼女の声が終わってから、この声が始まるまでの秒数。
  - `voice over her`: 彼女が話していた時間（直近 20 s）のうち、声がかぶっていた割合。
  - `voice in the 6 s before`: この声が始まる前の 6 s（彼女の声は除く）に、すでに声があった割合。
  - `barge`: この発話のうち、彼女が話している最中だった割合。
  - 無人で隣の音源だけのとき、`voice in the 6 s before` は 42〜61 % だった。静かなところから話し始める人なら低くなるはずで、ラベル付きの試験で確かめる。

### 音の方向は使えない（マイクは 1 本）・背景の声の判定を入れた（2026-11-09 15:10）

- **PROTO のマイクは 1 本**: ビューワのページから、全入力をブラウザの処理なし・2 チャンネルで 4 s ずつ録った（`POST /render-stream/mic-probe` → kind 7 → `logs/mic_probe/`、解析後に削除）。
  - 「Default」と「HD Audio Analog Stereo」は、左右がサンプル単位で同じだった（相関 1.0000、差 −180 dB）。音の来る方向は計算できない。
  - 「HDMI Input Video Analog Stereo」は無音。
  - 処理なしの音量は −77〜−96 dBFS と非常に小さい。自動音量調整を切ると声が小さくなるのは、このため。
  - 録ったときは部屋が静かで、4 kHz の切れ方は判定できなかった。
- **マイクが生きているかの確認**: `curl http://127.0.0.1:12394/render-stream/mic-level`（VAD が今聞いている音量と状態、loopback のみ）。静かな部屋では、ノイズ除去のため 0.0 dB と出る。
- **背景の声の判定（`websocket_handler._bg_talker`、`BG_TALKER_DROP=1`）**: カメラの規則が通した発話を、次の 3 つがそろったときだけ落とす。
  1. その声が始まる前の 6 s の 40 % 以上に、すでに声があった（`BG_BEFORE`）。
  2. その声の音量が、前から鳴っていた声より 2 dB を超えて大きくない（`BG_STEP_DB`、`turn_contingency` の `step`）。
  3. 前の顔の唇が、声の出ているコマ 25 コマ（1 s）以上で「話していない」（唇の値 −0.6 以下、正のコマ 10 % 以下、`BG_LIP`）。
  - 静かなときに話した人は 1 で通る。鳴っている声にかぶせて大きく話した人は 2 で通る。短い発話は 3 の「1 s 以上」に届かないので落とさない。
  - 唇の計算（0.1〜0.5 s）は、1 と 2 が成り立ったときだけ走る。
  - ログ: `[background] DROP|keep: …`。`BG_TALKER_DROP=0` で記録だけになる。
  - **隣の音源が鳴っている状態での実機試験はまだ**（入れたとき部屋が静かだった）。3 つの閾値は、今日のログの値（黙って立った 6 回、文を話した約 45 回）から決めた仮の値。
- 2026-11-09 15:20 マイク感度: 今日のログでは、答えた発話（67 回）と無人時の背景の声（1185 回）のピーク音量がほぼ同じ（中央値 73.5 / 72.1 dB）。話し始めの関門を 68→71 dB にすると、ユーザーの 93 %・背景の 85 % が残る（73 dB では 73 % / 59 %）。音量では分けられない。`mic_tuning.json` を onset 71・sustain 55 にした。現場での調整は、ビューワの「マイク設定」パネル。

### 小さい PROTO（Proto M2）に替えた・話し始めの関門を部屋に合わせる（2026-11-09 18:10）

- 17:25 から PROTO は Proto M2（小型、持ち運んで置き場所を変える）。マイクは「ProtoCam1」で**全帯域**（30 発話中 29 が `full-band`、返事は書き取りを待たない）。カメラは 1920×1080 30 fps。
- **距離は未較正**: 話しかけた人の推定が 0.2〜0.5 m、顔 270〜430 px。画角が旧 PROTO（100° の仮定）より狭いか、実際に近い。実距離を 1 回測って `FACE_HFOV_DEG` を決める必要がある（ユーザーの回答待ち）。
- **話し始めの関門を部屋の声に合わせる**（`turn_contingency.room_gate`、`VAD_ROOM_GATE=1`、`silero._base_gate`）: 前に誰もいない（カメラが動いていて 3 s 以上 `MAX_M` 以内に顔が無い、`Attention.empty_for`）・彼女が黙っている間に聞こえた声を「部屋の声」とし、直近 10 分の 95 パーセンタイル + 1.5 dB（`VAD_ROOM_MARGIN_DB`）を関門にする。範囲は 66〜72 dB（`VAD_ROOM_MIN_DB` / `VAD_ROOM_MAX_DB`）、部屋の声が 3 s 分たまるまでは 66。騒音の床による引き上げ（`VAD_ADAPT_*`）はその上に従来どおり乗る。**マイクパネルの onset 値は、これが有効な間は使われない**（`VAD_ROOM_GATE=0` で従来）。ログ `[vad] room gate: onset N dB (...)`。
  - 実測（隣で発表の音声が流れている部屋）: 起動 20 s で 66 → 72 dB（部屋の声 76.1 dB）、以後その音声は 65〜72 dB で関門の下に落ちた。
  - 弱点: 部屋の声が 72 dB を超える場所では上限で止まる。来訪者の声の山は 72〜77 dB なので、小声の人は関門の下に落ちうる（音量だけでは分けられない、は変わらず）。静かな部屋での確認はまだ。

### Proto M2 のマイクは 2 本ある → 声の方向が取れる（2026-11-09 19:50、記録だけ）

- 「ProtoCam1 Analog Stereo」を処理なし・ステレオで開くと、左右が別の音になる（相関 0.67〜0.87。旧 PROTO は完全に同じ = 1 本）。帯域も全帯域。
- 左右の到着時刻の差（GCC-PHAT、300〜6000 Hz、48 kHz のサンプル数）: 正面で話した大きい声は **+0.2〜+2**、部屋の別の声は **+6〜+8** や **−9〜−13**。最大は約 ±12〜14（マイクの間隔は 9〜10 cm 相当）。
- **`src/kehai/voice_direction.py`（`VOICE_DIRECTION=1`、記録だけ）**: ビューワが通常の取り込み（「Default」、処理あり・モノラル）の横で、同じマイクを**別のデバイス項目**（ProtoCam1）として処理なし・ステレオで開き、kind 8（4096 コマごとの int16 ステレオ）で送る（`render_view.html` の `openDirMic`、`?dir=0` で無効）。2 つの取り込みは設定が混ざらない（`[render] viewer stereo mic for direction:` の行に両方の実際の設定が出る。同じデバイス項目で 2 つ開くと混ざる、は 11-09 04:14 の件）。
- 発話ごとに `[direction] N% of M voiced blocks in front (|lag| <= 4), … median lag …, lags […]`。「正面」の幅 `VOICE_FRONT_LAG`（4）は仮の値。
- 最初の実例（前に誰もいない部屋の声）: 正面 10 %、中央値 −10.6。
- まだ判定には使っていない。正面で話す人・前に立って黙る人のラベル付き試験で幅を決めてから、(1) 宛先の判定、(2) 発話の区切り（横からの声は続きとみなさない）、(3) 割り込み、に使う。
- 限界: 正面の人の真後ろの声は同じ方向になる。Iris 自身の声（PROTO のスピーカー）がどの方向に出るかは未測定。
- **19:47 方向を判定に使い始めた**（`DIRECTION_DECIDES=1`、`websocket_handler._bg_talker` の先頭）。カメラの規則が通した発話について、Iris 自身の声の区間を除いた有声ブロックの「正面（|lag| ≤ 4）から来た割合（音量で重み付け）」が 30 % 以下（`DIRECTION_SIDE`）なら落とし、60 % 以上（`DIRECTION_FRONT`）なら唇の判定を通さず答える。間や方向が取れないときは従来の「前から鳴っていた声＋唇」。ログ `[background] DROP|keep: … of the voice came from in front`。
  - 根拠（19:44〜19:46、ユーザーが正面 0.4〜0.5 m で 10 発話）: 正面の割合 92〜100 %、中央値 −0.2〜+1.9、−25〜−33 dBFS。部屋の声は 0〜13 %（中央値 −10.6、+13.6）。同じ時間に、唇の規則は正面からの質問を 1 回誤って落とし（lips −0.82）、カメラの規則は横の音（"Chuvelo"、+13.6）に答えていた。
  - 1 人・1 部屋・1 回の配置での値。別の人・別の置き場所では未確認。発話の区切りと割り込みにはまだ使っていない。
- **19:52 Proto M2 で目線が上を向きすぎる**（ユーザー「目線が合わない」）: 描画タブの `headCamOffsetY`（カメラがパネル中心より何 m 上か）が大きい PROTO 用の 0.95 のままで、0.4〜0.5 m に立つ人の頭が実際より約 0.7 m 高い位置に計算され、目が上限（`lookAtMaxPitch` 22°）に張り付いていた。描画タブで `__space({headCamOffsetY: 0.27})`（21.5 型の縦パネルの半分 + 縁、推定値）＋ `localStorage.vrmSpace1` に保存。大きい PROTO に戻すときは 0.95。場面は 1.9 m のパネル前提のまま（`panelHeightM` は触っていない）。
- **19:56 「だいぶ離れても反応する」**: 19:54〜19:55 にユーザーが下がりながら同じ文を 3 回話し、推定 0.5 / 0.9 / 1.2 m の全部に答えていた（生の音量は −34 / −35 / −39 dBFS で 5 dB しか違わず、音量では距離を切れない）。`start_server.sh` で `ADDRESSEE_NEAR_M` を 2.0 → **0.8**（推定値の尺度、M2 用。大きい PROTO に戻すなら 2.0）。
- **20:00 返事が 1.8〜2.2 s 遅れる回の原因**: 夕方の 20 ターンのうち普通の回は「話し終わり判定→最初の PCM」12〜145 ms。遅い 4 回は全部、完結した質問を「文が途中」と判定して 1.5 s 待っていた（'So can you guide for me?'、'Okay, uh, can you talk about AI ASICs?' など、非ネイティブの語順・言い淀み）。自作の試験文で測ると、学習者英語の完結文 30 件中 9 件を「途中」と判定、ネイティブの完結文は 0/30。`omni_addressee._COMPLETE_SYSTEM` に「語順の誤り・冠詞や動詞の欠落・途中の uh/um は途中ではない、終わり方だけを見る」と手本 8 個を追加 → 学習者 0/30、ネイティブ 0/30、実際の 4 件 0/4、途中で切った文の検出は 40/52 → 39/52（見逃しは 'Tell me' のように文として成り立つもの）、実際に途中だった 2 件は 2/2。試験は scratchpad `complete/ev.py`（セッション限り、試験文は手本と別）。
- **20:03 途中で止まった文の待ちを短く**（`websocket_handler._hold_wait`、`VAD_UNFINISHED_QUIET_S`=1.0）: 「文が途中」の 1.5 s の待ちは固定の sleep だった（書き取りが届いてから 1.5 s = 最後の言葉から約 2.1 s）。ステレオの流れが生きていれば、**正面からの声が 1.0 s 途絶えた時点**で待ちを終える（`voice_direction.last_front_voice`、Iris 自身の声は除く。正面の声が再開すれば、関門を越える前でも待ちが続く）。見込みは最後の言葉から約 1.0〜1.1 s。ログ `[server-vad] hold ended early: no voice from in front for …`。ステレオが無いとき（大きい PROTO）は従来の固定待ち。`VAD_UNFINISHED_QUIET_S=0` で従来。
  - **実機での確認はまだ**（無人試験の mic-inject はステレオを流せない）。文の途中で 1 s 以上黙る人には、前半に答え始めてから割り込みの合流で言い直す（従来は 1.5 s まで待てた）。
- **20:10 方向の判定の不具合を 3 つ直した**（コード点検と 20:02〜20:05 のログ）:
  1. 騒がしい部屋では判定が出なかった。「有声」を「部屋の床 + 6 dB 以上」で選んでいて、床が −43 dBFS に上がると −38 dBFS の来訪者のブロックが 1 つも残らず（ログは「no stereo blocks」）、唇の規則に戻っていた。→ その発話の最大音量から 12 dB 以内のブロックを使う。判定なしのときは理由をログに出す。
  2. 少し横に立った人を「横の声」として落とす恐れ（0.45 m で 20 cm ずれると約 5 サンプル、正面の幅は 4）。→ 「横の声」で落とすのは、一番近い顔が画面の中央（±0.15、`DIRECTION_FACE_CENTRE`）にいるときだけ。`Attention.face_x` を足し、`[direction]` の行に `face x` を出す（顔の位置と時間差の対応を後で較正するため）。
  3. マイク 1 本を 2 チャンネルで出す入力（大きい PROTO の「HD Audio Analog Stereo」）だと、全部の音が時間差 0 =「正面」になる。→ 左右が同じ信号のブロックが続く間は判定を出さない（`voice_direction._same`）。
  - どれも実機での確認はまだ。再起動直後の統計の `audio 0.0 kS/s in` は不具合ではない（描画タブ→ビューワの音は、再起動後に彼女が最初に話すまで流れない。声は直送）。
- **20:20 会話の中身の点検で見つけた 2 つ**（19:37〜19:55 の履歴）:
  1. **同じ自己紹介ばかり返す**: 「How are you?」「Thank you.」「Can you make a guide for me?」に全部「(Hello.) I am Iris, the android guide for this maker space.」。作り直しの関門（`[askback-guard] … repeating an earlier reply — asked again`）は働いたが、作り直しても同じ文だった。原因は 2 つ: (a) 人格の手本にこの文がそのままあり、7B 級は「手本の言い回しを使うな」を守らず写す、(b) 履歴が 11-04 から 1 本で続いていて（660 件、彼女の返事 277 件中この文 14 件・「Hello. I am functioning optimally.」16 件）、文脈がお手本になっていた。同じ質問を履歴なしで投げると普通に答える。
     - 人格: 手本を「写されても困らない文」に替え（あいさつ・自分の説明・崩れた英語の料理の質問）、「自己紹介は聞かれたときだけ」を追加。履歴なし 16 問×3 回で自己紹介 9/48 → 3/48。
     - **新しい来訪者には新しい会話**（`websocket_handler._maybe_new_visitor`、`VISITOR_RESET=1`）: 彼女が最後に受けたターンのあと、前（`NEAR_M` 以内）が `VISITOR_GAP_S`（20 s）以上空いてから誰かが来たら、次の発話の始まりで新しい履歴を作って記憶を読み直す（古い履歴はファイルに残る）。起動後の最初の声でも 1 回行う（再開した履歴は別の来訪者のもの）。ログ `[visitor] … — new conversation`。**実機で発火を確認する前に記録した。**
  2. **知識の質問を断る**（「How to make a hamburger」→「I am unable to provide detailed instructions…」）: 11-09 03:40 に足した「できることは話す・時刻・検索だけ」を、話題の制限と読んでいた。「どんな話題でも知っていることを答える。できないのは操作（アラーム等）だけ」に書き直した。履歴なしの試験では断り 0/48（直す前も 0/48 だったので、効果は実機の会話で確認が要る）。
- **20:25 無人試験で見つけた分**（`omni_probe.py --via-viewer --cam`、前に誰もいないことを確認して実行、試験の履歴は削除）:
  1. 方向の判定が**雑音から「横の声」を作って落とした**: 試験の音は mic-inject で入るのに、ステレオは本物の部屋の音のまま流れていて、床と同じ音量（−44 dBFS）のブロックの時間差 −8.6 で「横」と判定。→ その発話の最大音量が部屋の床 + 6 dB に届かないときは判定を出さない。注入中はステレオを取り込まない（`render_stream`）。
  2. 返事の末尾の決まり文句（"Let me know if there's anything else you'd like to explore!"）: 人格で禁じても 7B 級は付ける。20 s の返事の最後の 4 s がこれだった。→ `transformers.sentence_divider` で、2 文目以降のこの種の文は声に出さない（`REPLY_CLOSER_FILTER=1`、`_CLOSER_RE`、ログ `[closer] not spoken`）。「I can look it up, if you like.」のような中身のある申し出は通す。
  3. 確認できたもの: `[visitor] first voice since this session began — new conversation`（発火）、新しい人格の読み込み、話し終わり判定→最初の PCM 73〜214 ms。
  - 試験の 2 本目の質問（彼女が 20 s 話している最中に流れた）は VAD のログに何も残らなかった。プローブ側の流し方の問題か、話している最中の取り込みの問題かは未調査（実際の人の割り込みは 19:45 に働いている）。
- **20:50 上の「未調査」2 件の原因**:
  1. **試験の 2 本目の質問が消えた件 = Silero VAD の内部状態**（話している最中の取り込みの問題ではなかった）。VAD のモデルは内部状態を前のコマから引き継ぎ、起動から一度もクリアしていなかった。「長い静かな時間 → 1 発話 → 約 10 s のデジタル無音 → 弱い録音（q12）」の並びで、q12 の確率が 0.16〜0.34（必要 0.4）にしかならず、発話として開かない。同じ並びをオフラインで再現（部屋のハム 60 s を前に置くと再現、20 s では 0.7）、実クラス `VADEngine` でも再現。部屋で動画が鳴り始めた 20:25 以降は起きなくなった（前の音が違う）。
     - 直し: 待機中に確率 0.1 未満が 2 s 続いたらモデルの状態をクリア（`silero.VADEngine._STATE_RESET_S`、`VAD_STATE_RESET_S`、0 で無効）。同じ並びで q12 が開く（0.99）。副作用の確認: 40 録音の 160 組・連続 60 組の模擬で、クリアあり・なしとも取りこぼし 0。
     - 頻度は低い（上の模擬では特定の組だけ）。実際の来訪者で起きていたかは分からない。
     - 調査の副産物: `mic_tuning.MONITOR` は全クライアント共通。WebRTC のマイク経路（`on_mic_chunk`）に VAD 直結の二重投入を防ぐ関門が無かったので足した（今は描画タブに WebRTC が無く、実害は出ていない）。
     - 調べ方: `logs/asr_debug/ENABLED` を置くと VAD に入った音が `tab.rec` に残る（来訪者の声が入るので終わったら消す）。
  2. **返事が長い件 = まれに止まらなくなる**: 本番のシステムプロンプト全文で 12 問×3 回を測ると中央値 24 語・3 文だが、「Tell me about Japan.」が 30 文 272 語（約 2 分）まで続いた例があった。人格の「1〜3 文」は大半で守られ、たまに外れる。→ 声に出した語数が 70 を超えたら、次の文からは話さない（`transformers._MAX_WORDS`、`REPLY_MAX_WORDS`、ログ `[length] reply cut after N words`）。4〜5 文の物語や説明（約 50 語）は通る。長い説明を頼まれた場合も 70 語あたりで止まる。
- **23:15 点検で見つけた分**（20:45〜23:07 のログ = 前に誰もいない部屋の声 24 発話、全部正しく無視。その中の記録とコードから）:
  1. **部屋の声の「言いかけ」が来訪者の発話として保留されていた**（22:48:23 'Zakir na.'、前に誰もいない）。「文が途中」の待ち（`pending_vad_transcript`）が宛先の判定より先だったため。保留中は次の物音が「同じ人の続き」（`_is_continuation`）になり、(a) その始まりで彼女に停止の合図が出る（誰かに答えている最中でも止まる）、(b) 言いかけの文字と音声が次の本物の質問につながる。→ 保留の前に宛先（カメラの規則＋方向／背景の声）を決め、彼女宛てでなければ保留せずに捨てる（`_finalize_vad_turn` の `_pre`、ログ `— ignored, not held`）。判定は保留のあとで使い回す。
     - 無人試験（`paused09/12.wav` = "Can you tell me about" ＋間＋ "the South Market, please."）: 前に誰もいない絵 → 前半を保留せず捨て、後半の始まりは「not to her」。近くでこちらを向いた人の絵 → 保留 → つないで 1 回答える（話し終わり判定→最初の PCM 444 ms）。
     - **残る弱点**: 来訪者の前半を保留している 1.5 s の間に横の声が始まると、後半として判定され、方向が「横」なら前半ごと捨てる（来訪者は言い直しが要る）。前半だけを切り出して答える処理は未実装。
  2. **待ちの早期終了が騒がしい部屋で効かなかった**: 部屋の声でも、ブロック単位では 10〜30 % が「正面」と読める（上の 24 発話の `[direction]`）。1 ブロックでも正面なら「正面の声が続いている」としていたので、1.0 s の静けさがほぼ成立しなかった。→ 正面の声は「2 ブロック連続、かつその発話の最大音量から 15 dB 以内」（`voice_direction.last_front_voice(t_on, t_end)`）。実機での確認はまだ。
  3. **新しい来訪者の判定が、会話の途中の顔の見失いでも成立しえた**（近づきすぎてカメラの下に外れる・下を向く・検出の抜けが 20 s 続いてから顔が戻る）。→ 彼女の最後の返事の終わり（と最後に受けたターン）から `VISITOR_GAP_S`（20 s）以上たっていることも条件にした。今の履歴にまだ何も話していなければ、新しい履歴ファイルは作らない（通りすがりごとに空のファイルが増えない）。
  4. **ビューワの接続直後の例外**（20:45:53 `Exception in ASGI application`）: サーバー起動中にビューワが 4 s の見張りで切断すると、最初の送信が例外で抜け、後始末（`hub.viewer` のクリア）を通らなかった。→ 最初の送信を try で囲み、ログ `[render] viewer left during the hello` を出して片付ける。
  5. **参照されないタスク**: `websocket_handler` の `asyncio.create_task(...)` 9 か所（発話の確定 `_finalize_and_settle`、先読み、割り込みの確認など）は戻り値を誰も持っていなかった。asyncio はタスクを弱参照でしか持たないので、待っている間に回収されるとそのターンが無言で消える（起きた記録は無い）。→ `_spawn()`（集合で保持）に替えた。
  6. 確認できたもの: `[length] reply cut after 72 words`（South Market の説明が 4 文 72 語・25.8 s、5 文目から打ち切り）。4 文で 26 s は長い — 上限 70 語は「止まらない返事」の保険で、ふだんの長さは人格の「1〜3 文」頼みのまま。
  7. 記録だけ追加: 返事が終わるたびに `[direction] her own voice — …`（彼女自身の声が 2 本のマイクでどの時間差に出るか）。割り込みの声を方向で見分けられるかの材料。**割り込み（彼女が話している最中の声）はまだカメラの規則で決まる**ので、前に人が立っていると横の声でも止まる。
  - 罠: ログを時刻で絞るときは日付も付ける（`awk '$1=="2026-11-09" && $2>=t'`）。時刻だけだと 1 か月分の同じ時刻が全部出る。
- **23:45 返事がときどき 1.2〜2.2 s 遅れる原因 = Thinker を KEHAI 以外が使っていた**（23:20:50 / 23:25:21 / 23:25:45 / 23:30:12 に `[omni] heard: transcription failed (TimeoutError())`・`omni look failed: ReadTimeout`・`llm first token +1523 ms`）。
  - Thinker のログ（`AICCORE-Nexus/stack/vllm-omni/server.log` の `[OmniTiming]`）には、KEHAI が何もしていない時間にも、2 本並行の `/tokenize` → `/v1/chat/completions`（1 本 0.3〜4 s）が続いていた。Thinker は `max_num_seqs: 2` なので、KEHAI の要求（返事・書き取り・カメラの look）がその後ろで待つ。
  - 送り主: `:8808` につながっているクライアント側のソケットは、読めるプロセスの fd に無い（= sshd のポート転送）。4 つある sshd のうち CPU 時間が要求と一緒に進むのは **22:47:54 に 10.144.5.20 から ai-ccore でログインした session 13269**（`loginctl show-session 13269 -p RemoteHost`）。VS Code の 3 本（10.144.2.103）ではない。止めていない（誰の作業か未確認）。
  - 調べ方: `ss -tnpe | grep :8808`（`users:` が出ない行が転送）、`/proc/<sshd pid>/stat` の utime+stime を 25 s おいて比べる。`nvidia-smi` のプロセス一覧には出ない（GPU を使うのは Thinker 自身）。
  - 罠: この箱の `ss` に `--cgroup` は無い（黙って何も出さない見張りを作った）。`pkill -f <自分のコマンドに含まれる語>` でまた自分のシェルを殺した。

### 再起動後「話しかけても答えない」（2026-11-12 23:20〜23:26）

PROTO がつながったあと、前に立って「Hi, how are you?」と何度話しかけても答えなかった。原因は 3 つ:
1. **取り消したはずの返事のタスクが 11 時間生き残っていた**。12:15:56 の先読み（🌛）を 12:15:59 に捨てた（`speculative.discard` の `task.cancel()`）が、
   取り消しがどこかで吸収され、タスクは「完了」も「cancelled」も出さずに `finalize_conversation_turn` の `frontend-playback-complete` 待ちに入った
   （ゲートが閉じたまま捨てた返事なので、ビューワに音は届いておらず、合図は来ない）。以後のすべての発話が「彼女の返事の途中」
   （`[spec] skip: … and she is still answering`、`onset while she speaks`）として扱われた。どこで吸収されたかは未特定（Python 3.11 の
   `wait_for` の取り消し消失の可能性）。→ `discard` は、取り消し後もタスクが終わっていなければ、最大 5 回取り消し直す
   （ログ `[spec] a dropped run survived its cancel — cancelling again`。これが出たら吸収の場所を探す手がかり）。
   症状の見分け方: 誰もいないのに `onset while she speaks` が出続け、最後の `Conversation Chain … started` に対応する `completed` / `cancelled` が無い。直すのは `./kehai.sh restart`。
2. **距離の閾値が M2 用の 0.8 m のままだった**。今日つながったのは大きい PROTO（カメラ「UHD Camera」、マイク「HD Audio Analog Stereo」は左右が同じ信号）で、
   話しかけた人の推定は 1.0〜1.6 m。→ ビューワが知らせるカメラ名で自動で選ぶ（`face_detect.set_viewer_camera`: 「ProtoCam」を含めば M2 として 0.8 m、
   それ以外は 2.0 m。`ADDRESSEE_NEAR_M_SMALL` / `_BIG`、`ADDRESSEE_NEAR_M` を指定すればそれが優先）。`start_server.sh` の既定 0.8 はやめた。
   切り替わるとログ `[addressee] camera '…': … within N m`。
3. **マイク 1 本の PROTO で方向の判定が「100 % 正面」と出ていた**。左右が同じかどうかの移動平均（`voice_direction._same`）に、完全な無音（−180 dBFS、
   この PROTO の待機中）のブロックが「左右が違う」として入り、0.5 を越えなかった。→ −100 dBFS 以下のブロックは数えない、係数を 0.02 → 0.1 に
   （1 s の発話で確定。本物のステレオで左右差が 40 dB 下になることは無いので誤判定しない）。
- PROTO のビューワはサーバーの再起動のあと自分でつながり直す（証明書が変わらなければ）。

### 展示（小学生の団体、2026-11-13 09:50〜13:00）のデータと修正

- **収集**: `scripts/expo_collect.py --until HH:MM`（カメラ 1 枚/秒・VAD の音量・描画タブの動き・健康状態・判定画像と Thinker に渡した音声の写し → `logs/expo/<日付>/`）。
  同じスクリプトに `--watch` を付けた 2 本目は何も記録せず、異常（ポート停止・ビューワ統計が 60 s 無い・fps 低下・例外）で exit 3。
  マイクの元の音と VAD の各区間は `touch logs/asr_debug/ENABLED`（来場者の顔と声が入る。分析が済んだら消す）。
- **マイクを切ってタイプで話す**: `logs/MIC_DEAF` があると、ビューワのマイクは録音だけして VAD に渡さない（再起動不要、1 s ごとに確認）。
  `http://localhost:12394/render-stream/type`（loopback のみ）に入力欄と切り替えのチェックボックス。`POST /render-stream/say {"text"}` は描画タブのセッションに
  text-input として入れる（宛先の判定を通らない）。PROTO につないだキーボードで描画タブの入力欄に打つのも同じ経路。
- **返事が毎回打ち切られていた（09:50〜10:33 の音声 43 分）**: 「彼女宛て」68 ターンのうち 44 が、返事を始めて 1 秒以内に取り消し。
  「受けたターンの直後 1.5 s 以内の声は同じ人の続き → 即停止」（11-08、`_is_continuation`）が、常に誰かが話している部屋では毎回成立する。
  子どもは同じ質問を 3 回繰り返していた（12.6 s の 1 発話）。
  → **騒がしい部屋では彼女は言い切る**（`turn_contingency.busy()`、`TURN_BUSY_OVER_HER`=0.4、0 で無効）: 直近 90 s の彼女の発声（2 s 以上）のうち、
  声がかぶっていた割合が 40 % 以上なら、話している最中の声では止まらない（発話の始まり・区切りの先読み・話し終わりの 3 か所。話し終わりでは、
  発話の半分以上が彼女の声と重なっていれば捨てる `spoken over her reply … in a busy room`）。
  根拠（`[contingency] voice over her`）: 11-09 は中央値 0 %・p90 22 %・40 % 以上のターン 0 %。今日の展示は中央値 56 %・40 % 以上が 81 %。
  **音声を切ったあとに入れたので、実際の騒がしい部屋での確認はまだ。**
- **近くの人の顔が端に外れる**: `ADDRESSEE_EDGE_FRAC`（0.15）。近い顔が時々しか写らなくても、写ったコマの半分以上でこちらを向いていれば答える
  （09:50〜10:33 に 17 ターンがこの規則で通った）。
- **未解決（構造的）**:
  1. **子どもはカメラに写らない**。大きい PROTO のカメラ（UHD Camera、首振り・ズームなし）は上に付いていて、近くに立った小学生の頭は画面の下端か外。
     「顔が無い」で捨てた 152 発話の中に、子どもの質問が入っている。顔に頼らない「近くにいる」の手がかりが要る（カメラを下に向ける・別のカメラ・体の検出）。
  2. **発話が切れない**: 先読み HIT のターンの発話長は中央値 7.1 s、35 中 15 が 8 s 超。部屋の声（60〜70 dB）が発話の維持（55 dB、または発話の音量 −15 dB）を
     越え続ける。返事の遅い回（話し終わり→声の p90 4.9 s、最大 22 s）もここから。録音（viewer.rec と区間）で VAD を机上で流し直して決める。
  3. Whisper は部屋の雑音を「Thank you so much for joining us today, and we'll see you next time.」などと書く（今日 10 回以上。顔が無いので捨てられている）。
  4. 推定 2.2〜2.4 m の顔がいた発話を 107 件捨てている（`NEAR_M` 2.0、`MAX_M` 2.5）。団体は少し離れて立つ。このカメラ（UHD Camera）の距離は未較正。
  5. 描画タブの `headCamOffsetY` が M2 用の 0.27 のまま（大きい PROTO は 0.95）。来場者の「Why are you looking up?」（10:30:15）あり。視線は現地で見て決める。

### 展示の録音を流し直して直した（2026-11-13 午後）

- **試験台**: `scripts/expo_replay.py --dir logs/expo/2026-11-13 --from 10:00:00 --to 10:15:00 --label X`。録音したマイク（`asr_debug/viewer.rec`）とカメラ（`cam/`、1 枚/秒を
  10 回ずつ送る）を `/render-stream/mic-inject` に実時間で流し、終わりにサーバーログから「答えた数・打ち切り・捨てた理由・声までの時間・返事の文」を数える。
  **流している間は PROTO から声が出る**（誰もいないときに）。ターンは描画タブの履歴に入るので、前に `chat_history/iris_android_001` を退避し、後で戻して再起動する。
  返事は抽選で毎回変わるので、1 ターンずつではなく数で比べる。
- **同じ 15 分（10:00〜10:15）の結果**:

  | | A 修正なし | B 言い切る | C ＋一致した発話だけ | D ＋判定を 5 分保持（採用） |
  |---|---|---|---|---|
  | 話し切った返事 | 10 | 29 | 14 | 12 |
  | 打ち切られた返事 | 39 | 8 | 9 | 6 |
  | 言葉が一致せず答えなかった | – | – | 24 | 28 |
  | 話し終わり判定→最初の PCM（中央値） | 197 ms | 198 ms | 857 ms | 688 ms |

  B の 29 件は大半が雑音・作り話の書き取りへの返事（「I understand you are going to the store.」「The restroom is down the hall to your left.」）。
  C・D の返事は実際に聞こえた言葉への返事（「Tell us about dinosaurs.」「Do you like marshmallows?」「Can you hear me?」、一致 74〜100 %）。
- **騒がしい部屋では、2 つの書き取りが一致した発話にだけ答える**（`websocket_handler._understood_in_busy_room`、`BUSY_UNDERSTOOD_GATE`=1、`BUSY_UNDERSTOOD_MIN`=0.5）。
  騒がしい部屋 = `turn_contingency.busy_room()`（`busy()` が今か直近 `TURN_BUSY_HOLD_S`=300 s 以内に真）。その間は fast release をしない（Whisper を待つ）。
  一致は単語の重なりの F1（`turn_contingency.agreement`）。ログ `busy room, the two transcripts agree N%` / `busy room and no words both transcripts heard`。
  - 根拠: 展示の 43 分で、Whisper の書き取りの 48 % が「Thank you for watching.」の類、Thinker は「I'm going to go to the store.」を 108 回書いた。
    この種の組 527 のうち一致は 6。一致した組は本物の発話だった。
  - **静かな部屋では使わない**: 11-09（4 kHz のマイク、訛りのある話者）は本物の 72 ターン中 36 が不一致（Whisper だけが作り話）。一律に掛けると半分落とす。
  - 代償: 騒がしい部屋では返事が約 0.5〜0.65 s 遅い。「Hi. Hi. Hi.」のような短いあいさつは片方が作り話をすると落ちる。
  - 残る誤り: 近くの大人どうしの会話（言葉は本物）には答える（「One at a time.」）。
- 試したが弱かった案: 「両方がその場で繰り返されている決まり文句で、食い違う」ときだけ捨てる（固定リストなし）。静かな 3 日の誤り 0/106 だが、作り話への返事 12 件中 4 件しか防げない。
- **取り消しを生き延びる返事**（11-12 の件）は流し直しの 15 分で 1〜3 回再現する（`[spec] a dropped run survived its cancel`、取り消し直しで回収）。原因の調査に使える。
- 未着手: 発話が部屋の声でつながる件（`VAD_REL_DROP_DB` を変えて同じ試験台で比べる）、起動直後の数分（騒がしいと判定できる前）の打ち切り、子どもがカメラに写らない件
  （カメラは動かせない＝ユーザー回答）、距離の較正、子ども向けの返事、視線の補正値。
- **13:30 追記 — Whisper を待つ版はやめた**（ユーザー「0.5 秒遅くなるのは許されない」）。採用は **E: すぐ答え、Whisper が届いて食い違ったら止める**
  （`_speculate_inner` の released 分岐、ログ `retracted after fast release — busy room, the two transcripts agree N%`。話し終わりの時点で Whisper が届いていれば声を出す前に捨てる）。
  同じ 15 分: 返事 14（本物の言葉へのもの）、打ち切り 3、**話し終わり判定→最初の PCM 中央値 169 ms**（修正なし 197 ms）、撤回 21 回。
  撤回のうち 18 回は声が少し出たあと（TTS のチャンク中央値 3 個、先読み開始から 0.45〜0.7 s で撤回 = 聞こえるのは 0.1〜0.3 s の出かかり。1 s 超が 2 回）。
- **Whisper を使わない見分け方は 5 つ測って全部不可**（区間 220 個、ラベルは Whisper との一致）: 「聞き取れなければ [unclear]」0/162、
  Thinker 2 回（温度 0.4〜1.0）の自己一致 127/162 で本物を 9/40 落とす、頼み方を変えた 2 回 59/162・4/40、「近くの声が 1 つあるか」0〜9 が 114/162・11/40
  （落とす本物は「What are marshmallows made out of?」など子どもの質問そのもの）、音量・長さ 約 3 割・2 割。vLLM-Omni は `n: 2` を無視する（1 件だけ返す）。
- **15:30 追記 — 言いかけて黙る回の根本原因と修正**（同じ 15 分: 声が出てから撤回 18 回 → **1 回**、撤回そのもの 21 → 5）。
  返事は区切りの 280 ms 後に出るのに、照合用の Whisper の書き取りが 400〜700 ms 後に届いていた。遅い理由は 3 つ:
  1. Nexus `model-service/app.py` の `/stt` が、受け取った WAV を一時ファイルに書いて PyAV で読み直していた（短い発話 245 ms のうち 86 ms）。
     → 16 kHz モノラル 16-bit の WAV はメモリ上の配列で渡す（`_pcm16k_from_wav`、写し `app.py.bak-inmem-20261113`）。単体で 281 → 189 ms。
  2. 区切りの書き取りが large-v3・ビーム 5（短い発話 390 ms、6 s 超 1.3 s）。→ 騒がしい部屋の間だけ turbo・ビーム 1
     （`gateway_whisper_asr.async_transcribe_fast`、`ASR_FAST_MODEL`、`ASR_FAST_BUSY=0` で無効）。部屋の中で中央値 233 ms、414 件中 321 件が 280 ms 未満。
     **罠**: `asr_model` は `shadow_asr`（無効でも包む層）で、新しいメソッドを通していなかった → 流し直し 2 回（F・G）は遅いモデルのままだった。
     ログ `[asr] quick transcript in N ms (…): waited its turn …, wav …, request … (server …)` と `[gateway_asr] backend model =` で確認すること。
  3. 取り消した古い書き取りが Whisper 側で走り続ける（要求はスレッドで投げっぱなし。11-13 は 1481 件中 366 件）。Whisper は単独 192 ms、2 件同時 351 ms、3 件 486 ms。
     → セッションごとに同時 1 件（`_asr_one_at_a_time`。順番待ちのうちに取り消された分は送らない）。
  - 効かなかったもの: Whisper の解析窓を音の長さに縮める（158 → 61 ms だが同じ語を繰り返し、本物を 29/40 落とす）、`chunk_length`（時間が変わらない）、
    無音除去と語彙ヒントを外す（10〜30 ms しか変わらない）。
- **「話しかけている人の声だけ」を音の加工で取り出す案は不可**（同じ区間で実測）: DeepFilterNet3 は本物 3.9 dB・雑音 4.9 dB と同じだけ下げる（周りも人の声）。
  SepFormer（whamr16k）の大きい方の流れは、本物の質問 30 件で 2 つの書き取りが成立するのが 24 → 16 件（声を壊す）。マイク 1 本＋AGC では手がかりが無い。
- **次に残っている遅い回（今日の変更とは無関係、流し直しで毎回 4〜9 回）**: (a) Thinker の書き取りが雑音で同じ句を繰り返し（`heard (discarded, looping) in 555 ms`）、
  4 kHz で切れたマイクでは返事がそれを待つので +0.4 s。(b) 話し終わりの瞬間に別の声が始まると先読みが `dropped (new turn)` で捨てられ、返事を作り直す（+0.5 s）。

### 「目の前の人の声だけ」— 無料・ローカルの候補を実測（2026-11-16）

条件はユーザー指定: ソフトだけ・完全ローカル・無料（ai-coustics / Krisp は契約が要るので対象外）。大きい PROTO はマイク 1 本（機器一覧で確認、UHD Camera にマイク無し）。
生の音量（自動音量調整の前）も分かれない: M2 の記録で正面 0.4 m が中央値 −33 dBFS、部屋の声 −35 dBFS。
- **VoxLock**（`pip install voxlock`、Apache-2.0、ONNX、2026-10 公開の個人作。試験は scratchpad の別 venv、本番には未導入）: 3 s の声の手本を渡すと、その声だけ残す
  （手本の既定は音の中で一番大きい 3 s）。11-13 の区間のうち 3 s 以上の 124 個で、発話ごとに `isolate()`:
  - 本物の発話 21 個: 2 つの書き取りの一致 15 → 15（壊さない。DeepFilterNet3 は 24→17/30、SepFormer は 24→16/30 だった）、Whisper の語数 389 → 258（周りの声の語が減る）。
    例: 28 s の発話の Whisper が「What's the real? Why do you tell someone…」→「What do you think my name is?」、12 s が「What dog names do you like? … Did she answer?」。
  - 雑音だけの 92 個: 消えない（一番大きい声を「その人」として残す）。ここは「2 つの書き取りの一致」の関門の担当のまま。
  - 速さ: fp32・CPU 8 スレッドで実時間の約 3 倍（5 s の音に約 1.7 s）。話し終わりにまとめて掛けると遅れるので、使うなら音が届くそばから流す形（`Stream.push`、遅延 24 ms）。
  - 3 s 未満の発話には使えない（手本が取れない）。
- 使うとしたら: 来訪者の最初の発話（カメラの規則で受けたもの）から手本を取り、その人が立ち去る（`VISITOR_RESET`）までマイクの音を VoxLock に通してから VAD・Whisper・Thinker に渡す。未実装。
- Hush（pulp-vision、ライセンス表記なし）は未試験。距離で分ける研究（Patterson 2022）は学習済みモデルの公開を確認できず。
