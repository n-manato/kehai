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
