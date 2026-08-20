# kehai-render-probe — レンダリングを数値で検証する MCP サーバー

Claude Code（や他の MCP クライアント）から KEHAI の 3D 描画を**数値で**検証するための
stdio MCP サーバー。TAA/スカート調査（`~/.claude/plans/ui-sprightly-rain.md`）で
scratchpad に散らばっていた計測スクリプト群を、恒久ツールとして固めたもの。

依存の追加はゼロ: playwright-core / pngjs / pixelmatch を
`frontend-src/node_modules` から `createRequire` で借りる。プロトコルは
JSON-RPC 2.0（改行区切り、MCP stdio transport）を手書き実装。

## 前提（起動していないとツールが修復コマンド付きで isError を返す）

```bash
bash scripts/gpu-chrome.sh   # 実GPUの headless Chrome (:9222)。SwiftShader なら拒否される
bash start_server.sh         # アプリ本体 (:12394)。素の run_server.py は使わない
```

## ツール

| ツール | 何を返すか |
|---|---|
| `render_screenshot` | 決定論モード（`?debug=1&t=…&seed=1`、バイト再現）または live でスクショ。PNG の絶対パス |
| `render_flicker` | 連続フレーム間で変化した画素の割合を outline / inner-edge / flat のクラス別に。`sweepKey`/`sweepValues` で `__space` キーの掃引 |
| `render_compare` | 2つの設定パッチの A/B。%diff・平均/最大チャネル差・最密差分領域・A/B/diff PNG パス |
| `render_framestats` | ハーネスが手を離した状態の実フレームコスト（fps / msMedian / msP95）。`seconds` は長いほど安定 |
| `flash_box` | アプリ内フラッシュレコーダ（キー9 / 自動検出）が `logs/flash/` に落としたフレームの一覧/掃除 |

カメラは `preset`（front/final/fullbody/face/hud）か、`camera:[x,y,z]` + `target:[x,y,z]`
の厳密指定（内部で `fitBody:false` を先行させる — framing ソルバが毎フレーム上書きするため）。

## このサーバーが成文化している罠（全部実際に踏んだもの）

- **SwiftShader ゲート**: renderer 文字列が NVIDIA でなければ計測拒否
- **localStorage 衛生**: `vrmSpace1` は実行前後に削除（`cameraDrift:0` の残留が
  taa=0 基準値を 2.05%→0.68% に壊した事故が根拠）。`vrmLook8` は生文字列で退避/復元
- **`taaSettle:0` を常時注入**: 決定論ラッチは保持フレームを再生するので、
  ラッチ中は安定度が偽の 0.000% になり、**ライブの材質変更が画面に出ない**
- **ページ並行禁止**（mutex 直列化）: バックグラウンドタブは rAF が止まり、
  ロックステップ駆動はデッドロックする。クロックが決定論的なので逐次で比較可能
- **`stepFrame` 基準**: `setTime` は時計を止めるだけでループは回り続け、
  時間蓄積系（TAA）がハーネスの往復中に収束してしまう
- **ページは必ず close**（finally 保証）: リークすると WebGL コンテキストを食い潰す

## 単体テスト（GPU 不要のプロトコル確認）

```bash
printf '%s\n' \
 '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"flash_box","arguments":{"action":"list"}}}' \
 | node scripts/mcp-render-probe/server.mjs
```

stdout は正確に3行の JSON（ログは stderr のみ）。stdin を閉じても実行中の
ツールの応答は書き切ってから終了する（in-flight カウンタで exit を遅延）。

## Claude Code への登録

`.mcp.json`（リポジトリルート、登録済み）:

```json
"render-probe": { "command": "node",
  "args": ["/home/ai-ccore/Documents/DEV/KEHAI/scripts/mcp-render-probe/server.mjs"] }
```

プロジェクトスコープの MCP は**初回にセッション起動時の承認プロンプト**が出る。
過去に拒否した場合は `claude mcp reset-project-choices`。既存セッションは
/mcp から再接続。ツール名は `mcp__render-probe__render_screenshot` 等になる。

## 出力

- 画像: `logs/render-probe/`（gitignore 済み・`clean_cache()` の対象外なので残る）
- ツール結果は基本**パス返し**。base64 は返さない（コンテキスト保護）
