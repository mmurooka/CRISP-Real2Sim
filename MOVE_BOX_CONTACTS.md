# move_box_adjusted_by_leg の接触推定

入力は `videos/move_box_adjusted_by_leg.mp4` の実写版です。
`videos/move_box_adjusted_by_leg_crisp.mp4` は別PCの復元結果で、入力には使いません。
両動画、venv、モデル、展開フレーム、結果はGitの追跡対象外です。

## 環境

このPCではCUDA 12.4とRTX 5000 Adaを使用します。
Codexの制限付き実行環境からはGPUが見えないため、GPU処理は制限外で実行します。
Conda版セットアップの代わりに、次のvenv版を利用できます。

```bash
bash setups/setup_crisp_venv.sh
bash setups/setup_crisp_contact_venv.sh
```

`TORCH_CUDA_ARCH_LIST` の既定値はこのGPU用の `8.9` です。
他のGPUで構築する場合は値を変更してください。

## 手動取得が必要なデータ

人体モデルは [SMPL](https://smpl.is.tue.mpg.de/) と
[SMPL-X](https://smpl-x.is.tue.mpg.de/) で登録し、`prep/README.md` に従って配置します。
GVHMRでは特に以下が必要です。

```text
prep/HMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
prep/HMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
```

接触推定は [InteractVLM](https://interactvlm.is.tue.mpg.de/login.php) で登録後、
`data.zip` と `interactvlm-3d-hcontact-wScene-damon-lemon-rich.zip` を取得します。
`prep/Contact-Predictor` で展開し、次の構成にしてください。

```text
prep/Contact-Predictor/data/smpl_to_smplx.pkl
prep/Contact-Predictor/data/hcontact_vitruvian/
prep/Contact-Predictor/trained_models/interactvlm-3d-hcontact-wScene-damon-lemon-rich/config.json
```

重みファイル、tokenizer、その他の付属データも各配布アーカイブの構成を保って配置します。
認証情報をGitやチャットに記載する必要はありません。
CRISPのその他の公開重みは `setups/fetch_crisp_assets.sh` が取得します。
Google Driveのダウンロード上限により再試行が必要になる場合があります。

## 実行

リポジトリのルートから実行します。入力へのシンボリックリンクと341枚のフレームは準備済みです。
入力動画を再度変更する場合はフレームも作り直してください。

```bash
source .venv/bin/activate
bash run_crisp_video.sh "$PWD/data/move_box"

# 16 GB GPU向けに接触モデルを4bitでロード。対象は箱との接触。
CONTACT_QUANTIZATION=4bit CUDA_VISIBLE_DEVICES=0 \
  bash scripts/0_interactvlm.sh "$PWD/data/move_box" box

# 人体上の接触領域を赤く表示する。
HUMAN_CONTACT=on USE_CONTACT=off \
  bash vis_scripts/viser_m/vis.sh move_box_adjusted_by_leg
```

接触推定は `.venv-contact/bin/python` を自動選択します。
別の環境を使う場合は `CONTACT_PYTHON=/absolute/path/to/bin/python` を指定します。
接触モデルの場所は `CONTACT_MODEL_PATH` で変更できます。

出力はフレームごとの `results/init/contacts/move_box_adjusted_by_leg/00000.npz` などです。
`pred_contact_3d_smplh` がSMPL頂点順の接触ラベル、
`pred_contact_3d_raw_smplh` がモデルのraw推定値で、SMPL-X版の配列も保存されます。
表示ではSMPLラベルが `0.5` より大きい頂点を赤色にします。
頂点順を維持し、元動画のフレーム番号で描画フレームと対応させます。
可視化画像は `results/init/contact_vis/move_box_adjusted_by_leg/` に保存されます。

CRISPのシーン復元は静的シーンを前提とします。
この動画の移動する箱との接触を静的シーンの拘束に流用しないよう、
上記では `HUMAN_CONTACT=on USE_CONTACT=off` を指定しています。
箱の時系列6D姿勢はこの変更では推定しません。
接触表示を含む実動画での推論と精度確認には、上記のモデル配置が必要です。

## 検証

```bash
uv run --no-project --python 3.10 --with numpy --with trimesh \
  python vis_scripts/viser_m/test_contact_visualization.py
```

フレーム対応、頂点数の不一致、欠損・非有限値、ビューア用GLBへの色の保持を確認します。
