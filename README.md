# ColabUMA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabUMA/blob/main/ColabUMA.ipynb)

Google Colabで開いて「すべて実行」するだけで、Metaの機械学習ポテンシャル **UMA** を使って分子の構造最適化ができるノートブックです。SMILESか xyz ファイルを入れて、電荷とスピン多重度を指定するだけで使えます。

<!-- TODO: 画面のスクリーンショットを置く（例：docs/screenshot.png） -->

## できること

- SMILES（RDKitで3D化）または xyz ファイルからの構造最適化（LBFGS / BFGS / FIRE）
- 複数の初期配座を生成して全部最適化し、最安定配座を選ぶ
- 振動計算で虚振動の有無を確認
- 3D表示、エネルギー（eV・Hartree）、収束グラフの表示と、結果一式のZIPダウンロード

対象外（v1）：遷移状態探索、IRC、周期系（結晶・表面）の最適化

## 事前準備（初回のみ）

モデルの重み [facebook/UMA](https://huggingface.co/facebook/UMA) はアクセス申請が必要です。

1. [Hugging Face](https://huggingface.co/join) のアカウントを作る
2. [facebook/UMA](https://huggingface.co/facebook/UMA) のページでアクセスを申請し、FAIR Chemistry License v1 に同意する（承認されるまで待つ）
3. [Access Tokens](https://huggingface.co/settings/tokens) で **Read** 権限のトークンを作る
4. Colabでノートを開き、左サイドバーの🔑（シークレット）に名前 `HF_TOKEN`、値にトークンを登録して「ノートブックからのアクセス」をオンにする

シークレットに登録しなかった場合は、実行中にトークンの入力欄が出ます。トークンはノートには保存されません。

## 使い方

1. 上の「Open in Colab」バッジからノートを開く
2. **ランタイム → ランタイムのタイプを変更** で **T4 GPU** を選ぶ
3. 初回のみ、シークレットに `HF_TOKEN` を登録してノートからのアクセスを許可する
4. 「1. 入力フォーム」で SMILES を入れるか xyz アップロードを選び、電荷とスピン多重度を指定する
5. **ランタイム → すべて実行** を押す
6. 3D表示、エネルギー、収束グラフを確認する
7. 結果のZIPをダウンロードする

2回目以降は手順4〜7を繰り返すだけです。インストールとモデル読み込みは自動でスキップされます。

初回のセットアップ（インストールとモデルのダウンロード）にかかる時間：<!-- TODO: 実測して記入 --> 約 X 分

### 入力項目

| 項目 | 既定値 | 説明 |
|---|---|---|
| 入力方法 | SMILES | SMILES / xyzアップロード |
| SMILES | `CCO` | 入力方法がSMILESのときに使用 |
| 電荷 | 0 | 分子全体の電荷 |
| スピン多重度 | 1 | 2S+1。電子数と偶奇が合わなければエラー |
| モデル | uma-s-1p1 | uma-s-1p1 / uma-s-1p2 / uma-m-1p1 |
| タスク | omol | v1は分子用のomolのみ |
| 最適化アルゴリズム | LBFGS | LBFGS / BFGS / FIRE（ASE） |
| 収束判定 fmax | 0.05 | 単位 eV/Å |
| 最大ステップ数 | 500 | |
| 配座数 | 1 | SMILESのときRDKitで生成する初期配座の数。2以上なら全部最適化して最安定を採用 |
| 振動計算 | オフ | オンなら最適化後に虚振動の有無を確認 |
| 出力名 | `result` | 出力ファイル名の接頭辞 |

xyzファイルは1ファイル1構造です。複数の構造が入っている場合は先頭だけを使います。サンプルは [examples/example.xyz](examples/example.xyz) にあります。

### 出力（ZIPの中身）

| ファイル | 内容 |
|---|---|
| `<出力名>_opt.xyz` | 最適化後の構造。コメント行にエネルギーと計算条件 |
| `<出力名>_traj.xyz` | 最適化の全ステップ（複数構造xyz） |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、各ステップのエネルギーと fmax |
| `<出力名>_conformers.csv` | 配座ごとのエネルギー（配座数2以上のとき） |
| `<出力名>_freq.txt` | 振動数の一覧（振動計算オンのとき） |

## 注意事項

- SMILESから作った初期構造は電荷やスピンを考慮していないため、鞍点に収束することがあります。重要な構造は振動計算で確認してください。
- UMAは機械学習ポテンシャルであり、DFTの代わりにはなりません。最終的なエネルギー比較はDFTで確認してください。
- 小さな分子（数十原子以下）では、DFTを直接使う方が確実な場面が多くあります。
- 扱える元素は H〜Bi（原子番号1〜83）です。
- 振動計算では、絶対値が 20 cm⁻¹ 未満の虚振動は数値誤差として本数に数えません（`freq.txt` には全部載ります）。

## 引用

論文で使う場合は、以下を引用してください。

- B. M. Wood *et al.*, "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971 (2025).
- D. S. Levine *et al.*, "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models", arXiv:2505.08762 (2025).
- fairchem: https://github.com/facebookresearch/fairchem

## ライセンス

- ノートのコード：[MIT License](LICENSE)
- UMAのモデル重み：[FAIR Chemistry License v1](https://huggingface.co/facebook/UMA) に従います

## 公開する前に

ノートを更新してコミットする前に、**編集 → すべての出力をクリア** してから保存し、出力セルやトークンが残っていないことを確認してください。
