# ColabUMA

| ノート | 内容 | 開く |
|---|---|---|
| ColabUMA | 構造最適化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabUMA/blob/main/ColabUMA.ipynb) |
| ColabUMA Scan | 緩和スキャン（結合長・角度・二面角） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabUMA/blob/main/ColabUMA_Scan.ipynb) |
| ColabUMA TS | 遷移状態探索（DMF → Sella → DFT一点計算） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabUMA/blob/main/ColabUMA_TS.ipynb) |
| ColabUMA Workflow | 最適化 → スキャン → 遷移状態を1つのセッションで続けて実行 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabUMA/blob/main/ColabUMA_Workflow.ipynb) |

Google Colabで開いて「すべて実行」するだけで、Metaの機械学習ポテンシャル **UMA** を使って分子の構造最適化、緩和スキャン、遷移状態探索ができるノートブックです。xyz か Gaussian の gjf で構造を入れて、電荷とスピン多重度を指定するだけで使えます。

<!-- TODO: 画面のスクリーンショットを置く（例：docs/screenshot.png） -->

## できること

- xyz ファイル、Gaussian の入力ファイル（.gjf / .com）からの構造最適化（LBFGS / BFGS / FIRE）
- 最適化の経路を Gaussian の log 形式でも出力（GaussView で各ステップの構造とエネルギー変化を確認できる）
- 振動計算で虚振動の有無を確認（振動数は Gaussian の freq 出力の書式でも出力し、GaussView でモードを見られる）
- 3D表示、エネルギー（eV・Hartree）、収束グラフの表示と、結果一式のZIPダウンロード

- 結合長・角度・二面角の緩和スキャンと、スキャン前後の最適化（[ColabUMA Scan](#colabuma-scan)）
- 反応物と生成物からの遷移状態探索（DMF → Sella）、虚振動の確認、簡易IRC、DFT の一点計算（[ColabUMA TS](#colabuma-ts)）
- 最適化 → スキャン → 遷移状態を、インストールとモデル読み込み1回で続けて実行（[ColabUMA Workflow](#colabuma-workflow)）

対象外：IRC そのもの（TS ノートの簡易IRCは、虚振動の向きにずらして最適化するだけ）、周期系（結晶・表面）の最適化

## 事前準備（初回のみ）

モデルの重み [facebook/UMA](https://huggingface.co/facebook/UMA) はアクセス申請が必要です。

1. [Hugging Face](https://huggingface.co/join) のアカウントを作る
2. [facebook/UMA](https://huggingface.co/facebook/UMA) のページでアクセスを申請し、FAIR Chemistry License v1 に同意する（承認されるまで待つ）
3. [Access Tokens](https://huggingface.co/settings/tokens) で **Read** 権限のトークンを作る
4. Colabでノートを開き、左サイドバーの🔑（シークレット）に名前 `HF_TOKEN`、値にトークンを登録して「ノートブックからのアクセス」をオンにする

シークレットに登録しなかった場合は、ファイル選択の前にトークンの入力欄が出ます。トークンはノートには保存されません。

## 使い方

1. 上の「Open in Colab」バッジからノートを開く
2. **ランタイム → ランタイムのタイプを変更** で **T4 GPU** を選ぶ
3. 初回のみ、シークレットに `HF_TOKEN` を登録してノートからのアクセスを許可する
4. 「1. 入力フォーム」で出力名、電荷、スピン多重度、収束判定を指定する
5. **ランタイム → すべて実行** を押す
6. 「2. ファイル選択と事前確認」にファイル選択ボタンが出るので、xyz または gjf を選ぶ。あとはインストールも計算も放置できます
7. 3D表示、エネルギー、収束グラフを確認する
8. 結果のZIPをダウンロードする

2回目以降は手順4〜8を繰り返すだけです。インストールとモデル読み込みは自動でスキップされます。ファイル選択で Cancel upload を押すと、前回のファイルをそのまま使います。

初回のセットアップ（インストールとモデルのダウンロード）にかかる時間：<!-- TODO: 実測して記入 --> 約 X 分

### 入力項目

| 項目 | 既定値 | 説明 |
|---|---|---|
| 出力名 | `result` | 出力ファイル名の接頭辞 |
| 電荷 | 0 | 分子全体の電荷 |
| スピン多重度 | 1 | 2S+1。電子数と偶奇が合わなければエラー |
| モデル | uma-s-1p1 | uma-s-1p1 / uma-s-1p2 / uma-m-1p1 |
| タスク | omol | v1は分子用のomolのみ |
| 最適化アルゴリズム | LBFGS | LBFGS / BFGS / FIRE（ASE） |
| 収束判定 | loose（0.05） | fmax（eV/Å）。loose 0.05 / normal 0.02（Gaussian の既定と同程度）/ tight 0.001（Gaussian の opt=tight と同程度）/ 数値指定 |
| 数値指定の fmax | 0.05 | 収束判定で「数値指定」を選んだときに使用 |
| 最大ステップ数 | 500 | |
| 振動計算 | オフ | オンなら最適化後に虚振動の有無を確認 |

xyzファイルは1ファイル1構造です。複数の構造が入っている場合は先頭だけを使います。サンプルは [examples/example.xyz](examples/example.xyz) と [examples/example.gjf](examples/example.gjf) にあります。

Gaussianの入力ファイル（拡張子 .gjf / .com / .gjc）からは**構造だけ**を読み込みます。ルートセクション（計算方法・基底関数など）は無視し、電荷と多重度はフォームの値を使います（gjfの値と違うときは注意を表示します）。デカルト座標のほか、GaussViewが書く原子タイプ付きの行（`C-CA--0.1`）や凍結フラグ、フラグメント指定、Z-matrix にも対応しています。`--Link1--` がある場合は最初のジョブの構造を使います。

### 出力（ZIPの中身）

| ファイル | 内容 |
|---|---|
| `<出力名>_opt.xyz` | 最適化後の構造。コメント行にエネルギーと計算条件 |
| `<出力名>_opt.log` | 最適化の全ステップを Gaussian の opt 出力と同じ書式で書いたもの。GaussView で開くと経路のアニメーションとエネルギー変化のグラフが見られる |
| `<出力名>_traj.xyz` | 最適化の全ステップ（複数構造xyz） |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、各ステップのエネルギーと fmax |
| `<出力名>_freq.txt` | 振動数の一覧（振動計算オンのとき） |
| `<出力名>_freq.log` | 振動数と基準振動を Gaussian の freq 出力の書式で書いたもの（振動計算オンのとき） |

## ColabUMA Scan

[ColabUMA_Scan.ipynb](ColabUMA_Scan.ipynb) は、結合長・角度・二面角を一定の刻みで動かしながら残りの自由度を最適化する緩和スキャンのノートです。Gaussian の `opt=modredundant` の `S` 指定に相当します。事前準備と使い方の流れは ColabUMA と同じです。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `scan_atoms` | `1 2 3 4` | 原子番号（1始まり）。2個なら結合長、3個なら角度、4個なら二面角 |
| `scan_start` | 空欄 | 開始値（Å または 度）。空欄なら入力構造の値（前の最適化がオンなら最適化後の値） |
| `scan_n_steps` | 36 | 刻みの回数。計算する点の数はこれに1を足した数 |
| `scan_step_size` | 10 | 刻み幅（Å または 度）。負の値で逆向き |
| `opt_before_scan` | オン | スキャンの前に入力構造を拘束なしで最適化する |
| `opt_after_scan` | オン | スキャンの後に拘束なしで最適化する。開始構造は、最後の点からさかのぼって最初に見つかる極小 |
| 収束判定 | normal（0.02） | 選択肢は ColabUMA と同じ。既定値だけが違う |

「後の最適化」は、行き過ぎたスキャンの端から戻って極小を取るためのものです。0.1 kcal/mol 未満の小さな凹凸と未収束の点は無視します。スキャンの端でもエネルギーが下がり続けている場合や、最初の点まで戻った場合は注意を表示します。

原子番号が分からないときは、一度実行すると「5. 構造の準備」に番号付きの3D表示が出ます。

| ファイル | 内容 |
|---|---|
| `<出力名>_scan.csv` | 各点の目標値、実測値、エネルギー、ΔE、fmax、ステップ数、収束（未収束の点も載せます） |
| `<出力名>_scan.xyz` | 収束した各点の最適化構造（複数構造xyz） |
| `<出力名>_scan.log` | 収束した各点の最終構造だけを Gaussian のスキャン出力の書式で書いたもの |
| `<出力名>_preopt.xyz` / `.log` | 前の最適化の結果と経路（オンのとき） |
| `<出力名>_postopt.xyz` / `.log` | 後の最適化の結果と経路（オンのとき） |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、各点と各ステップのエネルギーと fmax |

## ColabUMA TS

[ColabUMA_TS.ipynb](ColabUMA_TS.ipynb) は、反応の遷移状態（TS）を探すノートです。

1. **DMF**（Direct MaxFlux 法、[PyDMF](https://github.com/shin1koda/dmf)）で、反応物と生成物をつなぐ反応経路を最適化し、経路上の最高点を TS の初期構造にする（初期経路は FB-ENM で作る）
2. **Sella** で1次の鞍点に最適化する。最初のヘシアンは UMA の数値微分
3. 振動計算で虚振動が1本だけか確かめ、虚振動の向きに少しずらして両側に最適化し、反応物・生成物と結合のつながりが同じ構造に落ちるかを確かめる（簡易IRC）
4. **DFT の一点計算**（[GPU4PySCF](https://github.com/pyscf/gpu4pyscf)、密度フィッティング）で、UMA の構造のまま反応物・TS・生成物のエネルギーを計算し、UMA と DFT の活性化エネルギーを並べる

反応物と生成物は、**同じ原子が同じ順番で並んだ** 2つのファイル（xyz または gjf）で入れます。スキャンの `preopt.xyz` と `postopt.xyz` はそのまま使えます。TS の初期構造が手元にあるときは、「TSの初期構造（Sellaのみ）」を選ぶと DMF を飛ばします。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `ts_input` | 反応物と生成物 | 「反応物と生成物（DMF → Sella）」か「TSの初期構造（Sellaのみ）」 |
| `opt_endpoints` | オン | DMF の前に反応物と生成物を最適化する |
| `dmf_nmove` | 5 | 経路の途中でエネルギーを計算する点の数 |
| `dmf_convergence` | middle | DMF の収束判定（loose / middle / tight） |
| `dmf_max_iter` | 300 | DMF の最大反復回数 |
| `ts_exact_hessian` | オン | Sella の最初のヘシアンを数値微分で作る（力の計算 6N 回） |
| `check_connection` | オン | 簡易IRC |
| `run_dft` | オン | DFT の一点計算 |
| `dft_functional` / `dft_basis` | ωB97M-V / def2-SVP | ωB97M-V/def2-TZVPD が UMA（OMol25）の学習データと同じ計算レベル。T4 では def2-TZVPD は時間がかかる |
| `gaussian_route` | `# opt=(ts,calcfc,noeigentest) freq b3lyp/def2svp empiricaldispersion=gd3bj` | ZIP に入れる Gaussian 用 gjf のルート行 |

| ファイル | 内容 |
|---|---|
| `<出力名>_ts.xyz` / `_ts.gjf` | TS の構造と、Gaussian で DFT の TS 最適化をするための入力 |
| `<出力名>_ts_opt.log` | Sella の最適化経路（Gaussian の opt=ts の書式） |
| `<出力名>_ts_freq.log` / `.txt` | TS の振動数と基準振動（GaussView で虚振動のアニメーションが見られる） |
| `<出力名>_dmf_path.xyz` / `.log` / `.csv` | DMF の反応経路とエネルギー（GaussView で開くと経路のアニメーション） |
| `<出力名>_dmf_init.xyz`、`_ts_guess.xyz` | FB-ENM の初期経路、DMF の最高点 |
| `<出力名>_reactant.xyz` / `_product.xyz` | 端点（最適化したときは `_opt.log` も） |
| `<出力名>_irc_minus.xyz` / `_irc_plus.xyz` | 簡易IRCで落ちた構造 |
| `<出力名>_dft.csv` | DFT 一点計算のエネルギー |

見つかる TS は UMA のポテンシャル面の TS です。DFT の一点計算は UMA の構造のままなので、最終的には gjf などで DFT の TS 最適化と IRC を行ってください。初回は DMF に必要な IPOPT（cyipopt）をビルドするため、インストールに数分余分にかかります。

## ColabUMA Workflow

Colab ではノートごとに別のセッション（別のマシン）が割り当てられるため、ColabUMA → ColabUMA_Scan → ColabUMA_TS と別々のノートで続けると、そのたびにインストールとモデルのダウンロードをやり直すことになります。

[ColabUMA_Workflow.ipynb](ColabUMA_Workflow.ipynb) は、1つのセッションのまま **構造最適化 → スキャン → 遷移状態 → DFT一点計算** を続けて実行するノートです。前の段階の構造はメモリ上でそのまま次に渡します。

- 最適化した構造を反応物、スキャン後の最適化構造を生成物として DMF に渡す（`ts_source` を「スキャンの最高点から Sella」にすると、DMF を飛ばしてスキャンの最高点から始める）
- 途中の段階だけやり直したいときは、フォームを直してそのセルから下を実行する（**ランタイム → 以降のセルを実行**）。インストールとモデル読み込みはスキップされる
- 結果は全段階を1つの ZIP にまとめる

## コードの構成

ノートのセルはフォームの値を受け取って関数を呼ぶだけにして、計算の中身はリポジトリの [`colabuma/`](colabuma) パッケージにまとめています。各ノートは「2. ファイル選択と事前確認」でこのリポジトリを Colab に取ってきて（`git clone`）読み込みます。ノートを足しても共通部分は増えず、直すときも1か所で済みます。

| モジュール | 内容 |
|---|---|
| `env.py` | GPU とトークンの確認、インストール、Hugging Face の認証、モデル読み込み（同じセッションでは1回だけ） |
| `structure.py` | アップロード、xyz・gjf の読み込み、電荷・多重度の確認、3D表示 |
| `opt.py` | 最適化とプログレスバー、振動解析（数値ヘシアン）、Opt ノートの結果表示 |
| `scan.py` | 緩和スキャン |
| `ts.py` | DMF、Sella、TS の確認、簡易IRC |
| `dft.py` | DFT の一点計算（gpu4pyscf、使えなければ CPU の PySCF） |
| `gaussian.py` | Gaussian の log（opt / scan / freq / 経路）と gjf の書き出し |
| `output.py` | log.txt と ZIP |

別のブランチのコードで試すときは、「2.」のセルの `COLABUMA_REF` をそのブランチ名にしてください（すでに取ってきたあとなら、セッションを再起動してから）。リポジトリを手元に clone してノートを開いた場合は、そのフォルダの `colabuma` を使います。

## 注意事項

- 最適化は鞍点に収束することがあります。重要な構造は振動計算で確認してください。
- UMAは機械学習ポテンシャルであり、DFTの代わりにはなりません。最終的なエネルギー比較はDFTで確認してください。
- 小さな分子（数十原子以下）では、DFTを直接使う方が確実な場面が多くあります。
- 扱える元素は H〜Bi（原子番号1〜83）です。
- `_opt.log` は表示用に Gaussian の書式をまねたファイルで、Gaussian で計算した結果ではありません。エネルギー（Hartree）と力は UMA の値です。収束判定の閾値欄には、フォームの fmax を Hartree/Bohr に換算した値を、Gaussian の既定値と同じ比率で書いています。
- 振動計算は UMA の力の中心差分（0.01 Å）でヘシアンを作り、並進・回転を射影で取り除いてから対角化します。絶対値が 20 cm⁻¹ 未満の虚振動は数値誤差として本数に数えません（`freq.txt` には射影前の値も載ります）。

## 引用

論文で使う場合は、以下を引用してください。

- B. M. Wood *et al.*, "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971 (2025).
- D. S. Levine *et al.*, "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models", arXiv:2505.08762 (2025).
- fairchem: https://github.com/facebookresearch/fairchem

ColabUMA TS で遷移状態探索を使った場合は、次も引用してください。

- S.-i. Koda and S. Saito, *J. Open Source Softw.* **11**, 10379 (2026).（PyDMF）
- S.-i. Koda and S. Saito, *J. Chem. Theory Comput.* **20**, 2798–2811 (2024).（Direct MaxFlux）
- S.-i. Koda and S. Saito, *J. Chem. Theory Comput.* **20**, 7176–7187 (2024); **21**, 3513–3522 (2025).（FB-ENM、CFB-ENM）
- E. D. Hermes, K. Sargsyan, H. N. Najm, J. Zádor, *J. Chem. Theory Comput.* **18**, 6974–6988 (2022).（Sella）
- DFT の一点計算を使った場合：PySCF（Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020)）と GPU4PySCF（X. Wu *et al.*, arXiv:2404.09452 (2024)）

## ライセンス

- ノートのコード：[MIT License](LICENSE)
- UMAのモデル重み：[FAIR Chemistry License v1](https://huggingface.co/facebook/UMA) に従います

## 公開する前に

ノートを更新してコミットする前に、**編集 → すべての出力をクリア** してから保存し、出力セルやトークンが残っていないことを確認してください。
