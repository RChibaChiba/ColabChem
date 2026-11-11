# ColabChem

[English](README.md) | **日本語**

[ホームページ（ColabChem）](https://rchibachiba.github.io/ColabChem/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24035698.svg)](https://doi.org/10.5281/zenodo.24035698)

| ノート | 内容 | 開く |
|---|---|---|
| ColabUMA Opt | 構造最適化 → DFT一点計算 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Opt.ipynb) |
| ColabUMA Scan | 緩和スキャン（結合長・角度・二面角） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Scan.ipynb) |
| ColabUMA Conf | 配座探索（高温MD → 最適化 → 低エネルギーの数構造）。金属錯体と基質の会合体など | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Conf.ipynb) |
| ColabUMA TS | 遷移状態探索（DMF → Sella → DFT一点計算） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_TS.ipynb) |
| ColabUMA Workflow | 最適化 → スキャン → 遷移状態を1つのセッションで続けて実行 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabUMA_Workflow.ipynb) |
| ColabDFT SPE | DFT一点計算のみ（UMAもトークンも不要） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabDFT_SPE.ipynb) |
| ColabDFT ESP | DFT一点計算と静電ポテンシャル（ESP）の3D表示（UMAもトークンも不要） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabDFT_ESP.ipynb) |
| ColabDFT MO | DFT一点計算と分子軌道（HOMO・LUMO など）の3D表示（UMAもトークンも不要） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RChibaChiba/ColabChem/blob/main/ColabDFT_MO.ipynb) |

Google Colabで開いて「すべて実行」するだけで、Metaの機械学習ポテンシャル **UMA** を使って分子の構造最適化、緩和スキャン、遷移状態探索ができるノートブックです。xyz か Gaussian の gjf で構造を入れて、電荷とスピン多重度を指定するだけで使えます。

ノートのフォーム・メッセージ・ログは英語です。エラーには日本語の説明も出ます。ノート冒頭の説明とこの README は日本語です。冒頭の説明が英語のノートは、同じ名前に `_en` を付けたもの（例：`ColabUMA_Opt_en.ipynb`）で、中身のコードは同じです。

<!-- TODO: 画面のスクリーンショットを置く（例：docs/screenshot.png） -->

## できること

- xyz ファイル、Gaussian の入力ファイル（.gjf / .com）からの構造最適化（LBFGS / BFGS / FIRE）
- 最適化の経路を Gaussian の log 形式でも出力（GaussView で各ステップの構造とエネルギー変化を確認できる）
- 振動計算で虚振動の有無を確認（振動数は Gaussian の freq 出力の書式でも出力し、GaussView でモードを見られる）
- 最適化した構造での DFT の一点計算（GPU4PySCF）
- 3D表示、エネルギー（eV・Hartree）、収束グラフの表示と、結果一式のZIPダウンロード

- 結合長・角度・二面角の緩和スキャンと、スキャン前後の最適化（[ColabUMA Scan](#colabuma-scan)）
- 金属錯体と基質の会合体などの配座探索（高温 MD → 最適化 → 重複を除いて低エネルギーの数構造）（[ColabUMA Conf](#colabuma-conf)）
- 反応物と生成物からの遷移状態探索（DMF → Sella）、虚振動の確認、簡易IRC、DFT の一点計算（[ColabUMA TS](#colabuma-ts)）
- 最適化 → スキャン → 遷移状態を、インストールとモデル読み込み1回で続けて実行（[ColabUMA Workflow](#colabuma-workflow)）
- 手元の構造をそのまま DFT で一点計算（[ColabDFT SPE](#colabdft-spe)）
- 手元の構造の静電ポテンシャル（ESP）を等密度面に色付けして表示（[ColabDFT ESP](#colabdft-esp)）
- 手元の構造の HOMO・LUMO などの分子軌道を等値面で表示（[ColabDFT MO](#colabdft-mo)）

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
4. 「1. Input form」で出力名、電荷、スピン多重度、収束判定を指定する
5. **ランタイム → すべて実行** を押す
6. 「2. File selection and checks」にファイル選択ボタンが出るので、xyz または gjf を選ぶ。あとはインストールも計算も放置できます
7. 3D表示、エネルギー、収束グラフを確認する
8. 結果のZIPをダウンロードする

2回目以降は手順4〜8を繰り返すだけです。インストールとモデル読み込みは自動でスキップされます。ファイル選択で Cancel upload を押すと、前回のファイルをそのまま使います。

### 入力項目

| 項目 | 既定値 | 説明 |
|---|---|---|
| 出力名 | `result` | 出力ファイル名の接頭辞 |
| 電荷 | 0 | 分子全体の電荷 |
| スピン多重度 | 1 | 2S+1。電子数と偶奇が合わなければエラー |
| モデル | uma-s-1p1 | uma-s-1p1 / uma-s-1p2 / uma-m-1p1 |
| タスク | omol | v1は分子用のomolのみ |
| 最適化アルゴリズム | LBFGS | LBFGS / BFGS / FIRE（ASE） |
| 収束判定 | loose (0.05) | fmax（eV/Å）。loose 0.05 / normal 0.02（Gaussian の既定と同程度）/ tight 0.001（Gaussian の opt=tight と同程度）/ custom |
| custom の fmax | 0.05 | 収束判定で「custom」を選んだときに使用 |
| 最大ステップ数 | 500 | |
| 振動計算 | オフ | オンなら最適化後に虚振動の有無を確認 |
| DFT 一点計算 | オフ | 最適化した構造で DFT のエネルギーを計算する（GPU4PySCF、密度フィッティング）。UMA の構造のままで、DFT で最適化し直すものではない |
| 汎関数 / 基底関数 | B3LYP-D3(BJ) / def2-SVP | 既定は計算の速さを優先。ωB97M-V/def2-TZVPD が UMA（OMol25）の学習データと同じ計算レベルだが、T4 では時間がかかる |
| DFT の時間の上限 | 60 分 | DFT 一点計算にかける時間の上限（全構造の合計）。超えたら DFT を飛ばして UMA の結果を保存する。0 なら上限なし（[DFT のメモリと時間](#dft-のメモリと時間)） |

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
| `<出力名>_dft.csv` | DFT 一点計算のエネルギー（DFT 一点計算オンのとき） |

## ColabUMA Scan

[ColabUMA_Scan.ipynb](ColabUMA_Scan.ipynb) は、結合長・角度・二面角を一定の刻みで動かしながら残りの自由度を最適化する緩和スキャンのノートです。Gaussian の `opt=modredundant` の `S` 指定に相当します。事前準備と使い方の流れは ColabUMA Opt と同じです。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `scan_atoms` | `1 2 3 4` | 原子番号（1始まり）。2個なら結合長、3個なら角度、4個なら二面角 |
| `scan_start` | 空欄 | 開始値（Å または 度）。空欄なら入力構造の値（前の最適化がオンなら最適化後の値） |
| `scan_n_steps` | 36 | 刻みの回数。計算する点の数はこれに1を足した数 |
| `scan_step_size` | 10 | 刻み幅（Å または 度）。負の値で逆向き |
| `opt_before_scan` | オン | スキャンの前に入力構造を拘束なしで最適化する |
| `opt_after_scan` | オン | スキャンの後に拘束なしで最適化する。開始構造は、最後の点からさかのぼって最初に見つかる極小 |
| 収束判定 | normal (0.02) | 選択肢は ColabUMA Opt と同じ。既定値だけが違う |

「後の最適化」は、行き過ぎたスキャンの端から戻って極小を取るためのものです。0.1 kcal/mol 未満の小さな凹凸と未収束の点は無視します。スキャンの端でもエネルギーが下がり続けている場合や、最初の点まで戻った場合は注意を表示します。

原子番号が分からないときは、一度実行すると「5. Prepare the structures」に番号付きの3D表示が出ます。

| ファイル | 内容 |
|---|---|
| `<出力名>_scan.csv` | 各点の目標値、実測値、エネルギー、ΔE、fmax、ステップ数、収束（未収束の点も載せます） |
| `<出力名>_scan.xyz` | 収束した各点の最適化構造（複数構造xyz） |
| `<出力名>_scan.log` | 収束した各点の最終構造だけを Gaussian のスキャン出力の書式で書いたもの |
| `<出力名>_preopt.xyz` / `.log` | 前の最適化の結果と経路（オンのとき） |
| `<出力名>_postopt.xyz` / `.log` | 後の最適化の結果と経路（オンのとき） |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、各点と各ステップのエネルギーと fmax |

## ColabUMA Conf

[ColabUMA_Conf.ipynb](ColabUMA_Conf.ipynb) は、配座探索のノートです。金属錯体に基質を置いた構造を1つ入れると、エネルギーの低い順に数個（既定は5個）の配座を出します。事前準備と使い方の流れは ColabUMA Opt と同じです。追加でインストールするソフトはありません。

1. 入力構造を最適化し、結合のつながり（どの原子どうしが結合しているか）を読み取る
2. 高温の Langevin MD を何本か走らせ、等間隔にスナップショットを取る
3. スナップショットを粗く最適化し（fmax 0.1 eV/Å、最大60ステップ）、エネルギーの低いものだけを最後まで最適化する
4. 結合のつながりが変わった構造と、同じ配座の重複を除き、エネルギーの低い順に並べる

基質が金属に配位しているか、非共有結合で付いているだけかは指定しません。最適化した入力構造で結合している原子の組（共有結合半径の和の1.25倍より近い組。金属と配位原子の組も含む）をそのまま保ちます。入力で配位していれば配位を保ったまま配座だけが変わり、付いているだけなら基質の位置と向きも変わります。MD のあいだは、結合が伸びすぎたときだけ引き戻す拘束と、分子全体を囲む球の壁を掛けます（ふつうの揺れの範囲では力は0）。最適化には拘束を掛けません。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `n_conformers` | 5 | 出力する配座の数 |
| `md_temperature` | 600 | MD の温度（K） |
| `md_time_ps` | 20 | MD の時間（ps、全部の MD の合計）。刻みは 2 fs で、MD のあいだだけ水素の質量を 4 にする |
| `n_replicas` | 4 | MD の本数。初速を変えて同じ構造から始める |
| `n_snapshots` | 100 | MD から等間隔に取り出す構造の数 |
| `n_refine` | 20 | 粗い最適化のあと、最後まで最適化する候補の数（最安定から 15 kcal/mol 以内のもの） |
| `keep_coordination` | オン | 入力構造の金属まわりの結合を保つ。オフにすると金属が関わる組を見張らず、配位子や基質が金属に付いたり離れたりした構造も残る |
| `random_seed` | 0 | MD の初速と乱数の種 |
| 収束判定 | normal (0.02) | 最後の最適化の fmax。選択肢は ColabUMA Opt と同じ |
| `run_dft` | オフ | 出力する配座の DFT 一点計算。項目は ColabUMA Opt と同じ |

捨てる構造と重複の判定：

- 結合が切れた（判定距離の1.15倍より遠い）、新しい結合ができた（判定距離の0.9倍より近い）、金属以外の4配位の中心が反転した構造は捨てる
- 水素以外の原子で重ね合わせた RMSD が 0.25 Å 未満で、どの原子のずれも 0.7 Å 未満なら同じ配座
- エネルギーの差が 0.1 kcal/mol 未満で、3つの慣性モーメントがどれも 1% 以内なら同じ配座（メチル基の回転のように、原子を入れ替えると重なるもの。鏡像の配座もここで1つになる）

| ファイル | 内容 |
|---|---|
| `<出力名>_conformers.xyz` | 出力する配座（エネルギーの低い順、複数構造xyz）。1番目に重ね合わせてある |
| `<出力名>_conf1.xyz` … | 各配座を1つずつ |
| `<出力名>_conformers.csv` | ΔE（kcal/mol）、298 K での存在比、エネルギー、1番目との RMSD、収束 |
| `<出力名>_ensemble.xyz` | 最後まで最適化して残った配座すべて |
| `<出力名>_input_opt.xyz` | 入力構造を最適化したもの |
| `<出力名>_dft.csv` | DFT 一点計算のエネルギー（オンのとき） |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、探索の経過、各配座のエネルギー |

高温 MD による探索は、すべての配座を見つける保証がありません。`md_time_ps` を長くしたり `random_seed` を変えたりして、同じ最安定構造が出るか確かめてください。得られるのは入力構造と同じ結合のつながりを持つ配座なので、配位の仕方が違う異性体は、それぞれを入力にして別々に探索してください。存在比は、出力した配座の中での ΔE（電子エネルギー）だけから求めた目安です。

## ColabUMA TS

[ColabUMA_TS.ipynb](ColabUMA_TS.ipynb) は、反応の遷移状態（TS）を探すノートです。

1. **DMF**（Direct MaxFlux 法、[PyDMF](https://github.com/shin1koda/dmf)）で、反応物と生成物をつなぐ反応経路を最適化し、経路上の最高点を TS の初期構造にする（初期経路は FB-ENM で作る）
2. **Sella** で1次の鞍点に最適化する。最初のヘシアンは UMA の数値微分
3. 振動計算で虚振動が1本だけか確かめ、虚振動の向きに少しずらして両側に最適化し、反応物・生成物と結合のつながりが同じ構造に落ちるかを確かめる（簡易IRC）
4. **DFT の一点計算**（[GPU4PySCF](https://github.com/pyscf/gpu4pyscf)、密度フィッティング）で、UMA の構造のまま反応物・TS・生成物のエネルギーを計算し、UMA と DFT の活性化エネルギーを並べる

反応物と生成物は、**同じ原子が同じ順番で並んだ** 2つのファイル（xyz または gjf）で入れます。スキャンの `preopt.xyz` と `postopt.xyz` はそのまま使えます。TS の初期構造が手元にあるときは、「TSの初期構造（Sellaのみ）」を選ぶと DMF を飛ばします。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `ts_input` | reactant and product (DMF → Sella) | 「reactant and product (DMF → Sella)」（反応物と生成物から）か「TS guess (Sella only)」（TS の初期構造から） |
| `opt_endpoints` | オン | DMF の前に反応物と生成物を最適化する |
| `dmf_nmove` | 5 | 経路の途中でエネルギーを計算する点の数 |
| `dmf_convergence` | middle | DMF の収束判定（loose / middle / tight） |
| `dmf_max_iter` | 300 | DMF の最大反復回数 |
| `ts_exact_hessian` | オン | Sella の最初のヘシアンを数値微分で作る（力の計算 6N 回） |
| `check_connection` | オン | 簡易IRC |
| `run_dft` | オン | DFT の一点計算 |
| `dft_functional` / `dft_basis` | B3LYP-D3(BJ) / def2-SVP | 既定は計算の速さを優先。ωB97M-V/def2-TZVPD が UMA（OMol25）の学習データと同じ計算レベルだが、T4 では時間がかかる |
| `dft_time_limit` | 60 | DFT にかける時間の上限（分、反応物・TS・生成物の合計）。0 なら上限なし |
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

Colab ではノートごとに別のセッション（別のマシン）が割り当てられるため、ColabUMA_Opt → ColabUMA_Scan → ColabUMA_TS と別々のノートで続けると、そのたびにインストールとモデルのダウンロードをやり直すことになります。

[ColabUMA_Workflow.ipynb](ColabUMA_Workflow.ipynb) は、1つのセッションのまま **構造最適化 → スキャン → 遷移状態 → DFT一点計算** を続けて実行するノートです。前の段階の構造はメモリ上でそのまま次に渡します。

- 最適化した構造を反応物、スキャン後の最適化構造を生成物として DMF に渡す（`ts_source` を「Sella from the scan maximum」にすると、DMF を飛ばしてスキャンの最高点から始める）
- 途中の段階だけやり直したいときは、フォームを直してそのセルから下を実行する（**ランタイム → 以降のセルを実行**）。インストールとモデル読み込みはスキップされる
- 結果は全段階を1つの ZIP にまとめる
- 遷移状態の段階の後に、スキャンのエネルギー曲線に DMF の経路・TS の初期構造・Sella の TS・生成物を重ねた図を出す。横軸はそれぞれの構造でスキャンした座標を測った値
- DFT の一点計算は既定でオフ（`run_dft` をオンにすると計算する）。時間の上限 `dft_time_limit` は TS と同じ
- ZIP には Gaussian 用の TS の gjf（`_ts.gjf`、ルート行は `gaussian_route`）と IRC の gjf（`_irc.gjf`、ルート行は `gaussian_irc_route`）が入る。どちらも構造は UMA の TS

## ColabDFT SPE

[ColabDFT_SPE.ipynb](ColabDFT_SPE.ipynb) は、入力した構造のまま GPU4PySCF で DFT の一点計算だけを行うノートです。構造の最適化はしません。UMA を使わないので、[事前準備](#事前準備初回のみ)の Hugging Face のトークンはいりません。fairchem もインストールしません。

| 項目 | 既定値 | 説明 |
|---|---|---|
| 電荷 / スピン多重度 | 0 / 1 | gjf に書かれた値ではなくフォームの値を使う。多重度1は RKS、それ以外は UKS |
| 汎関数 | B3LYP-D3(BJ) | ωB97M-V / B3LYP-D3(BJ) / PBE0-D3(BJ) / M06-2X / r2SCAN-D4 |
| 基底関数 | def2-SVP | def2-SVP / def2-TZVP / def2-TZVPD |

xyz に複数の構造が入っている場合は先頭だけを計算します。扱える元素は H〜Rn（原子番号1〜86）です。

| ファイル | 内容 |
|---|---|
| `<出力名>_spe.xyz` | 計算した構造。コメント行にエネルギーと計算条件 |
| `<出力名>_dft.csv` | DFT 一点計算のエネルギー |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、計算結果 |

## ColabDFT ESP

[ColabDFT_ESP.ipynb](ColabDFT_ESP.ipynb) は、入力した構造のまま DFT の一点計算を行い、静電ポテンシャル（ESP）を等密度面に色付けして 3D 表示するノートです。赤が負、青が正です。構造の最適化はせず、UMA もトークンもいりません。電荷、多重度、汎関数、基底関数の項目は [ColabDFT SPE](#colabdft-spe) と同じです。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `input_mode` | file | 「file」は xyz・gjf を選ぶ。「draw」は構造エディタ（JSME）で描き、水素を付けて RDKit で 3D にする（ETKDG の 10 配座を MMFF94 で最適化し、一番低いもの）。構造の最適化はしない |
| `isosurface` | 0.002 | 色を付ける等密度面の値（e/Bohr³） |
| `grid_quality` | standard | 格子の間隔。coarse 0.5 / standard 0.35 / fine 0.25 Bohr。分子が大きいときは自動で粗くする |
| `esp_color_range` | 0 | 色の範囲 ±（kcal/mol）。0 なら自動。分子どうしを比べるときは同じ値にする |

| ファイル | 内容 |
|---|---|
| `<出力名>_esp.cube` | 格子上の ESP（a.u.） |
| `<出力名>_density.cube` | 格子上の電子密度（e/Bohr³）。GaussView では、これの等値面に `_esp.cube` を色付けする |
| `<出力名>_spe.xyz` | 計算した構造。コメント行にエネルギーと計算条件 |
| `<出力名>_dft.csv` | DFT 一点計算のエネルギー |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、計算結果、面の上の ESP の最小・最大 |

## ColabDFT MO

[ColabDFT_MO.ipynb](ColabDFT_MO.ipynb) は、入力した構造のまま DFT の一点計算を行い、HOMO・LUMO などの分子軌道（Kohn–Sham 軌道）を正と負の等値面で 3D 表示するノートです。青が正、赤が負です（軌道の符号には意味がないので、入れ替わることがあります）。左にエネルギー準位図（占有は ↑↓）を出し、線をクリックするとその軌道を右に表示します。フォームで指定していない軌道も、クリックしたときに Colab のランタイムで計算します（ランタイムにつながっている間だけ）。HOMO・LUMO のエネルギーとギャップ、HOMO・LUMO の近くの軌道のエネルギーの表も出します。ZIP に入る cube はフォームで指定した軌道だけです。開殻では α と β の軌道をそれぞれ表示します。構造の最適化はせず、UMA もトークンもいりません。電荷、多重度、汎関数、基底関数の項目は [ColabDFT SPE](#colabdft-spe) と同じです。

| 項目 | 既定値 | 説明 |
|---|---|---|
| `input_mode` | file | 「file」は xyz・gjf を選ぶ。「draw」は構造エディタ（JSME）で描き、水素を付けて RDKit で 3D にする（ETKDG の 10 配座を MMFF94 で最適化し、一番低いもの）。構造の最適化はしない |
| `orbitals` | HOMO, LUMO | 表示する軌道をカンマで区切って書く。`HOMO-1`、`LUMO+2` や、軌道の番号（エネルギーの低い順に 1 から）も使える。一度に 8 個まで |
| `isovalue` | 0.02 | 等値面の値 ±（Bohr^-3/2）。GaussView の既定値と同じ |
| `grid_quality` | standard | 格子の間隔。coarse 0.5 / standard 0.35 / fine 0.25 Bohr。分子が大きいときは自動で粗くする |

| ファイル | 内容 |
|---|---|
| `<出力名>_mo<番号>_<軌道>.cube` | 格子上の軌道の値（Bohr^-3/2）。例：`_mo5_homo.cube`、`_mo4_homom1.cube`（HOMO-1）、`_mo7_lumop1.cube`（LUMO+1）。開殻では `_alpha` / `_beta` が付く |
| `<出力名>_mo_energies.csv` | 全軌道のエネルギー（Hartree・eV）と占有数 |
| `<出力名>_spe.xyz` | 計算した構造。コメント行にエネルギーと計算条件 |
| `<出力名>_dft.csv` | DFT 一点計算のエネルギー |
| `<出力名>_log.txt` | 入力条件、ソフトのバージョン、計算結果、HOMO・LUMO のエネルギーとギャップ |

## DFT のメモリと時間

UMA のノート（ColabUMA Opt、TS、Workflow）では、DFT の一点計算がうまくいかなくても、**DFT だけを飛ばして UMA の結果の保存まで進みます**。飛ばしたときは理由を注意として表示し、`log.txt` にも書きます。途中の構造まで終わっていれば、その分のエネルギーは `dft.csv` に入ります。

DFT を飛ばすのは次のときです。

- **メモリが足りない見込みのとき**：計算を始める前に、密度フィッティングの3中心積分の大きさを見積もります。gpu4pyscf はこれを GPU のメモリに置き、収まらなければ CPU のメモリに置きます。CPU のメモリまで足りなくなると、Colab ではエラーにならずにランタイムごと落ちて、UMA の結果も消えます。そうなりそうなときは、計算を始めずに DFT を飛ばします
- **計算中にメモリが足りなくなったとき**（GPU のメモリ不足のエラーなど）
- **時間の上限を過ぎたとき**：フォームの DFT の時間の上限（`dft_time_limit`、既定 60 分、全構造の合計）を過ぎると、SCF の途中でも止めます。前の構造と同じくらいかかりそうで間に合わないときは、次の構造を始めません
- **Colab のランタイムが切れそうなとき**：Colab の無料版は、ランタイムが最長でおよそ12時間で切れます。ランタイムが始まってから12時間の15分前までに終わらない DFT は止めます（上限を 0 にしても同じ）
- そのほか、DFT の計算でエラーが起きたとき

**メモリの注意**：メモリが足りるか怪しいときは、「Prepare the structure(s)」のセル（UMA の計算を始める前）と DFT のセルで注意を出します。3中心積分の大きさは上限の見積もりで、実際は小さい積分を省くのでこれより小さくなることが多いです。目安として、T4（GPU 15 GB、CPU 約 12 GB）では、80 原子程度の分子なら def2-SVP は GPU に収まり、def2-TZVP 以上は CPU のメモリも足りなくなる見込みで DFT を飛ばします。

ColabDFT SPE、ESP、MO は DFT そのものが目的なので、飛ばさずに計算します。メモリが怪しいときは注意だけ出します。

Colab のタイムアウトについて：

- 無料版のランタイムは最長でおよそ12時間です（Colab Pro では長くなります）
- ブラウザを長く操作しないと、計算中でも切れることがあります。長い計算のときはタブを開いたままにしてください
- ランタイムが切れると、それまでの結果は Colab のマシンごと消えます。ZIP はそのたびにダウンロードしておいてください

## コードの構成

ノートのセルはフォームの値を受け取って関数を呼ぶだけにして、計算の中身はリポジトリの [`colabuma/`](colabuma) パッケージにまとめています。各ノートは「2. ファイル選択と事前確認」でこのリポジトリを Colab に取ってきて（`git clone`）読み込みます。ノートを足しても共通部分は増えず、直すときも1か所で済みます。

| モジュール | 内容 |
|---|---|
| `env.py` | GPU とトークンの確認、インストール、Hugging Face の認証、モデル読み込み（同じセッションでは1回だけ） |
| `structure.py`、`draw.html` | アップロード、構造を描く入力（JSME → RDKit で 3D 化）、xyz・gjf の読み込み、電荷・多重度の確認、3D表示 |
| `opt.py` | 最適化とプログレスバー、振動解析（数値ヘシアン）、Opt ノートの結果表示 |
| `scan.py` | 緩和スキャン |
| `conf.py` | 配座探索（結合のつながりの判定、高温 MD、重複の判定） |
| `ts.py` | DMF、Sella、TS の確認、簡易IRC |
| `dft.py` | DFT の一点計算（gpu4pyscf、使えなければ CPU の PySCF）、メモリの見積もり、時間の上限 |
| `esp.py` | 格子上の電子密度と ESP、cube の書き出し、等密度面の表示 |
| `mo.py`、`mo_explorer.html` | 格子上の分子軌道、軌道のエネルギー、クリックできる準位図と正・負の等値面の表示 |
| `gaussian.py` | Gaussian の log（opt / scan / freq / 経路）と gjf の書き出し |
| `output.py` | log.txt と ZIP |

別のブランチのコードで試すときは、「2.」のセルの `COLABUMA_REF` をそのブランチ名にしてください（すでに取ってきたあとなら、セッションを再起動してから）。リポジトリを手元に clone してノートを開いた場合は、そのフォルダの `colabuma` を使います。

ノートはふだん main の最新のコードを使います。論文などで使った版を再現したいときは、`COLABUMA_REF` をリリースのタグ名（例：`v0.3.0`）にしてください。使ったバージョンとコミットは `log.txt` の「[Software]」に書かれています。

日本語版と英語版（`_en`）のノートは、コードのセルが同じで、最初の説明のセルだけが違います。コードのセルを直すときは、両方のファイルを直してください。

## 注意事項

- 最適化は鞍点に収束することがあります。重要な構造は振動計算で確認してください。
- UMAは機械学習ポテンシャルであり、DFTの代わりにはなりません。最終的なエネルギー比較はDFTで確認してください。
- 小さな分子（数十原子以下）では、DFTを直接使う方が確実な場面が多くあります。
- 扱える元素は H〜Bi（原子番号1〜83）です。
- `_opt.log` は表示用に Gaussian の書式をまねたファイルで、Gaussian で計算した結果ではありません。エネルギー（Hartree）と力は UMA の値です。収束判定の閾値欄には、フォームの fmax を Hartree/Bohr に換算した値を、Gaussian の既定値と同じ比率で書いています。
- 振動計算は UMA の力の中心差分（0.01 Å）でヘシアンを作り、並進・回転を射影で取り除いてから対角化します。絶対値が 20 cm⁻¹ 未満の虚振動は数値誤差として本数に数えません（`freq.txt` には射影前の値も載ります）。

## 謝辞・参考にしたもの

ColabChem は、[ColabReaction](https://github.com/BILAB/ColabReaction)（東京大学 生物情報工学研究室、[ColabReaction.net](https://colabreaction.net/)）を大いに参考にしています。Google Colab で機械学習ポテンシャル（UMA）と Direct MaxFlux（DMF）を使って遷移状態を探すという考え方や、ノートの構成の多くは ColabReaction にならったものです。

- M. Karasawa, C. S. Leow, H. Yajima, S. Arai, H. Nishizaki, T. Terada, H. Sato, "ColabReaction: Accelerating Transition State Searches with Machine Learning Potentials on Google Colaboratory", *J. Chem. Inf. Model.* **65**, 11908–11914 (2025). https://doi.org/10.1021/acs.jcim.5c02398

## 引用

論文で使う場合は、ColabChem に加えて、使った手法の文献も引用してください。ColabChem の引用情報は [CITATION.cff](CITATION.cff) にあり、GitHub のリポジトリページ右側の「Cite this repository」から APA や BibTeX の形式でコピーできます。

- R. Chiba, ColabChem, Zenodo, https://doi.org/10.5281/zenodo.24035698（すべての版をまとめた DOI。版ごとの DOI は Zenodo のページにあり、v0.3.0 は 10.5281/zenodo.24035699）
- A. H. Larsen *et al.*, *J. Phys.: Condens. Matter* **29**, 273002 (2017).（ASE。すべてのノートで使います）

UMA を使うノート（ColabUMA）では、次も引用してください（ColabDFT では UMA を使わないので不要です）。

- B. M. Wood *et al.*, "UMA: A Family of Universal Models for Atoms", arXiv:2506.23971 (2025).
- D. S. Levine *et al.*, "The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models", arXiv:2505.08762 (2025).
- fairchem: https://github.com/facebookresearch/fairchem

ColabUMA TS で遷移状態探索を使った場合は、次も引用してください。

- S.-i. Koda and S. Saito, *J. Open Source Softw.* **11**, 10379 (2026).（PyDMF）
- S.-i. Koda and S. Saito, *J. Chem. Theory Comput.* **20**, 2798–2811 (2024).（Direct MaxFlux）
- S.-i. Koda and S. Saito, *J. Chem. Theory Comput.* **20**, 7176–7187 (2024); **21**, 3513–3522 (2025).（FB-ENM、CFB-ENM）
- E. D. Hermes, K. Sargsyan, H. N. Najm, J. Zádor, *J. Chem. Theory Comput.* **18**, 6974–6988 (2022).（Sella）

DFT の一点計算（ColabUMA Opt、TS、Workflow、ColabDFT）を使った場合は、次も引用してください。

- PySCF（Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020)）
- GPU4PySCF（X. Wu *et al.*, arXiv:2404.09452 (2024)）

ColabChem の開発には AI コーディング支援（Claude Code）を使いました。

## ライセンス

- ノートと共通コード（`colabuma/`）：[MIT License](LICENSE)
- UMAのモデル重み：[FAIR Chemistry License v1](https://huggingface.co/facebook/UMA) に従います。重みはこのリポジトリには含まれず、Hugging Face で各自がライセンスに同意してからダウンロードします

### 実行時にインストールするソフトウェア

このリポジトリには下のソフトウェアを含めていません。ノートを実行したときに PyPI または Ubuntu のパッケージ（apt）から Colab のセッションにインストールし、それぞれのライセンスに従って使います。

| ソフトウェア | 使うノート | ライセンス |
|---|---|---|
| [fairchem-core](https://github.com/facebookresearch/fairchem) | ColabDFT 以外 | MIT |
| [ASE](https://gitlab.com/ase/ase) | すべて | LGPL-2.1-or-later |
| [py3Dmol](https://github.com/3dmol/3Dmol.js) | すべて | MIT |
| [PyDMF](https://github.com/shin1koda/dmf)（`pydmf`） | TS、Workflow | MIT |
| [Sella](https://github.com/zadorlab/sella) | TS、Workflow | LGPL-3.0 |
| [cyipopt](https://github.com/mechmotum/cyipopt) | TS、Workflow | EPL-2.0 |
| [Ipopt](https://github.com/coin-or/Ipopt)（apt の `coinor-libipopt-dev`、3.11 系） | TS、Workflow | CPL-1.0（3.13 以降は EPL-2.0） |
| MUMPS（Ipopt が使う線形ソルバー。apt で一緒に入る） | TS、Workflow | CeCILL-C |
| [GPU4PySCF](https://github.com/pyscf/gpu4pyscf)（`gpu4pyscf-cuda12x`、`gpu4pyscf-libxc-cuda12x`） | Opt、TS、Workflow（DFT）、ColabDFT | Apache-2.0 |
| [CuPy](https://cupy.dev/)（`cupy-cuda12x`） | Opt、TS、Workflow（DFT）、ColabDFT | MIT |
| [cuTENSOR](https://developer.nvidia.com/cutensor)（`cutensor-cu12`） | Opt、TS、Workflow（DFT）、ColabDFT | NVIDIA のプロプライエタリライセンス |
| [PySCF](https://github.com/pyscf/pyscf) | Opt、TS、Workflow（DFT）、ColabDFT | Apache-2.0 |
| [pyscf-dispersion](https://github.com/pyscf/dispersion) | Opt、TS、Workflow（DFT）、ColabDFT | Apache-2.0 |
| [RDKit](https://github.com/rdkit/rdkit) | ColabDFT ESP・MO（入力を「draw」にしたときだけ） | BSD-3-Clause |
| [JSME](https://jsme-editor.github.io/)（構造エディタ。インストールせず、表示するときにブラウザが jsDelivr から読み込む） | ColabDFT ESP・MO（入力を「draw」にしたときだけ） | BSD-3-Clause |

ここに書いたライセンスは、2026年11月時点で各パッケージの配布元（PyPI と Ubuntu のパッケージ）に書かれているものです。ほかにも、これらが依存するライブラリ（PyTorch、NumPy、SciPy など）が一緒にインストールされます。正確な条件は、それぞれの配布元で確認してください。
