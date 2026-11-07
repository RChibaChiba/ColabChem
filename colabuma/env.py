"""GPUとトークンの確認、ライブラリのインストール、Hugging Faceの認証、モデル読み込み。

モデルはこのモジュールの中に覚えておくので、同じセッションでノートのセルを何度実行しても
読み込みは1回で済む（ColabUMA_Workflow で Opt → Scan → TS と続けるときも同じ）。
"""
import gc
import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .ui import ColabUMAError, ok, warn

FAIRCHEM_VERSION = "2.25.0"

# 追加の機能ごとに必要なもの。apt はシステムのライブラリ、pip は (配布名, 指定) の組
EXTRAS = {
    # fairchem が一緒に入れるもの。UMA を使わない ColabSPE では別に入れる
    "structure": {"apt": [], "pip": [("ase", "ase"), ("py3Dmol", "py3Dmol")]},
    # DMF（pydmf）は IPOPT を使う。cyipopt は pip のバイナリがないので、apt の Ipopt からビルドする
    "ts": {
        "apt": ["coinor-libipopt-dev", "liblapack-dev", "libblas-dev", "pkg-config"],
        "pip": [("cyipopt", "cyipopt==1.7.0"), ("pydmf", "pydmf==1.2.2"), ("sella", "sella==2.6.0")],
    },
    "dft": {
        "apt": [],
        # cuTENSOR がないと gpu4pyscf は cupy.einsum で計算する。CuPy との版の組み合わせは
        # gpu4pyscf 1.9.1 の requirements.txt に合わせる（pip の依存関係では cuTENSOR は入らない）
        "pip": [("cupy-cuda12x", "cupy-cuda12x==13.4.1"), ("cutensor-cu12", "cutensor-cu12==2.2.0"),
                ("gpu4pyscf-cuda12x", "gpu4pyscf-cuda12x==1.9.1"), ("pyscf-dispersion", "pyscf-dispersion")],
        # (残す配布, 同じ cupy/ に入る別の配布)。共有のファイルが壊れるので、両方消してから入れ直す
        "replace": ("cupy-cuda12x", ["cupy", "cupy-cuda11x", "cupy-cuda13x"]),
    },
}

DEVICE = None
_auth_ok = False
_model = {"key": None, "predictor": None}


def installed(pkg):
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        return None


def has_gpu():
    return shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0


def check_gpu():
    # torch を読み込む前に nvidia-smi で調べる
    if not has_gpu():
        warn("GPUが見つからないため、CPUで計算します。計算はかなり遅くなります。",
             "メニューの「ランタイム → ランタイムのタイプを変更」で「T4 GPU」を選び、もう一度すべて実行してください。")


def ensure_token():
    # Hugging Faceのトークン（ノートに保存しない。検証はインストール後に行う）
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
    if not token:
        warn("ColabのシークレットにHF_TOKENが登録されていません。",
             "左サイドバーの鍵アイコン（シークレット）で、名前 HF_TOKEN・値にトークンを登録し、"
             "「ノートブックからのアクセス」をオンにしてください。今回は下の欄に貼り付けても続行できます。")
        from getpass import getpass
        token = getpass("Hugging Faceのアクセストークンを貼り付けてEnter: ").strip()
    if not token:
        raise ColabUMAError("Hugging Faceのアクセストークンがありません。",
                            "https://huggingface.co/settings/tokens でRead権限のトークンを作り、シークレットにHF_TOKENとして登録してください。")
    os.environ["HF_TOKEN"] = token


def _pip(args, what):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    if r.returncode != 0:
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
        raise ColabUMAError(f"{what}のインストールに失敗しました（上にpipのメッセージを表示しています）。",
                            "「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")


def _pip_uninstall(pkgs, what):
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *pkgs], capture_output=True, text=True)
    if r.returncode != 0:
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
        raise ColabUMAError(f"{what}のインストールの準備に失敗しました（上にpipのメッセージを表示しています）。",
                            "「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")


def _install_extra(name):
    spec = EXTRAS[name]
    keep, others = spec.get("replace", (None, []))
    others = [pkg for pkg in others if installed(pkg)]
    missing = [pin for pkg, pin in spec["pip"]
               if (others and pkg == keep) or installed(pkg) is None or ("==" in pin and installed(pkg) != pin.split("==")[1])]
    if not missing:
        return False
    label = {"structure": "構造の読み込み（ASE）", "ts": "遷移状態探索（DMF・Sella）", "dft": "DFT（gpu4pyscf）"}[name]
    print(f"{label}のライブラリをインストールしています。1〜2分かかります…")
    if spec["apt"]:
        if shutil.which("apt-get") is None:
            raise ColabUMAError(f"{label}に必要なシステムのライブラリを入れられませんでした（apt-get がありません）。",
                                "Google Colab で実行してください。")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        r = subprocess.run(["apt-get", "install", "-y", "-q", *spec["apt"]], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            subprocess.run(["apt-get", "update", "-q"], capture_output=True, env=env)
            r = subprocess.run(["apt-get", "install", "-y", "-q", *spec["apt"]], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("\n".join(r.stderr.strip().splitlines()[-15:]))
            raise ColabUMAError(f"{label}に必要なシステムのライブラリ（{' '.join(spec['apt'])}）を入れられませんでした。",
                                "しばらく待ってからもう一度実行してください。")
    numpy_before = installed("numpy")
    if others:
        _pip_uninstall(others + ([keep] if installed(keep) else []), label)
    _pip(missing, label)
    # ノートは先に numpy を読み込んでいる。CuPy の条件で numpy の版が変わることがある
    if "cupy" in sys.modules or ("numpy" in sys.modules and installed("numpy") != numpy_before):
        raise ColabUMAError("インストールは完了しましたが、読み込み済みのライブラリを入れ替えるためセッションの再起動が必要です。",
                            "「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")
    return True


def setup(extras=()):
    """ライブラリのインストール（済みならスキップ）と Hugging Face の認証。DEVICE を返す。"""
    global DEVICE, _auth_ok
    if installed("fairchem-core") != FAIRCHEM_VERSION or not installed("py3Dmol"):
        torch_loaded_before = "torch" in sys.modules
        print(f"ライブラリをインストールしています（fairchem-core {FAIRCHEM_VERSION}）。数分かかります…")
        _pip([f"fairchem-core=={FAIRCHEM_VERSION}", "py3Dmol"], "ライブラリ")
        if torch_loaded_before:
            raise ColabUMAError("インストールは完了しましたが、読み込み済みのライブラリを入れ替えるためセッションの再起動が必要です。",
                                "「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")
        ok("インストールが完了しました。")
    else:
        print(f"fairchem-core {FAIRCHEM_VERSION} はインストール済みです。スキップします。")
    for name in extras:
        if _install_extra(name):
            ok("追加のライブラリのインストールが完了しました。")

    if not _auth_ok:
        _check_hf_access()
        _auth_ok = True
        ok("Hugging Faceの認証とfacebook/UMAへのアクセス権を確認しました。")

    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("計算デバイス:", device_name())
    return DEVICE


def setup_dft():
    """UMA を使わない DFT だけのインストール。fairchem も Hugging Face の認証もいらない。DEVICE を返す。"""
    global DEVICE
    for name in ("structure", "dft"):
        if _install_extra(name):
            ok("ライブラリのインストールが完了しました。")
    DEVICE = "cuda" if has_gpu() else "cpu"
    print("計算デバイス:", device_name())
    return DEVICE


def device_name():
    if DEVICE != "cuda":
        return "CPU"
    r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "GPU"


def _check_hf_access():
    from huggingface_hub import whoami
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        whoami()
    except Exception as e:
        if getattr(getattr(e, "response", None), "status_code", None) == 401:
            raise ColabUMAError("Hugging Faceのトークンが無効です。",
                                "https://huggingface.co/settings/tokens でトークンを作り直し、シークレットのHF_TOKENを更新してください。") from None
        raise ColabUMAError(f"Hugging Faceに接続できませんでした（{e}）。",
                            "しばらく待ってからもう一度実行してください。") from None
    try:
        from huggingface_hub import auth_check
        auth_check("facebook/UMA")
    except ImportError:
        pass  # 古いhuggingface_hubでは確認を省略し、モデル読み込み時に判定する
    except GatedRepoError:
        raise ColabUMAError("facebook/UMA を使う権限がまだありません。",
                            "https://huggingface.co/facebook/UMA でアクセス申請してライセンスに同意し、承認されてからもう一度実行してください。") from None
    except RepositoryNotFoundError:
        raise ColabUMAError("Hugging Faceのトークンが無効です。",
                            "トークンを作り直し、シークレットのHF_TOKENを更新してください。") from None
    except HfHubHTTPError as e:
        raise ColabUMAError(f"Hugging Faceに接続できませんでした（{e}）。",
                            "しばらく待ってからもう一度実行してください。") from None


def load_calculator(model_name, task_name):
    """UMA の ASE calculator を返す。同じモデルなら読み込みは1回だけ。"""
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    key = (model_name, DEVICE)
    if _model["key"] != key:
        _model.update(key=None, predictor=None)
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"{model_name} を読み込んでいます（初回はダウンロードに数分かかります）…")
        try:
            predictor = pretrained_mlip.get_predict_unit(model_name, device=DEVICE)
        except GatedRepoError:
            raise ColabUMAError("facebook/UMA を使う権限がまだありません。",
                                "https://huggingface.co/facebook/UMA でアクセス申請してライセンスに同意し、承認されてからもう一度実行してください。") from None
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                raise ColabUMAError("モデルのダウンロードが拒否されました（アクセス権がありません）。",
                                    "https://huggingface.co/facebook/UMA でアクセス申請し、トークンが正しいか確認してください。") from None
            raise ColabUMAError(f"モデルをダウンロードできませんでした（{e}）。",
                                "しばらく待ってからもう一度実行してください。") from None
        except KeyError:
            raise ColabUMAError(f"モデル名「{model_name}」はこのバージョンのfairchemにありません。",
                                "フォームで別のモデルを選んでください。") from None
        _model.update(key=key, predictor=predictor)
        ok(f"{model_name} を読み込みました。")
    else:
        print(f"{model_name} は読み込み済みです。スキップします。")
    return FAIRChemCalculator(_model["predictor"], task_name=task_name)


def software_lines():
    import platform

    import ase
    import numpy as np

    lines = [f"Python {platform.python_version()}", f"ase {ase.__version__}"]
    for pkg in ("fairchem-core", "pydmf", "sella", "cyipopt", "gpu4pyscf-cuda12x", "cupy-cuda12x", "cutensor-cu12", "pyscf"):
        if installed(pkg):
            lines.append(f"{pkg} {installed(pkg)}")
    try:
        import torch
        lines.append(f"torch {torch.__version__}")
    except ImportError:
        pass
    lines += [f"numpy {np.__version__}", f"デバイス: {device_name() if DEVICE else '-'}"]
    return lines
