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
    # fairchem が一緒に入れるもの。UMA を使わない ColabDFT では別に入れる
    "structure": {"apt": [], "pip": [("ase", "ase"), ("py3Dmol", "py3Dmol")]},
    # 描いた構造（JSME）から 3D の座標を作る。「描く」を選んだときだけ入れる
    "draw": {"apt": [], "pip": [("rdkit", "rdkit")]},
    # DMF（pydmf）は IPOPT を使う。cyipopt は pip のバイナリがないので、apt の Ipopt からビルドする
    "ts": {
        "apt": ["coinor-libipopt-dev", "liblapack-dev", "libblas-dev", "pkg-config"],
        "pip": [("cyipopt", "cyipopt==1.7.0"), ("pydmf", "pydmf==1.2.2"), ("sella", "sella==2.6.0")],
    },
    "dft": {
        "apt": [],
        # gpu4pyscf は CUDA 12 版を使う。CUDA 13 版の 1.9.1 には sm_80 以降の GPU コードしかなく、T4（sm_75）で動かない。
        # CuPy は 14 系にする。14 系は cuda-pathfinder で、Colab に入っている CUDA 12 のライブラリ
        # （site-packages/nvidia/cuda_nvrtc/lib など）を見つける。13 系は見つけられず libnvrtc.so.12 のエラーになる。
        # 14 系は numpy<2.6 なので、Colab の numpy を変えず、再起動もいらない。
        # cuTENSOR がないと gpu4pyscf は cupy.einsum で計算する（pip の依存関係では入らない）。
        # 版はこの CuPy の wheel がビルドされた版（cupy/.data/_wheel.json）に合わせる
        "pip": [("cupy-cuda12x", "cupy-cuda12x==14.2.0"), ("cutensor-cu12", "cutensor-cu12==2.4.1"),
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
        warn("No GPU was found, so the calculations run on the CPU. They will be much slower.",
             "Choose T4 GPU in Runtime → Change runtime type, then run all again.")


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
        warn("HF_TOKEN is not set in Colab's Secrets.",
             "Open the key icon (Secrets) in the left sidebar, add your token with the name HF_TOKEN, "
             "and turn on notebook access. For now you can also paste it in the box below.")
        from getpass import getpass
        token = getpass("Paste your Hugging Face access token and press Enter: ").strip()
    if not token:
        raise ColabUMAError("No Hugging Face access token was given.",
                            "Create a token with Read access at https://huggingface.co/settings/tokens and add it to Secrets as HF_TOKEN.",
                            ja="Hugging Faceのアクセストークンがありません。",
                            ja_hint="https://huggingface.co/settings/tokens でRead権限のトークンを作り、シークレットにHF_TOKENとして登録してください。")
    os.environ["HF_TOKEN"] = token


def _pip(args, what):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], capture_output=True, text=True)
    if r.returncode != 0:
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
        raise ColabUMAError(f"Installing {what} failed (pip's messages are shown above).", "Use Runtime → Restart session, then run all again.",
                            ja=f"{what}のインストールに失敗しました（上にpipのメッセージを表示しています）。",
                            ja_hint="「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")


def _pip_uninstall(pkgs, what):
    r = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *pkgs], capture_output=True, text=True)
    if r.returncode != 0:
        print("\n".join(r.stderr.strip().splitlines()[-15:]))
        raise ColabUMAError(f"Preparing to install {what} failed (pip's messages are shown above).", "Use Runtime → Restart session, then run all again.",
                            ja=f"{what}のインストールの準備に失敗しました（上にpipのメッセージを表示しています）。",
                            ja_hint="「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")


def _install_extra(name):
    spec = EXTRAS[name]
    keep, others = spec.get("replace", (None, []))
    others = [pkg for pkg in others if installed(pkg)]
    missing = [pin for pkg, pin in spec["pip"]
               if (others and pkg == keep) or installed(pkg) is None or ("==" in pin and installed(pkg) != pin.split("==")[1])]
    if not missing:
        return False
    label = {"structure": "structure reading (ASE)", "draw": "3D from drawings (RDKit)", "ts": "TS search (DMF, Sella)",
             "dft": "DFT (gpu4pyscf)"}[name]
    print(f"Installing the libraries for {label}. This takes 1–2 minutes…")
    if spec["apt"]:
        if shutil.which("apt-get") is None:
            raise ColabUMAError(f"Could not install the system libraries needed for {label} (apt-get is missing).",
                                "Run the notebook on Google Colab.",
                                ja=f"{label} に必要なシステムのライブラリを入れられませんでした（apt-get がありません）。",
                                ja_hint="Google Colab で実行してください。")
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        r = subprocess.run(["apt-get", "install", "-y", "-q", *spec["apt"]], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            subprocess.run(["apt-get", "update", "-q"], capture_output=True, env=env)
            r = subprocess.run(["apt-get", "install", "-y", "-q", *spec["apt"]], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print("\n".join(r.stderr.strip().splitlines()[-15:]))
            raise ColabUMAError(f"Could not install the system libraries needed for {label} ({' '.join(spec['apt'])}).",
                                "Wait a while and run it again.",
                                ja=f"{label} に必要なシステムのライブラリ（{' '.join(spec['apt'])}）を入れられませんでした。",
                                ja_hint="しばらく待ってからもう一度実行してください。")
    numpy_before = installed("numpy")
    if others:
        _pip_uninstall(others + ([keep] if installed(keep) else []), label)
    _pip(missing, label)
    # ノートは先に numpy を読み込んでいる。CuPy の条件で numpy の版が変わることがある
    if "cupy" in sys.modules or ("numpy" in sys.modules and installed("numpy") != numpy_before):
        raise ColabUMAError("Installation finished, but the session must restart to replace libraries that are already loaded.",
                            "Use Runtime → Restart session, then run all again.",
                            ja="インストールは完了しましたが、読み込み済みのライブラリを入れ替えるためセッションの再起動が必要です。",
                            ja_hint="「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")
    return True


def setup(extras=()):
    """ライブラリのインストール（済みならスキップ）と Hugging Face の認証。DEVICE を返す。"""
    global DEVICE, _auth_ok
    if installed("fairchem-core") != FAIRCHEM_VERSION or not installed("py3Dmol"):
        torch_loaded_before = "torch" in sys.modules
        print(f"Installing the libraries (fairchem-core {FAIRCHEM_VERSION}). This takes a few minutes…")
        _pip([f"fairchem-core=={FAIRCHEM_VERSION}", "py3Dmol"], "the libraries")
        if torch_loaded_before:
            raise ColabUMAError("Installation finished, but the session must restart to replace libraries that are already loaded.",
                                "Use Runtime → Restart session, then run all again.",
                                ja="インストールは完了しましたが、読み込み済みのライブラリを入れ替えるためセッションの再起動が必要です。",
                                ja_hint="「ランタイム → セッションを再起動」してから、もう一度すべて実行してください。")
        ok("Installation finished.")
    else:
        print(f"fairchem-core {FAIRCHEM_VERSION} is already installed. Skipping.")
    for name in extras:
        if _install_extra(name):
            ok("Additional libraries installed.")

    if not _auth_ok:
        _check_hf_access()
        _auth_ok = True
        ok("Hugging Face login and access to facebook/UMA confirmed.")

    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device_name())
    return DEVICE


def setup_dft(drawing=False):
    """UMA を使わない DFT だけのインストール。fairchem も Hugging Face の認証もいらない。DEVICE を返す。
    drawing なら、描いた構造を 3D にする RDKit も入れる。"""
    global DEVICE
    for name in ("structure", "dft") + (("draw",) if drawing else ()):
        if _install_extra(name):
            ok("Libraries installed.")
    DEVICE = "cuda" if has_gpu() else "cpu"
    print("Device:", device_name())
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
            raise ColabUMAError("The Hugging Face token is not valid.",
                                "Create a new token at https://huggingface.co/settings/tokens and update HF_TOKEN in Secrets.",
                                ja="Hugging Faceのトークンが無効です。",
                                ja_hint="https://huggingface.co/settings/tokens でトークンを作り直し、シークレットのHF_TOKENを更新してください。") from None
        raise ColabUMAError(f"Could not connect to Hugging Face ({e}).", "Wait a while and run it again.",
                            ja=f"Hugging Faceに接続できませんでした（{e}）。",
                            ja_hint="しばらく待ってからもう一度実行してください。") from None
    try:
        from huggingface_hub import auth_check
        auth_check("facebook/UMA")
    except ImportError:
        pass  # 古いhuggingface_hubでは確認を省略し、モデル読み込み時に判定する
    except GatedRepoError:
        raise ColabUMAError("You do not have access to facebook/UMA yet.",
                            "Request access and accept the license at https://huggingface.co/facebook/UMA, then run again once approved.",
                            ja="facebook/UMA を使う権限がまだありません。",
                            ja_hint="https://huggingface.co/facebook/UMA でアクセス申請してライセンスに同意し、承認されてからもう一度実行してください。") from None
    except RepositoryNotFoundError:
        raise ColabUMAError("The Hugging Face token is not valid.", "Create a new token and update HF_TOKEN in Secrets.",
                            ja="Hugging Faceのトークンが無効です。",
                            ja_hint="トークンを作り直し、シークレットのHF_TOKENを更新してください。") from None
    except HfHubHTTPError as e:
        raise ColabUMAError(f"Could not connect to Hugging Face ({e}).", "Wait a while and run it again.",
                            ja=f"Hugging Faceに接続できませんでした（{e}）。",
                            ja_hint="しばらく待ってからもう一度実行してください。") from None


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
        print(f"Loading {model_name} (the first download takes a few minutes)…")
        try:
            predictor = pretrained_mlip.get_predict_unit(model_name, device=DEVICE)
        except GatedRepoError:
            raise ColabUMAError("You do not have access to facebook/UMA yet.",
                                "Request access and accept the license at https://huggingface.co/facebook/UMA, then run again once approved.",
                                ja="facebook/UMA を使う権限がまだありません。",
                                ja_hint="https://huggingface.co/facebook/UMA でアクセス申請してライセンスに同意し、承認されてからもう一度実行してください。") from None
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                raise ColabUMAError("The model download was refused (no access).",
                                    "Request access at https://huggingface.co/facebook/UMA and check that the token is correct.",
                                    ja="モデルのダウンロードが拒否されました（アクセス権がありません）。",
                                    ja_hint="https://huggingface.co/facebook/UMA でアクセス申請し、トークンが正しいか確認してください。") from None
            raise ColabUMAError(f"Could not download the model ({e}).", "Wait a while and run it again.",
                                ja=f"モデルをダウンロードできませんでした（{e}）。",
                                ja_hint="しばらく待ってからもう一度実行してください。") from None
        except KeyError:
            raise ColabUMAError(f"The model \"{model_name}\" is not in this version of fairchem.", "Choose another model in the form.",
                                ja=f"モデル名「{model_name}」はこのバージョンのfairchemにありません。",
                                ja_hint="フォームで別のモデルを選んでください。") from None
        _model.update(key=key, predictor=predictor)
        ok(f"{model_name} loaded.")
    else:
        print(f"{model_name} is already loaded. Skipping.")
    return FAIRChemCalculator(_model["predictor"], task_name=task_name)


def _colabchem_commit():
    """共通コードを取ってきた git のコミット（短い形）。分からなければ空文字。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def software_lines():
    import platform

    import ase
    import numpy as np

    from . import __version__

    commit = _colabchem_commit()
    lines = [f"ColabChem {__version__}" + (f" (commit {commit})" if commit else ""),
             f"Python {platform.python_version()}", f"ase {ase.__version__}"]
    for pkg in ("fairchem-core", "pydmf", "sella", "cyipopt", "gpu4pyscf-cuda12x", "cupy-cuda12x", "cutensor-cu12", "pyscf", "rdkit"):
        if installed(pkg):
            lines.append(f"{pkg} {installed(pkg)}")
    try:
        import torch
        lines.append(f"torch {torch.__version__}")
    except ImportError:
        pass
    lines += [f"numpy {np.__version__}", f"Device: {device_name() if DEVICE else '-'}"]
    return lines
