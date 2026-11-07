"""DFT の一点計算（gpu4pyscf。GPU で使えなければ CPU の PySCF）。

UMA で求めた構造や入力の構造で DFT のエネルギーを計算する。反応物・TS・生成物なら活性化エネルギーも比べる。
"""
import math
import time
from dataclasses import dataclass

from .ui import ColabUMAError, ProgressDisplay, bar_row, ok, warn

EV_PER_HARTREE = 27.211386245988
KCALMOL_PER_HARTREE = 627.5094740631

# フォームの表示名 → (PySCF の xc, 分散補正)
FUNCTIONALS = {
    "ωB97M-V": ("wb97m-v", None),
    "B3LYP-D3(BJ)": ("b3lyp", "d3bj"),
    "PBE0-D3(BJ)": ("pbe0", "d3bj"),
    "M06-2X": ("m062x", None),
    "r2SCAN-D4": ("r2scan", "d4"),
}
BASIS_SETS = ["def2-SVP", "def2-TZVP", "def2-TZVPD"]
MAX_Z = 86  # def2 基底関数があるのは H〜Rn

GB = 1e9
# UMA のノートで DFT 一点計算にかける時間の既定値（分）。全構造の合計
TIME_LIMIT_MIN = 60
# Colab の無料版でランタイムが続く最長時間の目安（時間）と、そのあとの保存に残しておく時間（分）
COLAB_MAX_HOURS = 12
SAVE_MARGIN_MIN = 15
# gpu4pyscf は 3中心積分（cderi）が空きの 40% に収まれば GPU に、収まらなければ CPU のメモリに置く
GPU_CDERI_FRACTION = 0.4
# cderi を CPU のメモリに置くとき、空きのこの割合を超えたら危ない（ほかの配列や Python の分も要る）
HOST_WARN_FRACTION = 0.3
HOST_DANGER_FRACTION = 0.7


@dataclass
class Job:
    """UMA を使わない計算（ColabDFT SPE / ESP）のフォームの値。"""
    out_prefix: str
    charge: int
    multiplicity: int
    functional: str
    basis: str

    def condition_lines(self):
        return [f"Charge: {self.charge}", f"Spin multiplicity: {self.multiplicity}",
                f"Functional: {self.functional}", f"Basis set: {self.basis}"]


def parse_method(functional, basis):
    key = functional.split("（")[0].split(" (")[0].strip()  # 「ωB97M-V (same as OMol25)」→ ωB97M-V
    if key not in FUNCTIONALS:
        raise ColabUMAError(f"The functional \"{functional}\" is not supported.", f"Choose one of {', '.join(FUNCTIONALS)}.",
                            ja=f"汎関数「{functional}」には対応していません。", ja_hint=f"{', '.join(FUNCTIONALS)} から選んでください。")
    if basis not in BASIS_SETS:
        raise ColabUMAError(f"The basis set \"{basis}\" is not supported.", f"Choose one of {', '.join(BASIS_SETS)}.",
                            ja=f"基底関数「{basis}」には対応していません。", ja_hint=f"{', '.join(BASIS_SETS)} から選んでください。")
    return key, basis


class DFTSkipped(Exception):
    """DFT を最後まで計算できなかったとき（メモリ不足・時間切れなど）。done は計算が終わった構造の結果。"""
    def __init__(self, reason, hint="", done=None):
        super().__init__(reason)
        self.reason = reason
        self.hint = hint
        self.done = done or []


class _TimeUp(Exception):
    pass


def _backend():
    try:
        import cupy  # noqa: F401
        from gpu4pyscf import dft as gdft

        (cupy.ones(1) * 2.0).sum()  # GPU が実際に使えるか。カーネルのコンパイル（NVRTC）まで試す
        return "gpu4pyscf", gdft
    except Exception:
        from pyscf import dft as cdft

        return "pyscf (CPU)", cdft


def _make_mol(atoms, charge, multiplicity, basis):
    import pyscf

    return pyscf.M(atom=[(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                   basis=basis, ecp=basis if atoms.numbers.max() > 36 else None,  # def2 の ECP は Rb 以降
                   charge=int(charge), spin=int(multiplicity) - 1, unit="Angstrom", verbose=0)


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else a  # GPU の配列を NumPy に


def single_point(atoms, charge, multiplicity, functional, basis, max_cycle=100, keep_density=False, deadline=None,
                 keep_orbitals=False):
    """1構造の DFT エネルギー（Hartree）。dict を返す。keep_density なら mol と全電子の密度行列（NumPy）も入れる。
    keep_orbitals なら mol と分子軌道（mo_coeff, mo_energy, mo_occ。UKS は先頭の軸が α, β）も入れる。

    deadline（time.time() の値）を過ぎたら、SCF の反復の区切りで _TimeUp を投げて止める。
    """
    name, mod = _backend()
    xc, disp = FUNCTIONALS[functional]
    mol = _make_mol(atoms, charge, multiplicity, basis)
    mf = (mod.RKS(mol, xc=xc) if multiplicity == 1 else mod.UKS(mol, xc=xc)).density_fit()
    if disp:
        mf.disp = disp
    mf.max_cycle = max_cycle
    if deadline is not None:
        def _check_time(envs):
            if time.time() > deadline:
                raise _TimeUp()
        mf.callback = _check_time
    t0 = time.time()
    try:
        e = float(mf.kernel())
        out = {"energy": e, "converged": bool(mf.converged), "seconds": time.time() - t0, "backend": name,
               "nao": int(mol.nao)}
        if keep_density:
            dm = _to_numpy(mf.make_rdm1())
            dm = dm.sum(axis=0) if dm.ndim == 3 else dm  # UKS は α + β
            out.update(mol=mol, dm=(dm + dm.T) / 2)
        if keep_orbitals:
            out.update(mol=mol, mo_coeff=_to_numpy(mf.mo_coeff), mo_energy=_to_numpy(mf.mo_energy),
                       mo_occ=_to_numpy(mf.mo_occ))
    finally:
        del mf
        _free_gpu()
    return out


def _free_gpu():
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _gb(n):
    return f"{n / GB:.1f} GB" if n >= 0.95 * GB else f"{n / 1e6:.0f} MB"


def _host_available():
    """CPU のメモリの空き（バイト）。分からなければ None。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _gpu_available():
    """GPU のメモリの空き（バイト。CuPy が確保して使っていない分も含む）。GPU がなければ None。"""
    try:
        import cupy

        pool = cupy.get_default_memory_pool()
        return cupy.cuda.runtime.memGetInfo()[0] + pool.total_bytes() - pool.used_bytes()
    except Exception:
        return None


def _is_out_of_memory(e):
    if isinstance(e, MemoryError):
        return True
    text = f"{type(e).__name__} {e}".lower()
    return any(k in text for k in ("outofmemory", "out of memory", "memoryallocation", "not enough gpu memory"))


def estimate_memory(points, charge, multiplicity, basis):
    """いちばん大きい構造の基底関数の数と、密度フィッティングの 3中心積分（cderi）の大きさ（バイト）。

    cderi は (補助基底の数) × nao(nao+1)/2 の倍精度の配列。gpu4pyscf は小さい積分を省くので、実際はこれより小さい
    ことが多い（大きな分子ほど差が大きい）。
    """
    from pyscf import df

    atoms = max((a for _, a in points), key=len)
    mol = _make_mol(atoms, charge, multiplicity, basis)
    auxmol = df.addons.make_auxmol(mol, df.addons.make_auxbasis(mol))
    nao, naux = int(mol.nao), int(auxmol.nao)
    return {"nao": nao, "naux": naux, "cderi": naux * nao * (nao + 1) // 2 * 8, "n_atoms": len(atoms)}


def check_memory(points, charge, multiplicity, basis, quiet=False):
    """DFT のメモリが足りそうかを見積もる。("ok" | "warn" | "danger", 説明, 対処法, 見積もり) を返す。

    gpu4pyscf は cderi を GPU に置けなければ CPU のメモリに置く。CPU のメモリまで足りなくなると、Colab では
    エラーにならずにカーネルごと落ちて、それまでの結果も消えるので、その手前で止められるようにする。
    quiet でなければ、"warn" と "danger" のときに注意を表示する。
    """
    est = estimate_memory(points, charge, multiplicity, basis)
    need = est["cderi"]
    gpu = _gpu_available() if _backend()[0].startswith("gpu") else None
    host = _host_available()
    size = (f"{est['n_atoms']} atoms, {est['nao']} basis functions and {est['naux']} auxiliary functions: "
            f"the three-center integrals take up to about {_gb(need)}")
    free = ", ".join(x for x in (f"GPU {_gb(gpu)}" if gpu is not None else "",
                                 f"CPU {_gb(host)}" if host is not None else "") if x)
    hint = "Use a smaller basis set (def2-SVP), or a runtime with more memory such as Colab Pro high-RAM or an A100."
    if gpu is not None and need < GPU_CDERI_FRACTION * gpu:
        level, msg = "ok", f"{size}. They should fit in GPU memory (free: {free})."
    elif host is None or need < HOST_WARN_FRACTION * host:
        level = "ok" if gpu is None else "warn"
        msg = (f"{size}. They do not fit in GPU memory, so they go to CPU memory (free: {free}). "
               "The calculation will be slower." if gpu is not None else f"{size} (free: {free}).")
    elif need < HOST_DANGER_FRACTION * host:
        level, msg = "warn", (f"{size}. This is close to the free memory ({free}). "
                              "If memory runs out during the calculation, the Colab runtime can crash.")
    else:
        level, msg = "danger", (f"{size}. This is likely to exceed the free memory ({free}). "
                                "Running it is likely to crash the Colab runtime.")
    if gpu is None and level == "danger":
        # CPU の PySCF は、cderi がメモリに収まらなければディスクに置くので落ちない。ただしとても遅い
        level, msg = "warn", f"{size}. They do not fit in CPU memory, so the disk is used (free: {free}). This is very slow."
    if not quiet and level != "ok":
        warn(f"DFT memory: {msg}", hint)
    return level, msg, hint, est


def time_budget(limit_min):
    """DFT にかけてよい秒数。limit_min が 0 以下なら上限なし。Colab ではランタイムの最長時間の目安も超えない。"""
    budget = limit_min * 60 if limit_min and limit_min > 0 else math.inf
    try:
        import google.colab  # noqa: F401

        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])  # Colab ではランタイムの VM が起動してからの秒数
        budget = min(budget, COLAB_MAX_HOURS * 3600 - uptime - SAVE_MARGIN_MIN * 60)
    except (ImportError, OSError, ValueError):
        pass
    return max(budget, 0.0)


def run(points, charge, multiplicity, functional, basis, keep_density=False, time_limit_min=0, fallback=False,
        keep_orbitals=False):
    """points は [(名前, atoms), ...]。各構造の一点計算の結果を返す。

    time_limit_min：全構造の合計の時間の上限（分）。0 なら上限なし（Colab ではランタイムの最長時間の目安まで）。
    fallback：メモリ不足・時間切れのとき、ColabUMAError ではなく DFTSkipped を投げる（UMA のノートで DFT だけ飛ばす）。
    メモリが明らかに足りない見込みのときも、fallback なら計算を始めずに飛ばす。そうでなければ注意だけ出して計算する。
    """
    functional, basis = parse_method(functional, basis)
    if max(int(a.numbers.max()) for _, a in points) > MAX_Z:
        raise ColabUMAError("The structure has an element without a def2 basis set.", "Supported elements are H–Rn (Z = 1–86).",
                            ja="def2 基底関数がない元素が含まれています。", ja_hint="扱えるのは H〜Rn（原子番号1〜86）です。")
    name, _ = _backend()
    if not name.startswith("gpu"):
        warn("gpu4pyscf is not available, so DFT runs on the CPU with PySCF. This takes much longer.",
             "Check that T4 GPU is selected as the runtime type.")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # UMA が使った GPU メモリを空ける
    except ImportError:
        pass
    _free_gpu()

    def stop(reason, hint, done, ja="", ja_hint=""):
        if fallback:
            raise DFTSkipped(reason, hint, done)
        raise ColabUMAError(reason, hint, ja=ja, ja_hint=ja_hint)

    level, msg, hint, est = check_memory(points, charge, multiplicity, basis, quiet=True)
    if level == "danger" and fallback:  # UMA のノートでは、落ちて UMA の結果まで消えないように DFT を飛ばす
        stop(f"DFT is expected to run out of memory. {msg}", hint, [])
    if level != "ok":  # DFT だけのノートでは、見積もりは上限なので注意だけして計算してみる
        warn(f"DFT memory: {msg}", hint)

    t_start = time.time()
    budget = time_budget(time_limit_min)
    deadline = t_start + budget if math.isfinite(budget) else None
    limit_text = f", time limit {budget / 60:.0f} min" if deadline else ""
    print(f"DFT single point: {functional}/{basis} ({name}, density fitting, {est['nao']} basis functions{limit_text})")
    time_hint = ("Raise the DFT time limit in the form, or use a smaller basis set. "
                 "A free Colab runtime ends after at most about 12 hours.")
    time_hint_ja = ("フォームの DFT の時間の上限を長くするか、基底関数を小さくしてください。"
                    "Colab の無料版は、ランタイムが最長でおよそ12時間で切れます。")
    progress = ProgressDisplay()
    results = []
    for k, (label, atoms) in enumerate(points):
        if deadline is not None:
            left = deadline - time.time()
            slowest = max((r["seconds"] for r in results), default=0.0)
            if left <= 0 or slowest > left:  # 前の構造と同じくらいかかるなら、間に合わない
                stop(f"DFT of {label} is not expected to finish within the time limit ({budget / 60:.0f} min).",
                     time_hint, results, f"DFT の時間の上限（{budget / 60:.0f} 分）までに {label} の計算が終わらない見込みです。",
                     time_hint_ja)
        progress.show([bar_row("DFT", k / len(points), f"computing {label} ({k}/{len(points)} done)")], force=True)
        try:
            r = single_point(atoms, charge, multiplicity, functional, basis, keep_density=keep_density,
                             deadline=deadline, keep_orbitals=keep_orbitals)
        except _TimeUp:
            stop(f"The DFT time limit ({budget / 60:.0f} min) was reached, so DFT of {label} was stopped.", time_hint, results,
                 f"DFT の時間の上限（{budget / 60:.0f} 分）を過ぎたので、{label} の計算を止めました。", time_hint_ja)
        except Exception as e:
            if _is_out_of_memory(e):
                text = f"{type(e).__name__} {e}".lower()
                gpu_mem = any(k in text for k in ("outofmemory", "gpu", "cuda"))
                stop(f"DFT ran out of {'GPU ' if gpu_mem else ''}memory ({label}).", hint, results,
                     f"DFT の計算で {'GPU の' if gpu_mem else ''}メモリが足りなくなりました（{label}）。",
                     "基底関数を小さくする（def2-SVP）か、Colab Pro の大容量 RAM・A100 などのランタイムを使ってください。")
            raise
        r.update(label=label, atoms=atoms)
        results.append(r)
        print(f"  {label}: E = {r['energy']:.8f} Ha  {'converged' if r['converged'] else 'SCF NOT converged'}"
              f" ({r['nao']} basis functions, {r['seconds']:.1f} s)")
        if not r["converged"]:
            warn(f"The SCF of {label} did not converge. This energy is not reliable.")
    progress.show([bar_row("DFT", 1.0, f"{len(points)}/{len(points)} done")], force=True)
    ok("DFT single points finished.")
    return {"functional": functional, "basis": basis, "backend": name, "points": results}


def run_optional(points, charge, multiplicity, functional, basis, uma_rows, time_limit_min=TIME_LIMIT_MIN,
                 geometry="UMA structures"):
    """UMA のノートの DFT 一点計算。メモリ不足・時間切れ・計算のエラーでは DFT だけを飛ばし、UMA の結果の保存まで
    進められるようにする。(files, log) を返す（飛ばしたときも、終わった構造の分は入れる）。"""
    try:
        res = run(points, charge, multiplicity, functional, basis, time_limit_min=time_limit_min, fallback=True)
        return report(res, uma_rows, geometry)
    except DFTSkipped as e:
        reason, hint, done = e.reason, e.hint, e.done
    except ColabUMAError as e:
        reason, hint, done = e.message, e.hint, []
    except Exception as e:  # 予想していないエラーでも、UMA の結果は保存する
        reason, hint, done = f"An error occurred in DFT ({type(e).__name__}: {str(e)[:200]}).", "", []
    _free_gpu()
    warn(f"DFT single points were skipped. {reason}", (hint + " " if hint else "") + "The UMA results are saved next.")
    files, log = {}, []
    if done:
        print("DFT energies of the structures that finished:")
        files, log = report({"functional": parse_method(functional, basis)[0], "basis": basis,
                             "backend": done[0]["backend"], "points": done}, uma_rows, geometry)
        log = log[:-1]  # 最後の空行
    log = (log or [f"[DFT single point] {functional}/{basis}"]) + [f"Skipped: {reason}", ""]
    return files, log


def report(dft_res, uma_rows, geometry="UMA structures"):
    """UMA と DFT のエネルギー差を並べて表示する。geometry は構造の説明。(files, log) を返す。"""
    import pandas as pd
    from IPython.display import display

    pts = dft_res["points"]
    uma = dict(uma_rows) if uma_rows else {}
    base = pts[0]["energy"]
    ubase = uma.get(pts[0]["label"])
    rows = []
    for p in pts:
        row = {"": p["label"], "DFT (Hartree)": round(p["energy"], 8)}
        if len(pts) > 1:  # 1構造だけなら比べる相手がない
            row[f"ΔE DFT (vs {pts[0]['label']}, kcal/mol)"] = round((p["energy"] - base) * KCALMOL_PER_HARTREE, 2)
        if len(pts) > 1 and p["label"] in uma and ubase is not None:
            row["ΔE UMA (kcal/mol)"] = round((uma[p["label"]] - ubase) / EV_PER_HARTREE * KCALMOL_PER_HARTREE, 2)
        row["SCF converged"] = "yes" if p["converged"] else "no"
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"DFT single points ({dft_res['functional']}/{dft_res['basis']}, {geometry}):")
    display(df.style.hide(axis="index"))
    log = [f"[DFT single point] {dft_res['functional']}/{dft_res['basis']} ({dft_res['backend']}, density fitting, {geometry})"]
    log += [f"{p['label']}: E = {p['energy']:.10f} Ha  converged = {p['converged']}  nao = {p['nao']}  "
            f"{p['seconds']:.1f} s" for p in pts]
    if len(pts) == 3 and pts[1]["label"] == "TS":
        e_r, e_ts, e_p = (p["energy"] for p in pts)
        fwd = (e_ts - e_r) * KCALMOL_PER_HARTREE
        rev = (e_ts - e_p) * KCALMOL_PER_HARTREE
        rxn = (e_p - e_r) * KCALMOL_PER_HARTREE
        print(f"Activation energy (DFT): forward {fwd:.2f} kcal/mol, reverse {rev:.2f} kcal/mol, reaction energy {rxn:.2f} kcal/mol")
        log += [f"Activation energy, forward: {fwd:.4f} kcal/mol", f"Activation energy, reverse: {rev:.4f} kcal/mol",
                f"Reaction energy: {rxn:.4f} kcal/mol"]
    log.append("")
    files = {"dft.csv": df.to_csv(index=False)}
    return files, log
