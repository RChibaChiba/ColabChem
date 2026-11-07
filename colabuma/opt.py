"""構造最適化と振動解析。Opt・Scan・TS のノートで共通に使う。"""
import time
from dataclasses import dataclass

import numpy as np

from .ui import ColabUMAError, ProgressDisplay, bar_row, convergence_fraction, warn

FMAX_PRESETS = {"loose (0.05)": 0.05, "normal (0.02)": 0.02, "tight (0.001)": 0.001}
CUSTOM = "custom"
IMAG_THRESHOLD = 20.0  # cm^-1。これより小さい虚振動は数値誤差とみなす


def convergence_key(convergence):
    """フォームの収束判定を FMAX_PRESETS のキーか CUSTOM にする（以前の日本語・全角括弧の値も受け付ける）。"""
    key = convergence.strip().replace("（", " (").replace("）", ")").replace("  ", " ")
    key = CUSTOM if key in ("数値指定", CUSTOM) else key
    if key != CUSTOM and key not in FMAX_PRESETS:
        raise ColabUMAError(f"The convergence setting \"{convergence}\" is not supported.",
                            f"Choose one of {', '.join(FMAX_PRESETS)} or {CUSTOM}.",
                            ja=f"収束判定「{convergence}」には対応していません。",
                            ja_hint=f"{', '.join(FMAX_PRESETS)}、{CUSTOM} から選んでください。")
    return key


def fmax_from_form(convergence, fmax_custom):
    key = convergence_key(convergence)
    return fmax_custom if key == CUSTOM else FMAX_PRESETS[key]


@dataclass
class Job:
    """フォームの値のうち、どの計算でも共通に使うもの。"""
    out_prefix: str
    charge: int
    multiplicity: int
    model_name: str
    task_name: str
    optimizer_name: str = "LBFGS"
    fmax: float = 0.05
    max_steps: int = 500
    convergence: str = ""

    def condition_lines(self):
        return [f"Charge: {self.charge}", f"Spin multiplicity: {self.multiplicity}",
                f"Model: {self.model_name}", f"Task: {self.task_name}",
                f"Optimizer: {self.optimizer_name}",
                f"fmax: {self.fmax} eV/Å" + (f" ({convergence_key(self.convergence)})" if self.convergence else ""),
                f"Max steps: {self.max_steps}"]


def _oom_error():
    try:
        import torch
        return torch.cuda.OutOfMemoryError
    except ImportError:
        return MemoryError


def fmax_of(forces):
    return float(np.sqrt((forces ** 2).sum(axis=1).max()))


def _optimizer(name, atoms):
    from ase.optimize import BFGS, FIRE, LBFGS

    return {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}[name](atoms, logfile=None)


def run(atoms, calc, job, label, *, progress=None, extra_rows=None, optimizer=None, fmax=None, max_steps=None,
        quiet=False):
    """atoms をその場で最適化して結果の dict を返す。

    optimizer を渡さなければ job.optimizer_name を使う（Sella などを使うときは optimizer に作る関数を渡す）。
    extra_rows() はプログレス表示の上に足す行（スキャンの進み具合など）。
    """
    fmax = job.fmax if fmax is None else fmax
    max_steps = job.max_steps if max_steps is None else max_steps
    progress = progress or ProgressDisplay()
    atoms.calc = calc
    hist = {"energy": [], "fmax": [], "frames": [], "forces": []}
    opt = optimizer(atoms) if optimizer else _optimizer(job.optimizer_name, atoms)

    def rows():
        f = hist["fmax"]
        row = bar_row("Converge", convergence_fraction(f[0], f[-1], fmax),
                      f"{label}　fmax {f[-1]:.4f} → {fmax} eV/Å　step {len(f) - 1}/{max_steps}")
        return (extra_rows() if extra_rows else []) + [row]

    def record():
        e = float(atoms.get_potential_energy())
        forces = atoms.get_forces()  # 拘束があれば、拘束方向の成分を除いた力
        hist["energy"].append(e)
        hist["fmax"].append(fmax_of(forces))
        hist["frames"].append(atoms.positions.copy())
        hist["forces"].append(forces.copy())
        progress.show(rows())

    opt.attach(record, interval=1)
    t0 = time.time()
    try:
        opt.run(fmax=fmax, steps=max_steps)
    except _oom_error():
        raise ColabUMAError("The GPU ran out of memory.", "Use a smaller model such as uma-s-1p1, or fewer atoms.",
                            ja="GPUのメモリが足りません。",
                            ja_hint="モデルを uma-s-1p1 などの小さいものにするか、原子数を減らしてください。") from None
    if not hist["energy"]:  # 最初の構造ですでに収束していて記録が呼ばれなかったとき
        record()
    progress.show(rows(), force=True)
    r = {"label": label, "atoms": atoms, "energy": hist["energy"][-1], "fmax": hist["fmax"][-1], "steps": opt.nsteps,
         "converged": hist["fmax"][-1] < fmax, "hist": hist, "seconds": time.time() - t0, "max_steps": max_steps,
         "fmax_target": fmax}
    if not quiet:
        print(f"  {label}: step {r['steps']:4d}  E = {r['energy']:.6f} eV  fmax = {r['fmax']:.4f} eV/Å"
              f"  {'converged' if r['converged'] else 'NOT converged'} ({r['seconds']:.1f} s)")
    return r


def optimize(atoms, calc, job, label, **kw):
    """拘束なしの最適化。入力の atoms は変えない。"""
    atoms = atoms.copy()
    atoms.set_constraint()
    return run(atoms, calc, job, label, **kw)


def warn_unconverged(r):
    if not r["converged"]:
        warn(f"{r['label']} did not converge within the maximum number of steps ({r['max_steps']}). "
             "The result is the last structure reached.",
             "Increase the maximum steps or loosen fmax, then run again.")


# --- 振動解析 ---
def numerical_hessian(atoms, calc, delta=0.01, label="Hessian", progress=None):
    """中心差分のヘシアン（eV/Å^2、形は (3N, 3N)）。力の計算は 6N 回。"""
    a = atoms.copy()
    a.set_constraint()
    a.calc = calc
    x0 = a.positions.copy()
    n = 3 * len(a)
    H = np.empty((n, n))
    progress = progress or ProgressDisplay()
    for k in range(n):
        f = []
        for sign in (1, -1):
            x = x0.copy()
            x.flat[k] += sign * delta
            a.positions = x
            f.append(a.get_forces().ravel())
        H[k] = -(f[0] - f[1]) / (2 * delta)
        progress.show([bar_row(label, (k + 1) / n, f"{2 * (k + 1)}/{2 * n} force calls")], force=k == n - 1)
    a.positions = x0
    return 0.5 * (H + H.T)


def _external_basis(atoms, mass_weighted_sqrt):
    # 質量加重座標での並進・回転のベクトル（正規直交化したもの）
    x = atoms.positions - atoms.get_center_of_mass()
    n = len(atoms)
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = 1.0
        vecs.append(t)
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        vecs.append(np.cross(e, x))
    D = np.array([(v * mass_weighted_sqrt[:, None]).ravel() for v in vecs]).T
    u, s, _ = np.linalg.svd(D, full_matrices=False)
    return u[:, s > 1e-6 * s.max()]


def frequencies(atoms, H):
    """ヘシアンから振動数（cm^-1、虚振動は負）と基準振動を求める。並進・回転は射影で取り除く。"""
    import ase.units as u

    m = atoms.get_masses()
    sq = np.sqrt(np.repeat(m, 3))
    Hm = H / sq[:, None] / sq[None, :]
    s = u._hbar * 1e10 / np.sqrt(u._e * u._amu) / u.invcm  # sqrt(eV/Å^2/amu) → cm^-1

    def to_cm(lam):
        return np.sign(lam) * s * np.sqrt(np.abs(lam))

    D = _external_basis(atoms, np.sqrt(m))
    n_ext = D.shape[1]
    # 並進・回転の補空間で対角化する
    full = np.linalg.svd(D, full_matrices=True)[0]
    Q = full[:, n_ext:]
    lam, vec = np.linalg.eigh(Q.T @ Hm @ Q)
    mw_modes = (Q @ vec).T  # 質量加重座標で規格化されたモード
    cart = mw_modes / sq[None, :]
    red_mass = 1.0 / (cart ** 2).sum(axis=1)
    cart_norm = cart / np.linalg.norm(cart, axis=1)[:, None]
    freqs = to_cm(lam)
    k_mdyne = lam * red_mass * (u._e * 1e20 / 1e2)  # eV/Å^2 → mdyne/Å
    all_freqs = np.sort(to_cm(np.linalg.eigvalsh(Hm)))
    n_imag = int((freqs < -IMAG_THRESHOLD).sum())
    return {"freqs": freqs, "modes": cart_norm.reshape(len(freqs), len(atoms), 3), "red_mass": red_mass,
            "force_const": k_mdyne, "n_imag": n_imag, "n_ext": n_ext, "all": all_freqs}


def vibrations(atoms, calc, label="Vibrational analysis"):
    if len(atoms) == 1:
        warn("There is only one atom, so no vibrational analysis is done.")
        return None
    print(f"{label} ({6 * len(atoms)} force calls)…")
    H = numerical_hessian(atoms, calc, label=label)
    r = frequencies(atoms, H)
    r["hessian"] = H
    return r


def freq_text(vib):
    lines = [f"# Frequencies (cm^-1). Negative values are imaginary. {vib['n_ext']} translations/rotations removed",
             f"# Number of imaginary frequencies (|nu| >= {IMAG_THRESHOLD:.0f} cm^-1): {vib['n_imag']}"]
    lines += [f"{i + 1:4d} {v:12.2f}" for i, v in enumerate(vib["freqs"])]
    lines += ["", "# All modes (before projecting out translations and rotations)"] + [f"{i + 1:4d} {v:12.2f}" for i, v in enumerate(vib["all"])]
    return "\n".join(lines) + "\n"



# --- 結果の表示（Opt ノート） ---
def report(best, vib, job, label="Optimization"):
    """3D表示・エネルギー・収束グラフ・振動を表示し、(出力ファイルの dict, ログの行) を返す。"""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    from . import gaussian
    from .structure import show_mode, show_structure, xyz_block
    from .ui import ok

    eh = gaussian.EV_PER_HARTREE
    atoms = best["atoms"]
    symbols = atoms.get_chemical_symbols()
    E, E_h = best["energy"], best["energy"] / eh
    cond = (f"model={job.model_name} task={job.task_name} charge={job.charge} multiplicity={job.multiplicity} "
            f"optimizer={job.optimizer_name} fmax_target={job.fmax}")
    opt_xyz = xyz_block(symbols, atoms.positions,
                        f"energy_eV={E:.8f} energy_Ha={E_h:.10f} fmax={best['fmax']:.5f} converged={best['converged']} {cond}")
    show_structure(opt_xyz)

    display(pd.DataFrame({"Value": [
        f"{atoms.get_chemical_formula()} ({len(atoms)} atoms)", f"{E:.6f} eV", f"{E_h:.8f} Hartree",
        f"{best['fmax']:.4f} eV/Å", str(best["steps"]), "yes" if best["converged"] else "no",
    ]}, index=["Formula", "Final energy", "Final energy", "Final fmax", "Steps", "Converged"]))
    warn_unconverged(best)

    # --- 収束グラフ（Colabは日本語フォントがないため軸ラベルは英語） ---
    h = best["hist"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax1.plot(h["energy"], marker=".", color="#1f77b4")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("Energy")
    ax2.semilogy(h["fmax"], marker=".", color="#d62728")
    ax2.axhline(best["fmax_target"], ls="--", color="gray", label=f"fmax = {best['fmax_target']}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Max force (eV/Å)")
    ax2.set_title("Max force")
    ax2.legend()
    fig.tight_layout()
    plt.show()

    files = {"opt.xyz": opt_xyz, "opt.log": gaussian.opt_log(best, job, atoms.numbers),
             "traj.xyz": "".join(xyz_block(symbols, pos, f"step={i} energy_eV={e:.8f} fmax={fm:.5f}")
                                 for i, (pos, e, fm) in enumerate(zip(h["frames"], h["energy"], h["fmax"])))}
    log = [f"[{label} result]",
           f"E = {E:.8f} eV ({E_h:.10f} Ha)  fmax = {best['fmax']:.5f}  steps = {best['steps']}  converged = {best['converged']}",
           f"Vibrational analysis: {'on' if vib is not None else 'off'}",
           "", f"{'step':>5s} {'energy_eV':>18s} {'fmax_eV/A':>12s}"]
    log += [f"{i:5d} {e:18.8f} {fm:12.5f}" for i, (e, fm) in enumerate(zip(h["energy"], h["fmax"]))]

    # --- 振動計算の結果 ---
    if vib is not None:
        print(f"Imaginary frequencies: {vib['n_imag']} (those below {IMAG_THRESHOLD:.0f} cm⁻¹ are treated as numerical noise)")
        print(f"Lowest frequency: {vib['freqs'].min():.1f} cm⁻¹ (negative values are imaginary)")
        if vib["n_imag"] > 0:
            warn(f"There are {vib['n_imag']} imaginary frequencies. This may be a saddle point, not a minimum.",
                 "Displace the structure slightly and optimize again.")
            print("Mode of the largest imaginary frequency:")
            show_mode(atoms, vib["modes"][0])
        else:
            ok("No imaginary frequencies. The structure is a minimum.")
        files["freq.txt"] = freq_text(vib)
        files["freq.log"] = gaussian.freq_log(atoms, E, best["hist"]["forces"][-1], vib, job)
    return files, log
