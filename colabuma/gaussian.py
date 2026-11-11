"""Gaussian の出力（log）と入力（gjf）の書式で書き出す。GaussView で経路・スキャン・振動を見るため。

ここで書く log は表示用に Gaussian の書式をまねたもので、Gaussian で計算した結果ではない。
"""
import datetime

import numpy as np

EV_PER_HARTREE = 27.211386245988  # ase.units.Hartree と同じ値
BOHR = 0.5291772105638411  # Å（ase.units.Bohr と同じ値）

GRAD = " GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad"
DASH69 = " " + "-" * 69
DASH67 = " " + "-" * 67


def _symbol(z):
    from ase.data import chemical_symbols

    return chemical_symbols[z]


def header(job, numbers, route, first, program="ColabUMA", method=None):
    method = method or f"{job.model_name}/{job.task_name} {job.optimizer_name}"
    return [
        " Entering Gaussian System, Link 0=g16",
        " ******************************************",
        " Output format compatible with Gaussian 16 (Gaussian, Inc.) so that GaussView and other readers can open it.",
        f" This file was written by {program}. It is NOT the output of a Gaussian calculation:",
        f" energies and forces come from the UMA machine-learning potential ({job.model_name}, task={job.task_name}).",
        " ******************************************",
        " ---------------------------------------------",
        f" #p {route} external='ColabUMA {method}'",
        " ---------------------------------------------",
        " ---------------",
        f" {job.out_prefix}",
        " ---------------",
        " Symbolic Z-matrix:",
        f" Charge = {int(job.charge):2d} Multiplicity = {int(job.multiplicity)}",
    ] + [f" {_symbol(z):<2s} {x:20.8f}{y:14.8f}{zz:14.8f}" for z, (x, y, zz) in zip(numbers, first)] + [" "]


def orientation(numbers, pos, title="Input orientation:"):
    out = [" " * ((71 - len(title)) // 2) + title, DASH69,
           " Center     Atomic      Atomic             Coordinates (Angstroms)",
           " Number     Number       Type             X           Y           Z", DASH69]
    out += [f"{k + 1:7d}{z:11d}{0:12d}    {x:12.6f}{y:12.6f}{zz:12.6f}" for k, (z, (x, y, zz)) in enumerate(zip(numbers, pos))]
    return out + [DASH69]


def scf_done(e_ev, label="UMA"):
    return [f" SCF Done:  E({label}) = {e_ev / EV_PER_HARTREE:17.9f}     A.U. after    1 cycles"]


def forces_block(numbers, f_ev):
    fa = f_ev * BOHR / EV_PER_HARTREE  # eV/Å → Hartree/Bohr
    out = [DASH67, " Center     Atomic                   Forces (Hartrees/Bohr)",
           " Number     Number              X              Y              Z", DASH67]
    out += [f"{k + 1:7d}{z:9d}{fx:19.9f}{fy:15.9f}{fz:15.9f}" for k, (z, (fx, fy, fz)) in enumerate(zip(numbers, fa))]
    out += [DASH67, f" Cartesian Forces:  Max {np.abs(fa).max():15.9f} RMS {np.sqrt((fa ** 2).mean()):15.9f}"]
    return out, fa


def steps(r, numbers, step_suffix="", last_only=False, saddle=False):
    """1回の最適化のステップを Gaussian の opt 出力と同じ順序で書く。last_only なら最終ステップだけ。"""
    t = r["fmax_target"] * BOHR / EV_PER_HARTREE
    thresholds = (t, t * 2 / 3, t * 4, t * 8 / 3)  # Gaussian の既定値と同じ比率
    max_steps = r["max_steps"]
    h = r["hist"]
    n_frames = len(h["frames"])
    out = []
    for i, (pos, e, f) in enumerate(zip(h["frames"], h["energy"], h["forces"])):
        if last_only and i < n_frames - 1:
            continue
        out += orientation(numbers, pos) + scf_done(e)
        fb, fa = forces_block(numbers, f)
        out += fb
        out += [GRAD, " Berny optimization.",
                " Search for a saddle point of order  1." if saddle else " Search for a local minimum.",
                f" Step number{i + 1:4d} out of a maximum of{max_steps + 1:4d}{step_suffix}"]
        disp = (h["frames"][i + 1] - pos) / BOHR if i + 1 < n_frames else np.zeros_like(pos)
        values = (np.abs(fa).max(), np.sqrt((fa ** 2).mean()), np.abs(disp).max(), np.sqrt((disp ** 2).mean()))
        out.append("         Item               Value     Threshold  Converged?")
        for label, v, th in zip(("Maximum Force", "RMS     Force", "Maximum Displacement", "RMS     Displacement"),
                                values, thresholds):
            out.append(f" {label:<21s}{v:12.6f}{th:13.6f}     {'YES' if v < th else 'NO'}")
        de = (h["energy"][i + 1] - e) / EV_PER_HARTREE if i + 1 < n_frames else 0.0
        out.append(" Predicted change in Energy=" + f"{de:.6E}".replace("E", "D"))
        if i == n_frames - 1:
            if r["converged"]:
                out += [" Optimization completed.", "    -- Stationary point found."]
            else:
                out += [" Optimization stopped.", f"    -- Number of steps exceeded,  NStep={max_steps}"]
        out.append(GRAD)
    return out


def end(normal=True):
    if normal:
        return [f" Normal termination of Gaussian 16 at {datetime.datetime.now():%a %b %d %H:%M:%S %Y}."]
    return [" Error termination request processed by link 9999."]


def opt_log(r, job, numbers, saddle=False, program="ColabUMA", method=None):
    route = "opt=(ts,calcfc,noeigentest)" if saddle else "opt"
    out = header(job, numbers, route, r["hist"]["frames"][0], program, method)
    out += steps(r, numbers, saddle=saddle) + end(r["converged"])
    return "\n".join(out) + "\n"


def _fixed(v, width, digits):
    # -0.00 と書かないようにする（Gaussian の出力には出てこない）
    return f"{round(float(v), digits) + 0.0:{width}.{digits}f}"


def freq_section(vib, numbers):
    # Gaussian の freq 出力と同じ書式の振動数と基準振動（3モードずつ）
    out = [" Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering",
           " activities (A**4/AMU), depolarization ratios for plane and unpolarized",
           " incident light, reduced masses (AMU), force constants (mDyne/A),",
           " and normal coordinates:"]
    freqs = vib["freqs"]
    row = lambda values: "".join(_fixed(v, 12 if j == 0 else 23, 4) for j, v in enumerate(values))
    for s in range(0, len(freqs), 3):
        idx = list(range(s, min(s + 3, len(freqs))))
        out.append("".join(f"{k + 1:23d}" for k in idx))
        out.append("".join(f"{'A':>23s}" for _ in idx))
        out.append(" Frequencies --" + row(freqs[k] for k in idx))
        out.append(" Red. masses --" + row(vib["red_mass"][k] for k in idx))
        out.append(" Frc consts  --" + row(abs(vib["force_const"][k]) for k in idx))  # Gaussian は虚振動でも正の値
        out.append(" IR Inten    --" + row(0.0 for _ in idx))
        out.append("  Atom  AN" + "".join(("" if j == 0 else "  ") + "      X      Y      Z" for j in range(len(idx))))
        for a, z in enumerate(numbers):
            out.append(f"{a + 1:6d}{z:4d}" + "".join(
                "  " + "".join(_fixed(c, 7, 2) for c in vib["modes"][k][a]) for k in idx))
    return out


def freq_log(atoms, energy, forces, vib, job, program="ColabUMA"):
    """振動数と基準振動を、Gaussian の freq ジョブの出力と同じ並びで書く（GaussView の Results → Vibrations 用）。

    構造 → SCF Done → Low frequencies → 基準振動 → Thermochemistry の見出し → 力、の順。
    座標は Input orientation と Standard orientation の両方に同じものを書く（基準振動もこの座標系のもの）。
    """
    numbers = atoms.numbers
    n = len(numbers)
    out = header(job, numbers, "freq", atoms.positions, program, method=f"{job.model_name}/{job.task_name}")
    out += [f" NAtoms= {n:6d} NQM=   {n:6d} NQMF=       0 NMMI=      0 NMMIF=      0"]
    out += orientation(numbers, atoms.positions)
    out += orientation(numbers, atoms.positions, "Standard orientation:")
    out += scf_done(energy)
    low = list(vib["all"])  # 並進・回転を射影する前の全モード（小さい順）
    out += [" Full mass-weighted force constant matrix:",
            " Low frequencies ---" + "".join(_fixed(v, 10, 4) for v in low[:6])]
    if len(low) > 6:
        out += [" Low frequencies ---" + "".join(_fixed(v, 10, 4) for v in low[6:9])]
    n_neg = int((np.asarray(vib["freqs"]) < 0).sum())
    if n_neg:
        out += [f" ******{n_neg:5d} imaginary frequencies (negative Signs) ****** "]
    out += [" Diagonal vibrational polarizability:", "".join(f"{0.0:16.7f}" for _ in range(3))]
    out += freq_section(vib, numbers)
    out += ["", " -------------------", " - Thermochemistry -", " -------------------",
            " Temperature   298.150 Kelvin.  Pressure   1.00000 Atm."]
    out += [f" Atom {k + 1:5d} has atomic number {z:2d} and mass {m:9.5f}" for k, (z, m) in enumerate(zip(numbers, atoms.get_masses()))]
    out += [" ***** Axes restored to original set *****"] + forces_block(numbers, forces)[0]
    out += [" "] + end(True)
    return "\n".join(out) + "\n"


def path_log(frames, energies, forces, job, numbers, title_route="opt", program="ColabUMA"):
    """経路（DMFの反応経路など）を、各点を1ステップとする opt 出力の書式で書く。

    GaussView で開くと、経路のアニメーションとエネルギーのグラフ（横軸は点の番号）が見られる。
    """
    r = {"hist": {"frames": list(frames), "energy": list(energies), "forces": list(forces)},
         "converged": True, "fmax_target": 0.0, "max_steps": len(frames)}
    out = header(job, numbers, title_route, frames[0], program)
    out += steps(r, numbers) + end(True)
    return "\n".join(out) + "\n"


def gjf(atoms, job, route, title, chk=None, mem="8GB", nproc=8):
    lines = []
    if chk:
        lines.append(f"%chk={chk}")
    lines += [f"%mem={mem}", f"%nprocshared={nproc}", route, "", title, "", f"{int(job.charge)} {int(job.multiplicity)}"]
    lines += [f" {s:<2s} {x:16.8f}{y:16.8f}{z:16.8f}" for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions)]
    return "\n".join(lines) + "\n\n"
