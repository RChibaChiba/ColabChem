"""遷移状態（TS）の探索。DMF（Direct MaxFlux）で TS の初期構造を作り、Sella で鞍点に最適化する。

流れ:
  1. 反応物と生成物（原子の順番が同じもの）から、DMF で反応経路を最適化し、経路上の最高点を TS の初期構造にする
  2. Sella（order=1）で1次の鞍点に最適化する。最初のヘシアンは UMA の数値微分で作る
  3. 振動計算で虚振動が1本だけか確かめる
  4. 虚振動の向きに少しずらして両側に最適化し、反応物と生成物につながっているか確かめる（簡易IRC）
"""
import os
import tempfile
import time

import numpy as np

from . import gaussian, opt
from .opt import _oom_error
from .structure import show_animation, show_mode, show_structure, xyz_block
from .ui import ColabUMAError, ProgressDisplay, bar_row, convergence_fraction, ok, warn

EV_PER_KCALMOL = 0.043364104241800934
DMF_TOL = {"loose": 0.2, "middle": 0.1, "tight": 0.04}  # pydmf の dual_inf_tol のプリセット
N_PATH_POINTS = 31  # 経路のエネルギーを描く点の数


# --- 構造の比較 ---
def kabsch_align(mobile, ref):
    """mobile を ref に重ねた座標を返す（並進と回転だけ。原子の順番は同じとする）。"""
    p = mobile.positions - mobile.positions.mean(axis=0)
    q = ref.positions - ref.positions.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return p @ rot + ref.positions.mean(axis=0)


def rmsd(a, b):
    return float(np.sqrt(((kabsch_align(a, b) - b.positions) ** 2).sum(axis=1).mean()))


def bonds(atoms, scale=1.25):
    from ase.data import covalent_radii

    r = covalent_radii[atoms.numbers]
    m = atoms.get_all_distances() < scale * (r[:, None] + r[None, :])
    np.fill_diagonal(m, False)
    return {(i, j) for i, j in zip(*np.nonzero(m)) if i < j}


def bond_changes(a, b):
    ba, bb = bonds(a), bonds(b)
    return sorted(bb - ba), sorted(ba - bb)  # (できる結合, 切れる結合)


def bond_text(pairs, atoms):
    s = atoms.get_chemical_symbols()
    return ", ".join(f"{s[i]}{i + 1}–{s[j]}{j + 1}" for i, j in pairs) or "none"


def check_endpoints(reactant, product):
    """反応物と生成物が同じ分子か確かめ、生成物を反応物に重ねたものを返す。"""
    if len(reactant) != len(product):
        raise ColabUMAError(f"The reactant ({len(reactant)} atoms) and product ({len(product)} atoms) have different numbers of atoms.",
                            "Give two structures of the same molecule with the atoms in the same order.",
                            ja=f"反応物（{len(reactant)} 原子）と生成物（{len(product)} 原子）の原子数が違います。",
                            ja_hint="同じ分子の、原子の順番がそろった2つの構造を用意してください。")
    if list(reactant.numbers) != list(product.numbers):
        k = next(i for i, (x, y) in enumerate(zip(reactant.numbers, product.numbers)) if x != y)
        raise ColabUMAError(f"The atom order differs between the reactant and product (atom {k + 1} is "
                            f"{reactant.get_chemical_symbols()[k]} and {product.get_chemical_symbols()[k]}).",
                            "DMF needs two structures with the same atom order. Renumber them in GaussView or similar.",
                            ja=f"反応物と生成物で原子の順番が違います（{k + 1} 番目が "
                               f"{reactant.get_chemical_symbols()[k]} と {product.get_chemical_symbols()[k]}）。",
                            ja_hint="DMF は原子の順番が同じ2つの構造を必要とします。GaussView などで番号をそろえてください。")
    product = product.copy()
    product.positions = kabsch_align(product, reactant)
    if rmsd(product, reactant) < 0.05:
        raise ColabUMAError("The reactant and product are nearly the same structure.", "Check the product file.",
                            ja="反応物と生成物がほぼ同じ構造です。", ja_hint="生成物のファイルを確認してください。")
    return product


# --- DMF ---
def run_dmf(reactant, product, calc, job, nmove=10, tol="middle", max_iter=300):
    """DMF で反応経路を最適化する。TS の初期構造（経路上の最高点）を含む dict を返す。"""
    from dmf import DirectMaxFlux, interpolate_fbenm

    workdir = tempfile.mkdtemp()
    ref = [reactant.copy(), product.copy()]
    for a in ref:
        a.calc = None
    print("Building the initial path with FB-ENM…")
    t0 = time.time()
    fb = interpolate_fbenm(ref, correlated=True, output_file=os.path.join(workdir, "fbenm_ipopt.out"),
                           ipopt_options={"print_level": 0, "sb": "yes"})
    coefs = fb.coefs.copy()
    init_frames = [im.positions.copy() for im in fb.images]

    progress = ProgressDisplay()
    target = DMF_TOL[tol]
    state = {"iter": 0, "du": [], "emax": []}

    class _DMF(DirectMaxFlux):
        def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, *args):
            ret = super().intermediate(alg_mod, iter_count, obj_value, inf_pr, inf_du, *args)
            du = inf_du / self.ipopt_options["obj_scaling_factor"]
            state["iter"] = iter_count
            state["du"].append(du)
            emax = float(np.max(self.energies)) / EV_PER_KCALMOL  # 反応物基準（e0 を引いた値）
            state["emax"].append(emax)
            progress.show([bar_row("Converge", convergence_fraction(state["du"][0], du, target),
                                   f"DMF  iteration {iter_count}/{max_iter}  dual infeasibility {du:.3f} → {target}  "
                                   f"path maximum {emax:.1f} kcal/mol")])
            return ret

    mx = _DMF(ref, coefs=coefs, nmove=nmove, update_teval=True)
    for im in mx.images:
        im.info.update(reactant.info)
        im.calc = calc
    mx.add_ipopt_options({"output_file": os.path.join(workdir, "dmf_ipopt.out"), "max_iter": int(max_iter),
                          "print_level": 0, "sb": "yes"})
    print(f"Optimizing the reaction path with DMF (nmove = {nmove}, convergence {tol}, at most {max_iter} iterations)")
    status = None
    try:
        _, info = mx.solve(tol=tol)
        status = info.get("status")
    except _oom_error():
        raise ColabUMAError("The GPU ran out of memory.", "Use a smaller nmove, or fewer atoms.",
                            ja="GPUのメモリが足りません。", ja_hint="nmove を小さくするか、原子数を減らしてください。") from None
    except Exception as e:  # IPOPT が途中で止まっても、そこまでの経路は使える
        warn(f"DMF stopped early ({type(e).__name__}: {e}). Continuing with the path so far.")
    converged = status in (0, 1)  # 0: 収束、1: 許容範囲で収束
    seconds = time.time() - t0
    if not mx.history.images_tmax:
        raise ColabUMAError("The DMF path was never updated.", "Check the reactant and product structures (atom order).",
                            ja="DMF の経路が1回も更新されませんでした。", ja_hint="反応物と生成物の構造（原子の順番）を確かめてください。")

    # --- 経路のエネルギー（等間隔の点で計算し直す） ---
    ts_guess = mx.history.images_tmax[-1].copy()
    ts_guess.info.update(reactant.info)
    t_max = float(mx.history.tmax[-1])
    t_grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, N_PATH_POINTS), [t_max]]))
    frames = mx.get_positions(t=t_grid)
    energies, forces = [], []
    a = reactant.copy()
    a.calc = calc
    for pos in frames:
        a.positions = pos
        energies.append(float(a.get_potential_energy()))
        forces.append(a.get_forces().copy())
    progress.show([bar_row("Converge", 1.0 if converged else convergence_fraction(state["du"][0], state["du"][-1], target),
                           f"DMF  iteration {state['iter']}/{max_iter}  {'converged' if converged else 'NOT converged'}")], force=True)
    e0 = energies[0]
    i_max = int(np.argmax(energies))
    print(f"  DMF: {state['iter']} iterations  {'converged' if converged else 'NOT converged'}  path maximum "
          f"{(energies[i_max] - e0) / EV_PER_KCALMOL:.2f} kcal/mol (vs reactant, {seconds:.1f} s)")
    return {"ts_guess": ts_guess, "t": t_grid, "t_max": t_max, "frames": frames, "energies": energies, "forces": forces,
            "init_frames": init_frames, "converged": converged, "status": status, "iterations": state["iter"],
            "seconds": seconds, "nmove": nmove, "tol": tol, "max_iter": max_iter}


# --- Sella ---
def run_sella(guess, calc, job, internal=True, exact_hessian=True, label="TS optimization"):
    """Sella で1次の鞍点に最適化する。"""
    from sella import Sella

    progress = ProgressDisplay()

    def hessian(atoms):
        return opt.numerical_hessian(atoms, calc, label="Hessian", progress=progress)

    def make(atoms):
        kw = {"order": 1, "logfile": None}
        if exact_hessian:
            kw["hessian_function"] = hessian
        if internal:
            try:
                return Sella(atoms, internal=True, allow_fragments=True, **kw)
            except Exception as e:
                warn(f"Sella could not start in internal coordinates, so it optimizes in Cartesian coordinates ({type(e).__name__}).")
        return Sella(atoms, internal=False, **kw)

    atoms = guess.copy()
    atoms.set_constraint()
    print(f"Optimizing the saddle point with Sella (fmax < {job.fmax} eV/Å, at most {job.max_steps} steps"
          f"{', numerical initial Hessian' if exact_hessian else ''})")
    return opt.run(atoms, calc, job, label, optimizer=make, progress=progress)


def check_ts(ts_res, calc):
    vib = opt.vibrations(ts_res["atoms"], calc, label="TS vibrational analysis")
    if vib is None:
        return None
    imag = vib["freqs"][vib["freqs"] < -opt.IMAG_THRESHOLD]
    print(f"Imaginary frequencies: {vib['n_imag']} (those below {opt.IMAG_THRESHOLD:.0f} cm⁻¹ are treated as numerical noise)"
          + (f"  imaginary: {', '.join(f'{v:.1f}' for v in imag)} cm⁻¹" if len(imag) else ""))
    if vib["n_imag"] == 1:
        ok(f"A first-order saddle point with exactly one imaginary frequency ({imag[0]:.1f} cm⁻¹).")
    elif vib["n_imag"] == 0:
        warn("There is no imaginary frequency. The structure fell to a minimum, not a TS.",
             "Increase DMF's nmove or set its convergence to tight, then run again.")
    else:
        warn(f"There are {vib['n_imag']} imaginary frequencies. This is not a first-order saddle point (TS).",
             "Displace the structure along the other imaginary modes, or revise the initial TS guess.")
    return vib


def connect(ts_atoms, vib, calc, job, reactant=None, product=None, shift=0.15):
    """虚振動の向きに ±shift Å（一番動く原子の変位）ずらして最適化し、どこにつながるか調べる（簡易IRC）。"""
    mode = vib["modes"][0]
    step = mode / np.abs(np.linalg.norm(mode, axis=1)).max() * shift
    out = {}
    for sign, name in ((-1, "minus"), (1, "plus")):
        a = ts_atoms.copy()
        a.set_constraint()
        a.positions = ts_atoms.positions + sign * step
        out[name] = opt.optimize(a, calc, job, f"{'−' if sign < 0 else '+'} side of the imaginary mode")
    if reactant is not None and product is not None:
        # 反応物・生成物との結合の違いと RMSD で、どちら側がどちらにつながったか決める
        def score(r, ref):
            return (len(bonds(r["atoms"]) ^ bonds(ref)), rmsd(r["atoms"], ref))

        straight = (score(out["minus"], reactant), score(out["plus"], product))
        swapped = (score(out["minus"], product), score(out["plus"], reactant))
        if sum(s[0] for s in swapped) < sum(s[0] for s in straight) or (
                sum(s[0] for s in swapped) == sum(s[0] for s in straight)
                and sum(s[1] for s in swapped) < sum(s[1] for s in straight)):
            # minus / plus はずらした向きのまま残す（irc_minus.xyz / irc_plus.xyz の名前と中身を合わせる）
            out["to_reactant"], out["to_product"] = out["plus"], out["minus"]
            straight = (swapped[1], swapped[0])  # (反応物側, 生成物側) の順にそろえる
        else:
            out["to_reactant"], out["to_product"] = out["minus"], out["plus"]
        out["match"] = (straight[0][0] == 0, straight[1][0] == 0)
        out["rmsd"] = (straight[0][1], straight[1][1])
        if all(out["match"]):
            ok("The two sides of the imaginary mode reached structures bonded like the reactant and the product. "
               "The TS connects the reactant and the product.")
        else:
            side = [n for n, m in zip(("reactant", "product"), out["match"]) if not m]
            warn(f"The structures optimized along the imaginary mode are not bonded like the {' and '.join(side)}. "
                 "This TS may belong to another reaction.",
                 "Check irc_minus.xyz / irc_plus.xyz in the ZIP. Increasing DMF's nmove can help.")
    else:
        out["to_reactant"], out["to_product"] = out["minus"], out["plus"]
        out["match"] = None
    return out


# --- 結果の表示 ---
def report(job, reactant=None, product=None, dmf_res=None, ts_res=None, vib=None, conn=None, endpoints=None):
    """表示と出力ファイル。(files, log, rows) を返す。rows は [(名前, UMAのエネルギー eV), ...]（DFTとの比較用）。

    endpoints は端点の結果 {"reactant": r, "product": r}（r は "atoms" と "energy" を持つ dict）。
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    eh = gaussian.EV_PER_HARTREE
    files, log = {}, []
    ref = reactant if reactant is not None else ts_res["atoms"]
    symbols = ref.get_chemical_symbols()
    numbers = ref.numbers

    # --- DMF の経路 ---
    if dmf_res is not None:
        e = np.array(dmf_res["energies"])
        de = (e - e[0]) / EV_PER_KCALMOL
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(dmf_res["t"], de, marker="o", ms=3, color="#1f77b4", label="DMF path")
        i_guess = int(np.argmin(np.abs(dmf_res["t"] - dmf_res["t_max"])))  # t_max は t のグリッドに入っている
        ax.plot(dmf_res["t"][i_guess], de[i_guess], marker="*", ms=18, color="#ff7f0e", markeredgecolor="black",
                linestyle="none", zorder=5, label=f"TS guess for Sella (t = {dmf_res['t_max']:.3f})")
        ax.annotate(f"{de[i_guess]:.1f} kcal/mol", (dmf_res["t"][i_guess], de[i_guess]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=0.15)  # 星の上の数値がはみ出さないように
        if ts_res is not None:
            ax.axhline((ts_res["energy"] - e[0]) / EV_PER_KCALMOL, ls=":", color="#d62728", label="Sella TS")
        ax.set_xlabel("Path parameter t (reactant 0 → product 1)")
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title("DMF reaction path")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.show()
        if not dmf_res["converged"]:
            warn(f"DMF did not converge within {dmf_res['max_iter']} iterations (IPOPT status = {dmf_res['status']}). "
                 "The path maximum is used as the initial TS guess as it is.",
                 "If Sella does not find the TS, increase DMF's maximum iterations and run again.")
        path_xyz = "".join(xyz_block(symbols, pos, f"t={t:.4f} energy_eV={en:.8f} dE_kcal_mol={d:.4f}")
                           for t, pos, en, d in zip(dmf_res["t"], dmf_res["frames"], e, de))
        print("DMF path:")
        show_animation(path_xyz, interval=120)
        files["dmf_path.xyz"] = path_xyz
        files["dmf_path.log"] = gaussian.path_log(dmf_res["frames"], e, dmf_res["forces"], job, numbers,
                                                  program="ColabUMA TS (DMF path)")
        files["dmf_init.xyz"] = "".join(xyz_block(symbols, pos, f"FB-ENM initial path image={i}")
                                        for i, pos in enumerate(dmf_res["init_frames"]))
        files["ts_guess.xyz"] = xyz_block(symbols, dmf_res["ts_guess"].positions, f"DMF TS guess t={dmf_res['t_max']:.4f}")
        files["dmf_path.csv"] = pd.DataFrame({"t": np.round(dmf_res["t"], 6), "energy_eV": np.round(e, 8),
                                              "energy_Ha": np.round(e / eh, 10), "dE_kcal_mol": np.round(de, 4)}
                                             ).to_csv(index=False)
        log += ["[DMF]", f"nmove: {dmf_res['nmove']}", f"Convergence: {dmf_res['tol']}", f"Max iterations: {dmf_res['max_iter']}",
                f"Iterations: {dmf_res['iterations']}  converged: {dmf_res['converged']} (IPOPT status = {dmf_res['status']})",
                f"Path maximum: t = {dmf_res['t_max']:.4f}  ΔE = {de.max():.4f} kcal/mol (vs reactant)",
                f"Time: {dmf_res['seconds']:.1f} s", ""]

    # --- TS ---
    if ts_res is not None:
        ts = ts_res["atoms"]
        cmt = (f"TS energy_eV={ts_res['energy']:.8f} energy_Ha={ts_res['energy'] / eh:.10f} fmax={ts_res['fmax']:.5f} "
               f"converged={ts_res['converged']} model={job.model_name} charge={job.charge} multiplicity={job.multiplicity}")
        files["ts.xyz"] = xyz_block(symbols, ts.positions, cmt)
        files["ts_opt.log"] = gaussian.opt_log(ts_res, job, numbers, saddle=True, program="ColabUMA TS",
                                               method=f"{job.model_name}/{job.task_name} Sella")
        print("TS structure:")
        show_structure(files["ts.xyz"])
        opt.warn_unconverged(ts_res)
        log += ["[TS optimization (Sella)]",
                f"E = {ts_res['energy']:.8f} eV ({ts_res['energy'] / eh:.10f} Ha)  fmax = {ts_res['fmax']:.5f}  "
                f"steps = {ts_res['steps']}  converged = {ts_res['converged']}",
                f"{'step':>5s} {'energy_eV':>18s} {'fmax_eV/A':>12s}"]
        log += [f"{i:5d} {en:18.8f} {fm:12.5f}" for i, (en, fm) in
                enumerate(zip(ts_res["hist"]["energy"], ts_res["hist"]["fmax"]))]
        log.append("")
    if vib is not None:
        if vib["n_imag"] >= 1:
            print(f"Imaginary mode ({vib['freqs'][0]:.1f} cm⁻¹):")
            show_mode(ts_res["atoms"], vib["modes"][0])
        files["ts_freq.txt"] = opt.freq_text(vib)
        files["ts_freq.log"] = gaussian.freq_log(ts_res["atoms"], ts_res["energy"], ts_res["hist"]["forces"][-1], vib, job,
                                                 program="ColabUMA TS")
        log += ["[TS vibrational analysis]", f"Imaginary frequencies: {vib['n_imag']}",
                f"Lowest frequencies: {', '.join(f'{v:.1f}' for v in vib['freqs'][:3])} cm^-1", ""]
    if conn is not None:
        for name in ("minus", "plus"):
            r = conn[name]
            files[f"irc_{name}.xyz"] = xyz_block(symbols, r["atoms"].positions,
                                                 f"from TS along {'-' if name == 'minus' else '+'} imaginary mode "
                                                 f"energy_eV={r['energy']:.8f} converged={r['converged']}")
        log += ["[Simple IRC check (optimized after a displacement along the imaginary mode)]"]
        for name in ("minus", "plus"):
            r = conn[name]
            log.append(f"{name}: E = {r['energy']:.8f} eV  fmax = {r['fmax']:.5f}  steps = {r['steps']}  converged = {r['converged']}")
        if conn.get("match") is not None:
            log.append(f"Matches reactant: {conn['match'][0]} (RMSD {conn['rmsd'][0]:.3f} Å)  "
                       f"Matches product: {conn['match'][1]} (RMSD {conn['rmsd'][1]:.3f} Å)")
        log.append("")

    # --- エネルギーのまとめ ---
    rows = []
    sides = side_structures(endpoints, conn)
    if ts_res is not None and sides is None:
        rows = [("TS", ts_res["energy"])]
    if ts_res is not None and sides is not None:
        (r_name, r_atoms, r_e), (p_name, p_atoms, p_e) = sides
        e_ts = ts_res["energy"]
        rows = [(r_name, r_e), ("TS", e_ts), (p_name, p_e)]
        table = pd.DataFrame([{"": n, "Energy (eV)": round(v, 6), "Energy (Hartree)": round(v / eh, 8),
                               f"ΔE (vs {r_name}, kcal/mol)": round((v - r_e) / EV_PER_KCALMOL, 2)} for n, v in rows])
        print("UMA energies:")
        display(table.style.hide(axis="index"))
        print(f"Activation energy (UMA): forward {(e_ts - r_e) / EV_PER_KCALMOL:.2f} kcal/mol, "
              f"reverse {(e_ts - p_e) / EV_PER_KCALMOL:.2f} kcal/mol, reaction energy {(p_e - r_e) / EV_PER_KCALMOL:.2f} kcal/mol")
        log += ["[Energies (UMA)]"] + [f"{n}: {v:.8f} eV ({v / eh:.10f} Ha)" for n, v in rows]
        log += [f"Activation energy, forward: {(e_ts - r_e) / EV_PER_KCALMOL:.4f} kcal/mol",
                f"Activation energy, reverse: {(e_ts - p_e) / EV_PER_KCALMOL:.4f} kcal/mol",
                f"Reaction energy: {(p_e - r_e) / EV_PER_KCALMOL:.4f} kcal/mol", ""]
        if e_ts < max(r_e, p_e):
            warn(f"The TS energy is lower than that of the {(r_name if e_ts < r_e else p_name).lower()}.",
                 "The reactant and product may not be optimized, or the TS may belong to another reaction. "
                 "Turn on the endpoint optimization and run again.")
        formed, broken = bond_changes(r_atoms, p_atoms)
        print(f"{r_name} → {p_name}: bonds formed: {bond_text(formed, r_atoms)}  bonds broken: {bond_text(broken, r_atoms)}")
        log += [f"Bonds formed: {bond_text(formed, r_atoms)}", f"Bonds broken: {bond_text(broken, r_atoms)}", ""]
    if endpoints:
        for tag, key in (("reactant", "reactant"), ("product", "product")):
            r = endpoints[key]
            files[f"{tag}.xyz"] = xyz_block(symbols, r["atoms"].positions,
                                            f"{tag} energy_eV={r['energy']:.8f} energy_Ha={r['energy'] / eh:.10f} "
                                            f"converged={r.get('converged')}")
            if r.get("hist"):
                files[f"{tag}_opt.log"] = gaussian.opt_log(r, job, numbers, program="ColabUMA TS")
    return files, log, rows


def side_structures(endpoints=None, conn=None):
    """TS の両側の構造 [(名前, atoms, UMAのエネルギー), (名前, atoms, エネルギー)]。なければ None。"""
    if endpoints:
        return [("Reactant", endpoints["reactant"]["atoms"], endpoints["reactant"]["energy"]),
                ("Product", endpoints["product"]["atoms"], endpoints["product"]["energy"])]
    if conn is not None:
        if conn.get("match") is not None:
            names = ("Reactant", "Product")
        else:
            names = ("− side", "+ side")
        return [(names[0], conn["to_reactant"]["atoms"], conn["to_reactant"]["energy"]),
                (names[1], conn["to_product"]["atoms"], conn["to_product"]["energy"])]
    return None


def dft_points(ts_res, endpoints=None, conn=None):
    """DFT の一点計算をする構造 [(名前, atoms), ...]。"""
    sides = side_structures(endpoints, conn)
    if sides is None:
        return [("TS", ts_res["atoms"])]
    return [(sides[0][0], sides[0][1]), ("TS", ts_res["atoms"]), (sides[1][0], sides[1][1])]
