"""緩和スキャン（結合長・角度・二面角）。Gaussian の opt=modredundant の S 指定に相当する。"""
from dataclasses import dataclass, field

import numpy as np

from . import gaussian, opt
from .structure import show_animation, show_numbered, xyz_block
from .ui import ColabUMAError, ProgressDisplay, bar_row, ok, warn

SCAN_KINDS = {2: ("R", "bond length", "Å", "Distance (Å)"), 3: ("A", "angle", "deg", "Angle (deg)"),
              4: ("D", "dihedral", "deg", "Dihedral (deg)")}
SCAN_TOLERANCE = 0.01  # Å または 度。目標値と実測値の差がこれを超えたら注意を出す
MIN_PROMINENCE = 0.1  # kcal/mol。極小を探すとき、これより小さい凹凸は無視する
EV_PER_KCALMOL = 0.043364104241800934  # ase.units.kcal / ase.units.mol

_ATOMS_HINT = "2 atoms give a bond length, 3 an angle, 4 a dihedral (for example 1 2 3 4)."
_ATOMS_HINT_JA = "2個なら結合長、3個なら角度、4個なら二面角です（例：1 2 3 4）。"
_COLLINEAR_HINT = "Choose other atoms. An angle or dihedral is not defined for nearly collinear atoms."
_COLLINEAR_HINT_JA = "原子の指定を変えてください。直線に近い原子の組では、角度や二面角を決められません。"


def wrap180(v):
    # 二面角を (-180, 180] に収める
    return 180.0 - (180.0 - v) % 360.0


@dataclass
class ScanSpec:
    idx: list  # 0始まりの原子番号
    start: float = None
    n_steps: int = 36
    step_size: float = 10.0
    opt_before: bool = True
    opt_after: bool = True
    text: str = ""  # フォームに書かれた原子番号（メッセージ用）
    code: str = field(init=False)
    name: str = field(init=False)
    unit: str = field(init=False)
    xlabel: str = field(init=False)

    def __post_init__(self):
        self.code, self.name, self.unit, self.xlabel = SCAN_KINDS[len(self.idx)]

    @property
    def atoms_label(self):
        return " ".join(str(i + 1) for i in self.idx)

    @property
    def definition(self):
        return f"{self.code}({','.join(str(i + 1) for i in self.idx)})"


def parse_spec(scan_atoms, scan_start, n_steps, step_size, opt_before=True, opt_after=True):
    """フォームの値を確かめて ScanSpec にする（原子数が要らない確認だけ）。"""
    try:
        idx = [int(t) - 1 for t in scan_atoms.replace(",", " ").split()]
    except ValueError:
        raise ColabUMAError(f"Could not read scan_atoms \"{scan_atoms}\" as atom numbers.", _ATOMS_HINT,
                            ja=f"scan_atoms「{scan_atoms}」を原子番号として読み取れませんでした。", ja_hint=_ATOMS_HINT_JA) from None
    if len(idx) not in (2, 3, 4):
        raise ColabUMAError(f"Give 2–4 atom numbers in scan_atoms (now {len(idx)}).", _ATOMS_HINT,
                            ja=f"scan_atoms には原子番号を2〜4個指定してください（今は {len(idx)} 個です）。", ja_hint=_ATOMS_HINT_JA)
    if min(idx) < 0:
        raise ColabUMAError(f"Atom numbers in scan_atoms start at 1 (now {scan_atoms}).", _ATOMS_HINT,
                            ja=f"scan_atoms の原子番号は1以上で指定してください（今は {scan_atoms} です）。", ja_hint=_ATOMS_HINT_JA)
    if len(set(idx)) != len(idx):
        raise ColabUMAError(f"scan_atoms has the same atom number twice ({scan_atoms}).", _ATOMS_HINT,
                            ja=f"scan_atoms に同じ原子番号が入っています（{scan_atoms}）。", ja_hint=_ATOMS_HINT_JA)
    if n_steps < 1:
        raise ColabUMAError("The number of steps must be 1 or more.", "The number of points is the number of steps plus 1.",
                            ja="刻みの回数は1以上にしてください。", ja_hint="計算する点の数は、刻みの回数に1を足した数です。")
    if step_size == 0:
        raise ColabUMAError("The step size cannot be 0.", "About 0.1 Å for a bond length, or 10 deg for an angle or dihedral.",
                            ja="刻み幅に0は指定できません。", ja_hint="結合長なら 0.1 Å、角度や二面角なら 10 度 程度が目安です。")
    start = None
    if str(scan_start).strip():
        try:
            start = float(scan_start)
        except ValueError:
            raise ColabUMAError(f"Could not read scan_start \"{scan_start}\" as a number.",
                                "Leave it empty to start from the value in the input structure.",
                                ja=f"scan_start「{scan_start}」を数値として読み取れませんでした。",
                                ja_hint="空欄にすると入力構造の値から始めます。") from None
    if len(idx) == 4 and abs(n_steps * step_size) > 360.0 + 1e-6:
        warn("The dihedral moves more than one full turn. Some angles are computed twice.")
    return ScanSpec(idx, start, n_steps, step_size, opt_before, opt_after, scan_atoms)


def check_atom_range(spec, n_atoms, hint):
    if n_atoms and max(spec.idx) >= n_atoms:
        raise ColabUMAError(f"Atom numbers in scan_atoms must be 1–{n_atoms} (now {spec.text}).", hint,
                            ja=f"scan_atoms の原子番号は1〜{n_atoms}の範囲で指定してください（今は {spec.text} です）。",
                            ja_hint="3D表示で番号を確認してください。")


class Scan:
    """スキャン座標の測り方と動かし方。入力構造の結合から、一緒に動かす原子を決める。"""

    def __init__(self, spec, atoms0):
        from ase.data import covalent_radii

        self.spec = spec
        radii = covalent_radii[atoms0.numbers]
        self._bonded = atoms0.get_all_distances() < 1.25 * (radii[:, None] + radii[None, :])
        np.fill_diagonal(self._bonded, False)
        self.moving, self.in_ring = self._moving_atoms(spec.idx)

    def _reach(self, start, blocked):
        seen, stack = {start}, [start]
        while stack:
            for j in np.flatnonzero(self._bonded[stack.pop()]):
                if int(j) not in seen and int(j) not in blocked:
                    seen.add(int(j))
                    stack.append(int(j))
        return seen

    def _moving_atoms(self, idx):
        # 動かす側の断片を返す。環の中で切り離せないときは末端の原子だけを動かす
        pivot, start, moved = (idx[0], idx[1], idx[1]) if len(idx) == 2 else (idx[1], idx[2], idx[-1])
        side = self._reach(start, {pivot})
        if moved not in side:
            side = self._reach(moved, {pivot, start})
        fixed_side = (set(int(j) for j in np.flatnonzero(self._bonded[pivot])) | {idx[0]}) - {start, moved}
        if side & fixed_side:
            return [moved], True
        return sorted(side), False

    def check_geometry(self, atoms):
        s = self.spec
        if s.code == "A" and not 1.0 < atoms.get_angle(*s.idx) < 179.0:
            raise ColabUMAError(f"The three atoms are nearly collinear ({atoms.get_angle(*s.idx):.1f} deg).", _COLLINEAR_HINT,
                                ja=f"指定した3原子がほぼ直線上に並んでいます（{atoms.get_angle(*s.idx):.1f} 度）。",
                                ja_hint=_COLLINEAR_HINT_JA)
        if s.code == "D":
            for tri in (s.idx[:3], s.idx[1:]):
                if not 5.0 < atoms.get_angle(*tri) < 175.0:
                    raise ColabUMAError(f"Three atoms of the dihedral ({' '.join(str(i + 1) for i in tri)}) are nearly collinear.",
                                        _COLLINEAR_HINT,
                                        ja=f"二面角を作る3原子（{' '.join(str(i + 1) for i in tri)}）がほぼ直線上に並んでいます。",
                                        ja_hint=_COLLINEAR_HINT_JA)

    def value(self, atoms):
        s = self.spec
        if s.code == "R":
            return float(atoms.get_distance(*s.idx))
        if s.code == "A":
            return float(atoms.get_angle(*s.idx))
        return float(wrap180(atoms.get_dihedral(*s.idx)))

    def targets(self, atoms):
        s = self.spec
        start = self.value(atoms) if s.start is None else s.start
        targets = [start + k * s.step_size for k in range(s.n_steps + 1)]
        if s.code == "R" and min(targets) < 0.3:
            raise ColabUMAError(f"The bond length would shrink to {min(targets):.3f} Å.",
                                "Check the start value, the number of steps and the step size.",
                                ja=f"結合長が {min(targets):.3f} Å まで短くなる設定です。", ja_hint="開始値、刻みの回数、刻み幅を見直してください。")
        if s.code == "A" and not (0.0 < min(targets) and max(targets) < 180.0):
            raise ColabUMAError(f"The angle would move over {min(targets):.1f}–{max(targets):.1f} deg.",
                                "Keep the angle strictly between 0 and 180 deg. The constraint is not defined at exactly 180 deg.",
                                ja=f"角度が {min(targets):.1f}〜{max(targets):.1f} 度の範囲を動く設定です。",
                                ja_hint="角度は0度と180度を含まない範囲で指定してください。180度ちょうどでは拘束を定義できません。")
        return targets

    def drive(self, atoms, target):
        # 動かす側の断片ごと、スキャン座標を目標値まで動かす
        s = self.spec
        if s.code == "R":
            a, b = s.idx
            v = atoms.positions[b] - atoms.positions[a]
            d = np.linalg.norm(v)
            atoms.positions[self.moving] += v / d * (target - d)
        elif s.code == "A":
            atoms.set_angle(*s.idx, target, indices=self.moving)
        else:
            atoms.set_dihedral(*s.idx, target, indices=self.moving)

    def constraint(self, target):
        from ase.constraints import FixInternals

        s = self.spec
        if s.code == "R":
            return FixInternals(bonds=[[target, list(s.idx)]])
        if s.code == "A":
            return FixInternals(angles_deg=[[target, list(s.idx)]])
        return FixInternals(dihedrals_deg=[[wrap180(target), list(s.idx)]])


def prepare(spec, atoms0, show=True):
    """番号付きの3D表示を出し、スキャン座標を確かめて Scan を返す。"""
    if show:
        print("Atom numbers (red: the scanned atoms):")
        show_numbered(atoms0, spec.idx)
    check_atom_range(spec, len(atoms0), "Check the numbers in the 3D view above.")
    sc = Scan(spec, atoms0)
    sc.check_geometry(atoms0)
    targets = sc.targets(atoms0)
    if sc.in_ring:
        warn(f"The {spec.name} is in a ring, so only atom {spec.idx[-1] + 1} is moved to start each point. "
             "The starting structures can be strongly distorted.",
             "If the energy jumps, use a smaller step size.")
    ok(f"Scanning the {spec.name} {spec.atoms_label} over {len(targets)} points "
       f"(value in the input structure: {sc.value(atoms0):.3f} {spec.unit}).")
    return sc


def last_minimum_index(energies, converged):
    # 最後の点からさかのぼって最初の極小の番号を返す。未収束の点と小さな凹凸は無視する
    tol = MIN_PROMINENCE * EV_PER_KCALMOL
    usable = [i for i, c in enumerate(converged) if c] or list(range(len(energies)))
    best = usable[-1]
    for i in reversed(usable[:-1]):
        if energies[i] < energies[best]:
            best = i
        elif energies[i] - energies[best] > tol:
            break
    return best


def run(sc, atoms0, calc, job, skip_preopt_result=None):
    """スキャン本体。前後の最適化も含む。skip_preopt_result を渡すと、前の最適化の代わりにその結果を使う。"""
    spec = sc.spec
    progress = ProgressDisplay()
    state = {"done": 0, "n": len(sc.targets(atoms0))}

    def extra_rows():
        return [bar_row("Scan", state["done"] / state["n"], f"{state['done']}/{state['n']} points done")]

    def free_opt(start_atoms, label):
        r = opt.optimize(start_atoms, calc, job, label, progress=progress, extra_rows=extra_rows, quiet=True)
        r["value"] = sc.value(r["atoms"])
        print(f"  {label}: {spec.name} = {r['value']:.3f} {spec.unit}  E = {r['energy']:.6f} eV  fmax = {r['fmax']:.4f} eV/Å  "
              f"{r['steps']:4d} steps  {'converged' if r['converged'] else 'NOT converged'} ({r['seconds']:.1f} s)")
        return r

    pre_result = None
    current = atoms0
    if skip_preopt_result is not None:
        pre_result = dict(skip_preopt_result, value=sc.value(skip_preopt_result["atoms"]))
        current = pre_result["atoms"]
    elif spec.opt_before:
        print(f"Optimizing with {job.optimizer_name} before the scan (fmax < {job.fmax} eV/Å, at most {job.max_steps} steps)")
        pre_result = free_opt(atoms0, "Pre-scan optimization")
        current = pre_result["atoms"]
    sc.check_geometry(current)
    targets = sc.targets(current)
    state["n"] = len(targets)

    print(f"Optimizing each point with {job.optimizer_name} (fmax < {job.fmax} eV/Å, at most {job.max_steps} steps per point)")
    results = []
    for k, target in enumerate(targets):
        atoms = current.copy()
        atoms.set_constraint()
        sc.drive(atoms, target)
        atoms.set_constraint(sc.constraint(target))
        r = opt.run(atoms, calc, job, f"Point {k + 1}", progress=progress, extra_rows=extra_rows, quiet=True)
        value = sc.value(atoms)
        deviation = wrap180(value - target) if spec.code == "D" else value - target
        print(f"  Point {k + 1:3d}/{len(targets)}: {spec.name} = {target:9.3f} {spec.unit}  E = {r['energy']:.6f} eV  "
              f"fmax = {r['fmax']:.4f} eV/Å  {r['steps']:4d} steps  {'converged' if r['converged'] else 'NOT converged'}"
              f" ({r['seconds']:.1f} s)")
        r.update({"point": k + 1, "target": target, "value": target + deviation, "deviation": deviation})
        results.append(r)
        current = r["atoms"]
        state["done"] = k + 1
        progress.show(extra_rows(), force=True)

    post_result = post_index = None
    if spec.opt_after:
        post_index = last_minimum_index([r["energy"] for r in results], [r["converged"] for r in results])
        print(f"Going back from the last point, the first minimum is point {post_index + 1}. "
              "It is optimized from there without constraints.")
        post_result = free_opt(results[post_index]["atoms"], "Post-scan optimization")

    usable = [r for r in results if r["converged"]] or results  # 未収束の点のエネルギーは基準にしない
    lowest = min(usable, key=lambda r: r["energy"])
    for r in results:
        r["dE"] = (r["energy"] - lowest["energy"]) / EV_PER_KCALMOL
    return {"scan": sc, "targets": targets, "results": results, "pre": pre_result, "post": post_result,
            "post_index": post_index, "lowest": lowest, "highest": max(usable, key=lambda r: r["energy"])}


def gaussian_scan_log(res, job, numbers, converged_results):
    # 渡された各点の最終構造を Gaussian の opt=modredundant 出力と同じ書式で書く。
    # 点の番号はこのファイルの中で1から振り直す（未収束の点を除くと飛び番になるため）
    spec = res["scan"].spec
    atom_ids = [i + 1 for i in spec.idx]
    name = f"{spec.code}1"
    out = gaussian.header(job, numbers, "opt=modredundant", converged_results[0]["hist"]["frames"][0], "ColabUMA Scan")
    out += [" The following ModRedundant input section has been read:",
            f" {spec.code.replace('R', 'B')}" + "".join(f"{i:8d}" for i in atom_ids) + f" S{spec.n_steps:4d}{spec.step_size:7.3f}",
            " ",
            "                           ----------------------------",
            "                           !    Initial Parameters    !",
            "                           ! (Angstroms and Degrees)  !",
            " --------------------------                            --------------------------",
            " ! Name  Definition              Value          Derivative Info.                !",
            " " + "-" * 80,
            f" ! {name:<5s} {spec.definition:<21s} {converged_results[0]['target']:<16.4f} Scan" + " " * 28 + "!",
            " " + "-" * 80,
            f" Number of optimizations in scan={len(converged_results):4d}",
            " "]
    for n, r in enumerate(converged_results, 1):
        out += gaussian.steps(r, numbers, f" on scan point{n:6d} out of{len(converged_results):6d}", last_only=True)
    # 各点のまとめ（5列ずつ）。エネルギーは整数部を見出しに出して差分で書く
    eh = gaussian.EV_PER_HARTREE
    base = float(int(min(r["energy"] for r in converged_results) / eh))
    out.append(f" Summary of Optimized Potential Surface Scan (add {base:.1f} to energies):")
    for s in range(0, len(converged_results), 5):
        part = converged_results[s:s + 5]
        out.append(" " * 21 + "".join(f"{n:10d}" for n in range(s + 1, s + len(part) + 1)))
        out.append("     Eigenvalues --  " + "".join(f"{r['energy'] / eh - base:10.5f}" for r in part))
        out.append(f"{name:>13s}        " + "".join(f"{r['value']:10.5f}" for r in part))
    out.append(" ")
    out += gaussian.end(len(converged_results) == len(res["results"]))
    return "\n".join(out) + "\n"


def overlay_plot(res, dmf_res=None, points=()):
    """スキャンの図に、DMF の経路と構造（TS など）を重ねて描く。Workflow 用。

    横軸はどの構造でもスキャンした座標を測った値、縦軸はスキャンの最安定点を基準にした UMA のエネルギー。
    points は [(凡例の名前, atoms, エネルギー eV, matplotlib の書式 dict), ...]。
    """
    import matplotlib.pyplot as plt

    sc, spec = res["scan"], res["scan"].spec
    e_ref = res["lowest"]["energy"]
    xs = [r["target"] for r in res["results"]]
    mid = (min(xs) + max(xs)) / 2

    def x_of(atoms):
        v = sc.value(atoms)
        if spec.code == "D":  # 二面角は ±180° に折り返されるので、スキャンの範囲に近い側に寄せる
            v += 360.0 * round((mid - v) / 360.0)
        return v

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xs, [r["dE"] for r in res["results"]], marker="o", ms=4, color="#1f77b4", label="Relaxed scan")
    if dmf_res is not None:
        a = res["results"][0]["atoms"].copy()
        px = []
        for pos in dmf_res["frames"]:
            a.positions = pos
            px.append(x_of(a))
        py = (np.array(dmf_res["energies"]) - e_ref) / EV_PER_KCALMOL
        ax.plot(px, py, ls="--", marker=".", ms=4, color="#2ca02c", alpha=0.8, label="DMF path")
    for label, atoms, energy, style in points:
        x, y = x_of(atoms), (energy - e_ref) / EV_PER_KCALMOL
        unit = "Å" if spec.code == "R" else "deg"  # Colab には日本語フォントがないので「度」は使わない
        ax.plot(x, y, ls="none", zorder=5, label=f"{label} ({x:.2f} {unit}, {y:.1f} kcal/mol)", **style)
    ax.margins(y=0.12)
    ax.set_xlabel(f"{spec.xlabel}: {spec.atoms_label}")
    ax.set_ylabel("Relative energy (kcal/mol)")
    ax.set_title("Relaxed scan with DMF path and TS")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def report(res, job, atoms0):
    """プロファイル・表・注意を表示し、(出力ファイルの dict, ログの行) を返す。"""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    eh = gaussian.EV_PER_HARTREE
    sc, spec = res["scan"], res["scan"].spec
    results, lowest, highest, post_index = res["results"], res["lowest"], res["highest"], res["post_index"]
    pre_result, post_result = res["pre"], res["post"]
    symbols = atoms0.get_chemical_symbols()
    converged_results = [r for r in results if r["converged"]]

    scan_xyz = "".join(
        xyz_block(symbols, r["atoms"].positions,
                  f"point={r['point']} {spec.definition}={r['value']:.4f} "
                  f"energy_eV={r['energy']:.8f} energy_Ha={r['energy'] / eh:.10f} dE_kcal_mol={r['dE']:.4f} "
                  f"fmax={r['fmax']:.5f} converged={r['converged']}")
        for r in converged_results)

    # --- エネルギープロファイル（Colabは日本語フォントがないため軸ラベルは英語） ---
    xs = [r["target"] for r in results]
    ys = [r["dE"] for r in results]
    bad = [r for r in results if not r["converged"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xs, ys, marker="o", color="#1f77b4", label="Converged")
    if bad:
        ax.plot([r["target"] for r in bad], [r["dE"] for r in bad], ls="", marker="o", mfc="white", color="#d62728",
                label="Not converged")
    if post_index is not None:
        ax.plot(xs[post_index], ys[post_index], ls="", marker="*", ms=16, color="#ff7f0e", label="Start of post-scan opt")
    if bad or post_index is not None:
        ax.legend()
    ax.set_xlabel(f"{spec.xlabel}: {spec.atoms_label}")
    ax.set_ylabel("Relative energy (kcal/mol)")
    ax.set_title("Relaxed scan")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()

    # --- 表 ---
    display(pd.DataFrame([{
        "Point": r["point"],
        f"Target ({spec.unit})": round(r["target"], 4),
        f"Actual ({spec.unit})": round(r["value"], 4),
        "Energy (eV)": round(r["energy"], 6),
        "Energy (Hartree)": round(r["energy"] / eh, 8),
        "ΔE (kcal/mol)": round(r["dE"], 3),
        "fmax (eV/Å)": round(r["fmax"], 4),
        "Steps": r["steps"],
        "Converged": "yes" if r["converged"] else "no",
        "": " ".join(t for t, c in (("lowest", r is lowest), ("highest", r is highest),
                                    ("post-scan opt start", post_index == r["point"] - 1)) if c),
    } for r in results]).style.hide(axis="index"))

    print(f"Lowest: point {lowest['point']} ({lowest['target']:.3f} {spec.unit})  "
          f"Highest: point {highest['point']} ({highest['target']:.3f} {spec.unit})  "
          f"height {highest['dE']:.2f} kcal/mol")

    if bad:
        warn(f"{len(bad)} points did not converge within the maximum steps ({job.max_steps}): "
             f"{', '.join(str(r['point']) for r in bad)}. Their energies are not reliable, so they are left out of "
             "scan.xyz and scan.log (scan.log numbers only the converged points).",
             "Increase the maximum steps, loosen fmax, or use a smaller step size, then run again.")
    off = [r for r in results if abs(r["deviation"]) > SCAN_TOLERANCE]
    if off:
        warn(f"{len(off)} points are more than {SCAN_TOLERANCE} {spec.unit} off their target: "
             f"{', '.join(str(r['point']) for r in off)}.", "Use a smaller step size and run again.")
    if spec.code == "D" and abs(abs(spec.n_steps * spec.step_size) - 360.0) < 1e-6:
        gap = abs(results[-1]["dE"] - results[0]["dE"])
        if gap > 0.5:
            warn(f"After a full turn back to the same angle, the energy differs by {gap:.2f} kcal/mol. "
                 "The scan may have moved to another conformer.",
                 "Scan in the opposite direction too and compare, or use a smaller step size.")

    # --- 前後の最適化の結果 ---
    free = [r for r in (pre_result, post_result) if r is not None]
    if free:
        display(pd.DataFrame([{
            "": r["label"],
            f"{spec.name} ({spec.unit})": round(r["value"], 4),
            "Energy (eV)": round(r["energy"], 6),
            "Energy (Hartree)": round(r["energy"] / eh, 8),
            f"ΔE (vs {free[0]['label']}, kcal/mol)": round((r["energy"] - free[0]["energy"]) / EV_PER_KCALMOL, 3),
            "fmax (eV/Å)": round(r["fmax"], 4),
            "Steps": r["steps"],
            "Converged": "yes" if r["converged"] else "no",
        } for r in free]).style.hide(axis="index"))
        for r in free:
            if not r["converged"]:
                warn(f"{r['label']} did not converge within the maximum steps ({job.max_steps}).",
                     "Increase the maximum steps or loosen fmax, then run again.")
    if post_index == len(results) - 1:
        warn("The last scan point was picked as the minimum. The energy may still have been falling at the end of the scan.",
             "Extend the scan range past the minimum to be sure.")
    elif post_index == 0:
        warn("Going back reached the first point. No minimum was found along the scan, so the post-scan optimization "
             "returns to the starting valley.", "Check the scan range.")

    # --- 各点の構造のアニメーション ---
    if converged_results:
        show_animation(scan_xyz)

    # --- 出力ファイル ---
    files = {}
    csv_df = pd.DataFrame([{
        "point": r["point"], f"target_{spec.code}": round(r["target"], 6), f"value_{spec.code}": round(r["value"], 6),
        "energy_eV": round(r["energy"], 8), "energy_Ha": round(r["energy"] / eh, 10),
        "dE_kcal_mol": round(r["dE"], 4), "fmax_eV_A": round(r["fmax"], 5), "steps": r["steps"], "converged": r["converged"],
    } for r in results])
    files["scan.csv"] = csv_df.to_csv(index=False)
    if converged_results:
        files["scan.xyz"] = scan_xyz
        files["scan.log"] = gaussian_scan_log(res, job, atoms0.numbers, converged_results)
    else:
        warn("No point converged, so scan.xyz and scan.log were not written.")
    for tag, r in (("preopt", pre_result), ("postopt", post_result)):
        if r is not None:
            files[f"{tag}.xyz"] = xyz_block(symbols, r["atoms"].positions,
                                            f"{tag} {spec.definition}={r['value']:.4f} "
                                            f"energy_eV={r['energy']:.8f} energy_Ha={r['energy'] / eh:.10f} "
                                            f"fmax={r['fmax']:.5f} converged={r['converged']}")
            files[f"{tag}.log"] = gaussian.opt_log(r, job, atoms0.numbers, program="ColabUMA Scan")

    log = ["[Scan settings]",
           f"Scan coordinate: {spec.name} {spec.atoms_label}",
           f"Start: {res['targets'][0]:.4f} {spec.unit}", f"Number of steps: {spec.n_steps}",
           f"Step size: {spec.step_size} {spec.unit}",
           f"Pre-scan optimization: {'on' if pre_result is not None else 'off'}",
           f"Post-scan optimization: {'on' if spec.opt_after else 'off'}",
           "",
           "[Scan result]",
           f"Lowest: point {lowest['point']}  E = {lowest['energy']:.8f} eV ({lowest['energy'] / eh:.10f} Ha)",
           f"Highest: point {highest['point']}  E = {highest['energy']:.8f} eV ({highest['energy'] / eh:.10f} Ha)"]
    if post_index is not None:
        log.append(f"Post-scan optimization starts from: point {post_index + 1}")
    for r in free:
        log.append(f"{r['label']}: {spec.name} = {r['value']:.4f} {spec.unit}  E = {r['energy']:.8f} eV "
                   f"({r['energy'] / eh:.10f} Ha)  fmax = {r['fmax']:.5f}  steps = {r['steps']}  converged = {r['converged']}")
    for r in results:
        log += ["", f"[Point {r['point']}] target = {r['target']:.4f} {spec.unit}  actual = {r['value']:.4f} {spec.unit}  "
                    f"steps = {r['steps']}  converged = {r['converged']}",
                f"{'step':>5s} {'energy_eV':>18s} {'fmax_eV/A':>12s}"]
        log += [f"{i:5d} {e:18.8f} {fm:12.5f}" for i, (e, fm) in enumerate(zip(r["hist"]["energy"], r["hist"]["fmax"]))]
    return files, log
