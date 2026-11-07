"""緩和スキャン（結合長・角度・二面角）。Gaussian の opt=modredundant の S 指定に相当する。"""
from dataclasses import dataclass, field

import numpy as np

from . import gaussian, opt
from .structure import show_animation, show_numbered, xyz_block
from .ui import ColabUMAError, ProgressDisplay, bar_row, ok, warn

SCAN_KINDS = {2: ("R", "結合長", "Å", "Distance (Å)"), 3: ("A", "角度", "度", "Angle (deg)"),
              4: ("D", "二面角", "度", "Dihedral (deg)")}
SCAN_TOLERANCE = 0.01  # Å または 度。目標値と実測値の差がこれを超えたら注意を出す
MIN_PROMINENCE = 0.1  # kcal/mol。極小を探すとき、これより小さい凹凸は無視する
EV_PER_KCALMOL = 0.043364104241800934  # ase.units.kcal / ase.units.mol

_ATOMS_HINT = "2個なら結合長、3個なら角度、4個なら二面角です（例：1 2 3 4）。"
_COLLINEAR_HINT = "原子の指定を変えてください。直線に近い原子の組では、角度や二面角を決められません。"


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
        raise ColabUMAError(f"scan_atoms「{scan_atoms}」を原子番号として読み取れませんでした。", _ATOMS_HINT) from None
    if len(idx) not in (2, 3, 4):
        raise ColabUMAError(f"scan_atoms には原子番号を2〜4個指定してください（今は {len(idx)} 個です）。", _ATOMS_HINT)
    if min(idx) < 0:
        raise ColabUMAError(f"scan_atoms の原子番号は1以上で指定してください（今は {scan_atoms} です）。", _ATOMS_HINT)
    if len(set(idx)) != len(idx):
        raise ColabUMAError(f"scan_atoms に同じ原子番号が入っています（{scan_atoms}）。", _ATOMS_HINT)
    if n_steps < 1:
        raise ColabUMAError("刻みの回数は1以上にしてください。", "計算する点の数は、刻みの回数に1を足した数です。")
    if step_size == 0:
        raise ColabUMAError("刻み幅に0は指定できません。", "結合長なら 0.1 Å、角度や二面角なら 10 度 程度が目安です。")
    start = None
    if str(scan_start).strip():
        try:
            start = float(scan_start)
        except ValueError:
            raise ColabUMAError(f"scan_start「{scan_start}」を数値として読み取れませんでした。",
                                "空欄にすると入力構造の値から始めます。") from None
    if len(idx) == 4 and abs(n_steps * step_size) > 360.0 + 1e-6:
        warn("二面角を1回転より多く動かす設定です。同じ角度を重複して計算します。")
    return ScanSpec(idx, start, n_steps, step_size, opt_before, opt_after, scan_atoms)


def check_atom_range(spec, n_atoms, hint):
    if n_atoms and max(spec.idx) >= n_atoms:
        raise ColabUMAError(f"scan_atoms の原子番号は1〜{n_atoms}の範囲で指定してください（今は {spec.text} です）。", hint)


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
            raise ColabUMAError(f"指定した3原子がほぼ直線上に並んでいます（{atoms.get_angle(*s.idx):.1f} 度）。", _COLLINEAR_HINT)
        if s.code == "D":
            for tri in (s.idx[:3], s.idx[1:]):
                if not 5.0 < atoms.get_angle(*tri) < 175.0:
                    raise ColabUMAError(f"二面角を作る3原子（{' '.join(str(i + 1) for i in tri)}）がほぼ直線上に並んでいます。",
                                        _COLLINEAR_HINT)

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
            raise ColabUMAError(f"結合長が {min(targets):.3f} Å まで短くなる設定です。", "開始値、刻みの回数、刻み幅を見直してください。")
        if s.code == "A" and not (0.0 < min(targets) and max(targets) < 180.0):
            raise ColabUMAError(f"角度が {min(targets):.1f}〜{max(targets):.1f} 度の範囲を動く設定です。",
                                "角度は0度と180度を含まない範囲で指定してください。180度ちょうどでは拘束を定義できません。")
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
        print("原子番号（赤がスキャンに指定した原子です）：")
        show_numbered(atoms0, spec.idx)
    check_atom_range(spec, len(atoms0), "上の3D表示で番号を確認してください。")
    sc = Scan(spec, atoms0)
    sc.check_geometry(atoms0)
    targets = sc.targets(atoms0)
    if sc.in_ring:
        warn(f"指定した{spec.name}は環の中にあるため、原子 {spec.idx[-1] + 1} だけを動かして各点を始めます。"
             "各点の初期構造が大きくひずむことがあります。",
             "エネルギーが不連続になるときは、刻み幅を小さくしてください。")
    ok(f"{spec.name} {spec.atoms_label} を {len(targets)} 点スキャンします（入力構造の値 {sc.value(atoms0):.3f} {spec.unit}）。")
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
        return [bar_row("スキャン", state["done"] / state["n"], f"{state['done']}/{state['n']} 点完了")]

    def free_opt(start_atoms, label):
        r = opt.optimize(start_atoms, calc, job, label, progress=progress, extra_rows=extra_rows, quiet=True)
        r["value"] = sc.value(r["atoms"])
        print(f"  {label}: {spec.name} = {r['value']:.3f} {spec.unit}  E = {r['energy']:.6f} eV  fmax = {r['fmax']:.4f} eV/Å  "
              f"{r['steps']:4d} ステップ  {'収束' if r['converged'] else '未収束'}（{r['seconds']:.1f} 秒）")
        return r

    pre_result = None
    current = atoms0
    if skip_preopt_result is not None:
        pre_result = dict(skip_preopt_result, value=sc.value(skip_preopt_result["atoms"]))
        current = pre_result["atoms"]
    elif spec.opt_before:
        print(f"スキャンの前に {job.optimizer_name} で最適化します（fmax < {job.fmax} eV/Å、最大 {job.max_steps} ステップ）")
        pre_result = free_opt(atoms0, "前の最適化")
        current = pre_result["atoms"]
    sc.check_geometry(current)
    targets = sc.targets(current)
    state["n"] = len(targets)

    print(f"{job.optimizer_name} で各点を最適化します（fmax < {job.fmax} eV/Å、1点あたり最大 {job.max_steps} ステップ）")
    results = []
    for k, target in enumerate(targets):
        atoms = current.copy()
        atoms.set_constraint()
        sc.drive(atoms, target)
        atoms.set_constraint(sc.constraint(target))
        r = opt.run(atoms, calc, job, f"点 {k + 1}", progress=progress, extra_rows=extra_rows, quiet=True)
        value = sc.value(atoms)
        deviation = wrap180(value - target) if spec.code == "D" else value - target
        print(f"  点 {k + 1:3d}/{len(targets)}: {spec.name} = {target:9.3f} {spec.unit}  E = {r['energy']:.6f} eV  "
              f"fmax = {r['fmax']:.4f} eV/Å  {r['steps']:4d} ステップ  {'収束' if r['converged'] else '未収束'}"
              f"（{r['seconds']:.1f} 秒）")
        r.update({"point": k + 1, "target": target, "value": target + deviation, "deviation": deviation})
        results.append(r)
        current = r["atoms"]
        state["done"] = k + 1
        progress.show(extra_rows(), force=True)

    post_result = post_index = None
    if spec.opt_after:
        post_index = last_minimum_index([r["energy"] for r in results], [r["converged"] for r in results])
        print(f"最後の点からさかのぼって最初の極小は点 {post_index + 1} です。ここから拘束なしで最適化します。")
        post_result = free_opt(results[post_index]["atoms"], "後の最適化")

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
        "点": r["point"],
        f"目標値 ({spec.unit})": round(r["target"], 4),
        f"実測値 ({spec.unit})": round(r["value"], 4),
        "エネルギー (eV)": round(r["energy"], 6),
        "エネルギー (Hartree)": round(r["energy"] / eh, 8),
        "ΔE (kcal/mol)": round(r["dE"], 3),
        "fmax (eV/Å)": round(r["fmax"], 4),
        "ステップ数": r["steps"],
        "収束": "はい" if r["converged"] else "いいえ",
        "": " ".join(t for t, c in (("最安定", r is lowest), ("最高点", r is highest),
                                    ("後の最適化の開始点", post_index == r["point"] - 1)) if c),
    } for r in results]).style.hide(axis="index"))

    print(f"最安定: 点 {lowest['point']}（{lowest['target']:.3f} {spec.unit}）  "
          f"最高点: 点 {highest['point']}（{highest['target']:.3f} {spec.unit}）  "
          f"高さ {highest['dE']:.2f} kcal/mol")

    if bad:
        warn(f"最大ステップ数（{job.max_steps}）までに収束しなかった点が {len(bad)} 個あります："
             f"{', '.join(str(r['point']) for r in bad)}。これらの点のエネルギーは信頼できないため、"
             "scan.xyz と scan.log には入れていません（scan.log の点の番号は収束した点だけで振り直しています）。",
             "最大ステップ数を増やすか、fmax を緩めるか、刻み幅を小さくしてもう一度実行してください。")
    off = [r for r in results if abs(r["deviation"]) > SCAN_TOLERANCE]
    if off:
        warn(f"目標値から {SCAN_TOLERANCE} {spec.unit} より大きくずれた点が {len(off)} 個あります："
             f"{', '.join(str(r['point']) for r in off)}。", "刻み幅を小さくしてもう一度実行してください。")
    if spec.code == "D" and abs(abs(spec.n_steps * spec.step_size) - 360.0) < 1e-6:
        gap = abs(results[-1]["dE"] - results[0]["dE"])
        if gap > 0.5:
            warn(f"1回転して同じ角度に戻ったのに、エネルギーが {gap:.2f} kcal/mol 違います。途中で別の配座に移った可能性があります。",
                 "逆向きにもスキャンして比較するか、刻み幅を小さくしてください。")

    # --- 前後の最適化の結果 ---
    free = [r for r in (pre_result, post_result) if r is not None]
    if free:
        display(pd.DataFrame([{
            "": r["label"],
            f"{spec.name} ({spec.unit})": round(r["value"], 4),
            "エネルギー (eV)": round(r["energy"], 6),
            "エネルギー (Hartree)": round(r["energy"] / eh, 8),
            f"ΔE（{free[0]['label']}基準, kcal/mol）": round((r["energy"] - free[0]["energy"]) / EV_PER_KCALMOL, 3),
            "fmax (eV/Å)": round(r["fmax"], 4),
            "ステップ数": r["steps"],
            "収束": "はい" if r["converged"] else "いいえ",
        } for r in free]).style.hide(axis="index"))
        for r in free:
            if not r["converged"]:
                warn(f"{r['label']}は最大ステップ数（{job.max_steps}）までに収束しませんでした。",
                     "最大ステップ数を増やすか、fmax を緩めてもう一度実行してください。")
    if post_index == len(results) - 1:
        warn("スキャンの最後の点が極小として選ばれました。スキャンの端でもエネルギーが下がり続けていた可能性があります。",
             "スキャンの範囲を広げて、極小を過ぎるところまで動かすと確実です。")
    elif post_index == 0:
        warn("さかのぼって最初の点まで戻りました。スキャンの途中に極小が見つからなかったため、後の最適化は出発点の谷に戻っています。",
             "スキャンの範囲を見直してください。")

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
        warn("収束した点がないため、scan.xyz と scan.log は作りませんでした。")
    for tag, r in (("preopt", pre_result), ("postopt", post_result)):
        if r is not None:
            files[f"{tag}.xyz"] = xyz_block(symbols, r["atoms"].positions,
                                            f"{tag} {spec.definition}={r['value']:.4f} "
                                            f"energy_eV={r['energy']:.8f} energy_Ha={r['energy'] / eh:.10f} "
                                            f"fmax={r['fmax']:.5f} converged={r['converged']}")
            files[f"{tag}.log"] = gaussian.opt_log(r, job, atoms0.numbers, program="ColabUMA Scan")

    log = ["[スキャン条件]",
           f"スキャン座標: {spec.name} {spec.atoms_label}",
           f"開始値: {res['targets'][0]:.4f} {spec.unit}", f"刻みの回数: {spec.n_steps}", f"刻み幅: {spec.step_size} {spec.unit}",
           f"前の最適化: {'オン' if pre_result is not None else 'オフ'}", f"後の最適化: {'オン' if spec.opt_after else 'オフ'}",
           "",
           "[スキャンの結果]",
           f"最安定: 点 {lowest['point']}  E = {lowest['energy']:.8f} eV ({lowest['energy'] / eh:.10f} Ha)",
           f"最高点: 点 {highest['point']}  E = {highest['energy']:.8f} eV ({highest['energy'] / eh:.10f} Ha)"]
    if post_index is not None:
        log.append(f"後の最適化の開始点: 点 {post_index + 1}")
    for r in free:
        log.append(f"{r['label']}: {spec.name} = {r['value']:.4f} {spec.unit}  E = {r['energy']:.8f} eV "
                   f"({r['energy'] / eh:.10f} Ha)  fmax = {r['fmax']:.5f}  steps = {r['steps']}  converged = {r['converged']}")
    for r in results:
        log += ["", f"[点 {r['point']}] 目標値 = {r['target']:.4f} {spec.unit}  実測値 = {r['value']:.4f} {spec.unit}  "
                    f"steps = {r['steps']}  converged = {r['converged']}",
                f"{'step':>5s} {'energy_eV':>18s} {'fmax_eV/A':>12s}"]
        log += [f"{i:5d} {e:18.8f} {fm:12.5f}" for i, (e, fm) in enumerate(zip(r["hist"]["energy"], r["hist"]["fmax"]))]
    return files, log
