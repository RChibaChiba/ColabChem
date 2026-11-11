"""配座探索。UMA の高温 MD で構造を集め、最適化して重複を除き、エネルギーの低い順に並べる。

流れ：入力構造の最適化 → 結合のつながりを読み取る → 高温 MD（つながりを保つ拘束と、離れていかないための壁つき）
→ スナップショットを粗く最適化 → 低いものを最後まで最適化 → つながりが変わったものと重複を除く → 上位を出力。

金属に配位しているか、非共有結合で付いているだけかは指定しない。最適化した入力構造で結合している原子の組を
そのまま保つので、配位していれば配位を保ち、付いているだけなら基質は自由に動く。
"""
import time
from dataclasses import dataclass

import numpy as np

from . import opt
from .ui import ColabUMAError, ProgressDisplay, bar_row, heading, ok, warn

EV_PER_KCALMOL = 0.043364104241800934  # ase.units.kcal / ase.units.mol
KT_298 = 0.0019872041 * 298.15  # kcal/mol

# --- 結合の判定（共有結合半径の和に対する比） ---
BOND_FACTOR = 1.25  # 入力構造で、距離がこの比より短ければ結合
BREAK_FACTOR = 1.15  # 候補の構造で、結合の判定距離のこの倍より長ければ「切れた」
FORM_FACTOR = 0.90  # 候補の構造で、結合していなかった組が判定距離のこの倍より短ければ「できた」
NONMETALS = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36, 51, 52, 53, 54}
STEREO_MIN_VOLUME = 1.0  # 4配位の中心を立体中心として見張る下限（正四面体で約 3.1、平面で 0）

# --- MD ---
TIMESTEP_FS = 2.0
H_MASS = 4.0  # MD のあいだだけ水素を重くして、2 fs の刻みで安定に動かす
FRICTION_PER_FS = 0.02  # Langevin の摩擦
BOND_STRETCH = 1.2  # 結合が入力構造の長さのこの倍 + BOND_SLACK を超えたら引き戻す
BOND_SLACK = 0.1  # Å
K_BOND = 5.0  # eV/Å^2
WALL_MARGIN = 3.0  # Å。入力構造の半径にこれを足した球の外に出た原子を押し戻す
K_WALL = 1.0  # eV/Å^2

# --- 最適化と重複の判定 ---
PRE_FMAX = 0.1  # 粗い最適化の収束判定（eV/Å）
PRE_STEPS = 60
PRE_WINDOW = 15.0  # kcal/mol。粗い最適化のあと、最安定からこれより高いものは最後まで最適化しない
RMSD_TOL = 0.25  # Å（水素以外、重ね合わせ後）
MAXDEV_TOL = 0.7  # Å。RMSD が小さくても、どれかの原子がこれより動いていれば別の配座
E_TOL = 0.1  # kcal/mol
INERTIA_TOL = 0.01  # 慣性モーメントの相対差。エネルギーとこれが両方近ければ、原子の入れ替えで重なる同じ配座とみなす


@dataclass
class Settings:
    n_final: int = 5
    temperature: float = 600.0
    md_time_ps: float = 20.0
    n_replicas: int = 4
    n_snapshots: int = 100
    n_refine: int = 20
    keep_coordination: bool = True
    seed: int = 0

    @property
    def steps_per_snapshot(self):
        total = self.md_time_ps * 1000.0 / TIMESTEP_FS
        return max(int(round(total / self.n_snapshots)), 1)

    @property
    def snapshots_per_replica(self):
        return int(np.ceil(self.n_snapshots / self.n_replicas))

    def lines(self):
        return [f"Conformers to report: {self.n_final}", f"MD temperature: {self.temperature} K",
                f"MD time (all replicas): {self.md_time_ps} ps, time step {TIMESTEP_FS} fs, H mass {H_MASS} amu",
                f"MD replicas: {self.n_replicas}", f"Snapshots: {self.n_snapshots}",
                f"Candidates optimized fully: {self.n_refine}",
                f"Keep metal coordination: {'yes' if self.keep_coordination else 'no'}", f"Random seed: {self.seed}"]


def parse_settings(n_conformers, md_temperature, md_time_ps, n_replicas, n_snapshots, n_refine, keep_coordination, seed):
    """フォームの値を確かめて Settings にする。"""
    def bad(msg, hint, ja, ja_hint):
        raise ColabUMAError(msg, hint, ja=ja, ja_hint=ja_hint)

    if not 1 <= n_conformers <= 50:
        bad("The number of conformers must be between 1 and 50.", "A typical value is 5.",
            "出力する配座の数は 1〜50 にしてください。", "通常は 5 程度です。")
    if not 100 <= md_temperature <= 2000:
        bad("The MD temperature must be between 100 and 2000 K.", "A typical value is 600 K.",
            "MD の温度は 100〜2000 K にしてください。", "通常は 600 K 程度です。")
    if md_time_ps <= 0:
        bad("The MD time must be positive.", "A typical value is 20 ps.",
            "MD の時間は正の数にしてください。", "通常は 20 ps 程度です。")
    if n_replicas < 1:
        bad("The number of MD replicas must be 1 or more.", "A typical value is 4.",
            "MD の本数は 1 以上にしてください。", "通常は 4 程度です。")
    if n_snapshots < max(n_conformers, n_replicas):
        bad("The number of snapshots must be at least the number of conformers and of MD replicas.",
            "A typical value is 100.", "スナップショットの数は、出力する配座の数と MD の本数以上にしてください。",
            "通常は 100 程度です。")
    if n_refine < n_conformers:
        bad("The number of candidates optimized fully must be at least the number of conformers.",
            "A typical value is 4 times the number of conformers.",
            "最後まで最適化する候補の数は、出力する配座の数以上にしてください。", "通常は配座の数の 4 倍程度です。")
    if md_time_ps * 1000.0 / TIMESTEP_FS < n_snapshots:
        bad(f"The MD is too short for {n_snapshots} snapshots (time step {TIMESTEP_FS} fs).",
            "Make the MD longer or take fewer snapshots.",
            f"MD が短すぎて {n_snapshots} 個のスナップショットを取れません（刻み {TIMESTEP_FS} fs）。",
            "MD の時間を長くするか、スナップショットの数を減らしてください。")
    return Settings(int(n_conformers), float(md_temperature), float(md_time_ps), int(n_replicas), int(n_snapshots),
                    int(n_refine), bool(keep_coordination), int(seed))


# --- 結合のつながり ---
def _distances(pos):
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)


def _signed_volume(pos, center, nbrs):
    u = pos[nbrs] - pos[center]
    u /= np.linalg.norm(u, axis=1)[:, None]
    return float(np.linalg.det(u[:3] - u[3]))


class Topology:
    """最適化した入力構造の結合のつながり。候補の構造が同じつながりかを調べるのに使う。"""

    def __init__(self, atoms, keep_coordination=True):
        from ase.data import covalent_radii

        z = atoms.numbers
        pos = atoms.positions
        n = len(atoms)
        rsum = covalent_radii[z][:, None] + covalent_radii[z][None, :]
        d = _distances(pos)
        upper = np.triu(np.ones((n, n), bool), k=1)
        self.is_metal = np.array([int(v) not in NONMETALS for v in z])
        metal_pair = self.is_metal[:, None] | self.is_metal[None, :]
        all_bonds = (d < BOND_FACTOR * rsum) & upper
        # 見張る原子の組。配位を保たないときは、金属が関わる組を見ない
        self.watched = upper if keep_coordination else upper & ~metal_pair
        self.bonded = all_bonds & self.watched
        self.pairs = np.argwhere(self.bonded)
        self.d_ref = d[self.bonded]
        self.break_at = (BOND_FACTOR * BREAK_FACTOR * rsum)[self.bonded]
        self.form_at = BOND_FACTOR * FORM_FACTOR * rsum
        self.n_metal_bonds = int((all_bonds & metal_pair).sum())
        self.keep_coordination = keep_coordination
        self.heavy = np.flatnonzero(z > 1) if (z > 1).sum() >= 3 else np.arange(n)
        self.radius = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).max())

        # つながっていない部分（フラグメント）の数
        label = np.arange(n)
        for i, j in self.pairs:
            a, b = label[i], label[j]
            if a != b:
                label[label == b] = a
        self.n_fragments = len(set(label))

        # 金属以外の4配位の中心は、向き（立体配置）も見張る
        nbrs = {i: [] for i in range(n)}
        for i, j in self.pairs:
            nbrs[int(i)].append(int(j))
            nbrs[int(j)].append(int(i))
        self.stereo = []
        for i, nb in nbrs.items():
            if len(nb) == 4 and not self.is_metal[i]:
                v = _signed_volume(pos, i, nb)
                if abs(v) > STEREO_MIN_VOLUME:
                    self.stereo.append((i, nb, v > 0))

    def lines(self):
        metals = int(self.is_metal.sum())
        out = [f"Bonds kept: {len(self.pairs)} ({metals} metal atom{'s' if metals != 1 else ''}, "
               f"{self.n_metal_bonds} metal–ligand contacts"
               f"{'' if self.keep_coordination else ', not kept'})",
               f"Fragments not bonded to each other: {self.n_fragments}",
               f"Stereocentres watched: {len(self.stereo)}"]
        return out

    def check(self, pos):
        """つながりが入力と同じなら None、違えば理由。"""
        d = _distances(pos)
        if (d[self.pairs[:, 0], self.pairs[:, 1]] > self.break_at).any():
            return "bond broken"
        if ((d < self.form_at) & self.watched & ~self.bonded).any():
            return "bond formed"
        for i, nb, positive in self.stereo:
            if (_signed_volume(pos, i, nb) > 0) != positive:
                return "stereocentre inverted"
        return None


class Restraint:
    """MD のあいだだけ掛ける拘束（ASE の constraint として渡す）。

    結合が伸びすぎたら引き戻し（底が平らなばね）、分子の重心から遠く離れた原子を押し戻す（球の壁）。
    ふつうの揺れの範囲では力は 0 なので、配座の動きは妨げない。
    """

    def __init__(self, topo):
        self.pairs = topo.pairs
        self.r_max = BOND_STRETCH * topo.d_ref + BOND_SLACK
        self.wall = topo.radius + WALL_MARGIN

    def _evaluate(self, pos):
        f = np.zeros_like(pos)
        e = 0.0
        if len(self.pairs):
            i, j = self.pairs[:, 0], self.pairs[:, 1]
            v = pos[j] - pos[i]
            r = np.linalg.norm(v, axis=1)
            over = np.clip(r - self.r_max, 0.0, None)
            if over.any():
                e += 0.5 * K_BOND * float((over ** 2).sum())
                g = (K_BOND * over / r)[:, None] * v
                np.add.at(f, i, g)
                np.add.at(f, j, -g)
        v = pos - pos.mean(axis=0)
        r = np.linalg.norm(v, axis=1)
        over = np.clip(r - self.wall, 0.0, None)
        if over.any():
            e += 0.5 * K_WALL * float((over ** 2).sum())
            g = (K_WALL * over / np.where(r > 0, r, 1.0))[:, None] * v
            f -= g - g.mean(axis=0)  # 重心も一緒に動くので、全体の力の和は 0
        return e, f

    def adjust_positions(self, atoms, new):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass

    def adjust_forces(self, atoms, forces):
        forces += self._evaluate(atoms.positions)[1]

    def adjust_potential_energy(self, atoms):
        return self._evaluate(atoms.positions)[0]

    def get_removed_dof(self, atoms):
        return 0

    def todict(self):
        return {"name": "ColabChemRestraint", "kwargs": {}}


# --- 重ね合わせと重複の判定 ---
def align(p, q, sel):
    """p を q に重ねる（sel の原子で最小二乗、鏡像にはしない）。(動かした p, RMSD, 最大のずれ) を返す。"""
    pc, qc = p[sel].mean(axis=0), q[sel].mean(axis=0)
    u, _, vt = np.linalg.svd((p[sel] - pc).T @ (q[sel] - qc))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    moved = (p - pc) @ rot.T + qc
    dev = np.linalg.norm(moved[sel] - q[sel], axis=1)
    return moved, float(np.sqrt((dev ** 2).mean())), float(dev.max())


def _inertia(atoms):
    return np.sort(atoms.get_moments_of_inertia())


def _same(a, b, sel, use_energy=True):
    if use_energy and abs(a["energy"] - b["energy"]) / EV_PER_KCALMOL < E_TOL:
        ia, ib = a["inertia"], b["inertia"]
        if np.all(np.abs(ia - ib) <= INERTIA_TOL * np.maximum(ia, 1e-6)):
            return True
    _, rmsd, maxdev = align(a["atoms"].positions, b["atoms"].positions, sel)
    return rmsd < RMSD_TOL and maxdev < MAXDEV_TOL


def unique(cands, sel, use_energy=True):
    """エネルギーの低い順に並べ、すでに残したものと同じ配座を除く。(残したもの, 除いた数) を返す。"""
    kept = []
    for c in sorted(cands, key=lambda c: c["energy"]):
        if "inertia" not in c:
            c["inertia"] = _inertia(c["atoms"])
        if not any(_same(c, k, sel, use_energy) for k in kept):
            kept.append(c)
    return kept, len(cands) - len(kept)


def _eta(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# --- 高温 MD ---
def sample(atoms, calc, topo, st):
    """高温の Langevin MD を st.n_replicas 本走らせ、等間隔のスナップショットを返す。"""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md import velocitydistribution as vd
    from ase.md.velocitydistribution import Stationary, ZeroRotation

    # 新しい ASE では thermalize_momenta、古い ASE では MaxwellBoltzmannDistribution
    thermalize = getattr(vd, "thermalize_momenta", None) or vd.MaxwellBoltzmannDistribution

    every = st.steps_per_snapshot
    per_replica = st.snapshots_per_replica
    total_steps = every * per_replica * st.n_replicas
    print(f"MD at {st.temperature:.0f} K: {st.n_replicas} replica(s) × {every * per_replica * TIMESTEP_FS / 1000:.2f} ps "
          f"({total_steps} steps of {TIMESTEP_FS} fs), a snapshot every {every * TIMESTEP_FS / 1000:.2f} ps")
    progress = ProgressDisplay()
    snapshots, stopped = [], []
    done = 0
    t0 = time.time()
    for rep in range(st.n_replicas):
        a = atoms.copy()
        masses = a.get_masses()
        masses[a.numbers == 1] = H_MASS
        a.set_masses(masses)
        a.set_constraint(Restraint(topo))
        a.calc = calc
        rng = np.random.RandomState(st.seed + rep)
        thermalize(a, temperature_K=st.temperature, rng=rng)
        Stationary(a)
        ZeroRotation(a)
        kw = dict(temperature_K=st.temperature, friction=FRICTION_PER_FS / units.fs, rng=rng, logfile=None)
        try:  # 重心は固定しない（壁は重心から測るので、全体が動いても困らない）
            dyn = Langevin(a, TIMESTEP_FS * units.fs, fixcm=False, **kw)
        except TypeError:  # fixcm がなくなった ASE
            dyn = Langevin(a, TIMESTEP_FS * units.fs, **kw)
        for k in range(per_replica):
            try:
                dyn.run(every)
            except opt._oom_error():
                raise ColabUMAError("The GPU ran out of memory.", "Use a smaller model such as uma-s-1p1, or fewer atoms.",
                                    ja="GPUのメモリが足りません。",
                                    ja_hint="モデルを uma-s-1p1 などの小さいものにするか、原子数を減らしてください。") from None
            done += every
            temp = a.get_temperature()
            if not np.isfinite(a.positions).all() or not np.isfinite(temp) or temp > 5 * st.temperature:
                stopped.append(rep + 1)  # MD が壊れた。この MD のここまでのスナップショットは使う
                done = every * per_replica * (rep + 1)
                break
            snap = atoms.copy()
            snap.positions = a.positions.copy()
            snapshots.append({"atoms": snap, "source": f"MD {rep + 1}, {(k + 1) * every * TIMESTEP_FS / 1000:.2f} ps"})
            left = (time.time() - t0) / done * (total_steps - done)
            progress.show([bar_row("MD", done / total_steps,
                                   f"replica {rep + 1}/{st.n_replicas}　{temp:.0f} K　{len(snapshots)} snapshots　"
                                   f"about {_eta(left)} left")], force=done == total_steps)
    seconds = time.time() - t0
    progress.show([bar_row("MD", 1.0, f"{len(snapshots)} snapshots ({_eta(seconds)})")], force=True)
    if stopped:
        warn(f"MD replica {', '.join(map(str, stopped))} became unstable and was stopped early.",
             "Lower the MD temperature if this leaves too few snapshots.")
    if not snapshots:
        raise ColabUMAError("The MD became unstable before the first snapshot.", "Lower the MD temperature and run again.",
                            ja="最初のスナップショットを取る前に MD が不安定になりました。",
                            ja_hint="MD の温度を下げて、もう一度実行してください。")
    log = [f"MD: {len(snapshots)} snapshots in {seconds:.0f} s ({total_steps / max(seconds, 1e-9):.1f} steps/s)"]
    if stopped:
        log.append(f"MD replicas stopped early (unstable): {', '.join(map(str, stopped))}")
    return snapshots, log


# --- 最適化して選ぶ ---
def _optimize_all(cands, calc, job, title, fmax, max_steps):
    progress = ProgressDisplay()
    t0 = time.time()
    out = []
    for k, c in enumerate(cands):
        def overall(k=k):
            left = (time.time() - t0) / k * (len(cands) - k) if k else None
            return [bar_row(title, k / len(cands), f"{k}/{len(cands)} done" + (f"　about {_eta(left)} left" if left else ""))]

        r = opt.run(c["atoms"].copy(), calc, job, f"{title} {k + 1}", progress=progress, extra_rows=overall,
                    fmax=fmax, max_steps=max_steps, quiet=True)
        out.append({"atoms": r["atoms"], "energy": r["energy"], "fmax": r["fmax"], "steps": r["steps"],
                    "converged": r["converged"], "source": c["source"]})
    progress.show([bar_row(title, 1.0, f"{len(cands)}/{len(cands)} done ({_eta(time.time() - t0)})")], force=True)
    return out


def _drop_changed(cands, topo, counts):
    kept = []
    for c in cands:
        reason = topo.check(c["atoms"].positions)
        if reason is None:
            kept.append(c)
        else:
            counts[reason] = counts.get(reason, 0) + 1
    return kept


def run(atoms0, calc, job, st):
    """配座探索の全体。結果の dict を返す。"""
    heading("Optimizing the input structure")
    first = opt.optimize(atoms0, calc, job, "Input")
    opt.warn_unconverged(first)
    topo = Topology(first["atoms"], st.keep_coordination)
    for line in topo.lines():
        print("  " + line)
    if topo.n_fragments > 1:
        print("  The fragments are free to move against each other. A spherical wall keeps them from drifting apart.")

    heading("High-temperature MD")
    snapshots, md_log = sample(first["atoms"], calc, topo, st)

    heading("Quick optimization of the snapshots")
    print(f"{len(snapshots)} snapshots, at most {PRE_STEPS} steps each (fmax < {PRE_FMAX} eV/Å)")
    changed = {}
    rough = _drop_changed(_optimize_all(snapshots, calc, job, "Quick opt", PRE_FMAX, PRE_STEPS), topo, changed)
    rough, n_dup_rough = unique(rough, topo.heavy, use_energy=False)
    n_rough = len(rough)
    if rough:
        low = rough[0]["energy"]
        rough = [c for c in rough if (c["energy"] - low) / EV_PER_KCALMOL <= PRE_WINDOW][:st.n_refine]
    print(f"  {n_rough} different structures kept their bonding; the {len(rough)} lowest go on.")

    heading("Full optimization of the candidates")
    start = {"atoms": first["atoms"], "energy": first["energy"], "fmax": first["fmax"], "steps": first["steps"],
             "converged": first["converged"], "source": "input"}
    refined = []
    if rough:
        print(f"{len(rough)} candidates, at most {job.max_steps} steps each (fmax < {job.fmax} eV/Å)")
        refined = _drop_changed(_optimize_all(rough, calc, job, "Full opt", job.fmax, job.max_steps), topo, changed)
    ensemble, n_dup = unique([start] + refined, topo.heavy)
    final = ensemble[:st.n_final]
    for k, c in enumerate(ensemble):
        c["label"] = f"Conf {k + 1}"

    if changed:
        print("  Discarded because the bonding changed: " + ", ".join(f"{v} ({k})" for k, v in changed.items()))
    if len(final) < st.n_final:
        warn(f"Only {len(final)} different conformer(s) were found (asked for {st.n_final}).",
             "The molecule may be rigid. Otherwise make the MD longer or hotter, or take more snapshots.")
    else:
        ok(f"{len(ensemble)} different conformers found. The {len(final)} lowest are reported.")
    if final[0]["source"] != "input":
        d = (first["energy"] - final[0]["energy"]) / EV_PER_KCALMOL
        print(f"  The lowest conformer is {d:.2f} kcal/mol below the optimized input structure.")

    log = ["[Conformer search]"] + st.lines() + topo.lines() + md_log
    log += [f"Quick optimization: {len(snapshots)} snapshots, fmax {PRE_FMAX} eV/Å, at most {PRE_STEPS} steps",
            f"Different structures with the bonding kept: {n_rough} ({n_dup_rough} duplicates removed)",
            f"Fully optimized: {len(rough)} (within {PRE_WINDOW} kcal/mol of the lowest, at most {st.n_refine})",
            f"Different conformers after full optimization, with the optimized input: {len(ensemble)} "
            f"({n_dup} duplicates removed)",
            "Discarded because the bonding changed: "
            + (", ".join(f"{v} ({k})" for k, v in changed.items()) if changed else "0")]
    return {"first": first, "topology": topo, "ensemble": ensemble, "final": final, "settings": st, "log": log}


def dft_points(res):
    """DFT 一点計算に渡す (points, uma_rows)。"""
    return ([(c["label"], c["atoms"]) for c in res["final"]], [(c["label"], c["energy"]) for c in res["final"]])


# --- 結果の表示 ---
def _show_grid(xyz_blocks, labels):
    import py3Dmol

    n = len(xyz_blocks)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    view = py3Dmol.view(width=300 * cols, height=280 * rows, viewergrid=(rows, cols), linked=True)
    for k, (xyz, label) in enumerate(zip(xyz_blocks, labels)):
        cell = (k // cols, k % cols)
        view.addModel(xyz, "xyz", viewer=cell)
        view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.24}}, viewer=cell)
        view.addLabel(label, {"useScreen": True, "position": {"x": 8, "y": 8, "z": 0}, "fontSize": 12,
                              "fontColor": "black", "backgroundColor": "white", "backgroundOpacity": 0.8}, viewer=cell)
        view.zoomTo(viewer=cell)
    view.setBackgroundColor("white")
    view.show()


def report(res, job):
    """表・3D表示・エネルギーの図を出し、(出力ファイルの dict, ログの行) を返す。"""
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    from . import gaussian
    from .structure import xyz_block

    eh = gaussian.EV_PER_HARTREE
    final, ensemble, topo = res["final"], res["ensemble"], res["topology"]
    symbols = final[0]["atoms"].get_chemical_symbols()
    ref = final[0]["atoms"].positions
    e0 = final[0]["energy"]
    cond = (f"model={job.model_name} task={job.task_name} charge={job.charge} multiplicity={job.multiplicity} "
            f"optimizer={job.optimizer_name} fmax_target={job.fmax}")

    def rel(c):
        return (c["energy"] - e0) / EV_PER_KCALMOL

    weights = np.exp(-np.array([rel(c) for c in final]) / KT_298)
    weights /= weights.sum()
    rows, blocks = [], []
    for c, w in zip(final, weights):
        moved, rmsd, _ = align(c["atoms"].positions, ref, topo.heavy)
        c["aligned"] = moved
        rows.append({"": c["label"], "ΔE (kcal/mol)": round(rel(c), 2), "Population at 298 K (%)": round(100 * w, 1),
                     "E (eV)": round(c["energy"], 6), "E (Hartree)": round(c["energy"] / eh, 8),
                     "RMSD to Conf 1 (Å)": round(rmsd, 2), "fmax (eV/Å)": round(c["fmax"], 4),
                     "Converged": "yes" if c["converged"] else "no", "From": c["source"]})
        blocks.append(xyz_block(symbols, moved, f"{c['label']} dE_kcal={rel(c):.4f} energy_eV={c['energy']:.8f} "
                                                f"energy_Ha={c['energy'] / eh:.10f} fmax={c['fmax']:.5f} "
                                                f"converged={c['converged']} {cond}"))
    df = pd.DataFrame(rows)
    print(f"The {len(final)} lowest conformers (UMA energies; populations are among these, from ΔE only). "
          "RMSD is over non-hydrogen atoms.")
    display(df.style.hide(axis="index"))
    for c in final:
        if not c["converged"]:
            warn(f"{c['label']} did not converge within the maximum number of steps ({job.max_steps}).",
                 "Increase the maximum steps or loosen fmax, then run again.")
    print("3D view (rotating one rotates all; the structures are superimposed on Conf 1):")
    _show_grid(blocks, [f"{c['label']}  {rel(c):+.2f} kcal/mol" for c in final])

    # --- 見つかった配座すべてのエネルギー（Colabは日本語フォントがないため軸ラベルは英語） ---
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(1, len(ensemble) + 1)
    y = np.array([rel(c) for c in ensemble])
    n = len(final)
    ax.plot(x[n:], y[n:], "o", color="#9aa0a6", label="other conformers")
    ax.plot(x[:n], y[:n], "o", color="#1a73e8", label="reported")
    ax.set_xlabel("Conformer (sorted by energy)")
    ax.set_ylabel("ΔE (kcal/mol)")
    ax.set_title("Conformer energies (UMA)")
    ax.legend()
    fig.tight_layout()
    plt.show()

    first = res["first"]
    files = {"conformers.xyz": "".join(blocks), "conformers.csv": df.to_csv(index=False)}
    for k, block in enumerate(blocks):
        files[f"conf{k + 1}.xyz"] = block
    files["ensemble.xyz"] = "".join(
        xyz_block(symbols, align(c["atoms"].positions, ref, topo.heavy)[0],
                  f"{c['label']} dE_kcal={rel(c):.4f} energy_eV={c['energy']:.8f} converged={c['converged']} from={c['source']}")
        for c in ensemble)
    files["input_opt.xyz"] = xyz_block(symbols, first["atoms"].positions,
                                       f"optimized input energy_eV={first['energy']:.8f} fmax={first['fmax']:.5f} "
                                       f"converged={first['converged']} {cond}")
    log = res["log"] + ["", f"{'conformer':>10s} {'dE_kcal/mol':>12s} {'energy_eV':>18s} {'fmax_eV/A':>10s} "
                            f"{'steps':>6s} {'converged':>9s}  from"]
    log += [f"{c['label']:>10s} {rel(c):12.4f} {c['energy']:18.8f} {c['fmax']:10.5f} {c['steps']:6d} "
            f"{str(c['converged']):>9s}  {c['source']}" for c in ensemble]
    return files, log
