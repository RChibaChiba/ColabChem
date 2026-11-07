"""分子軌道（HOMO・LUMO など）：DFT の軌道を格子上で計算し、正と負の等値面を 3D 表示する。

軌道の値の単位は Bohr^(-3/2)。軌道の符号（位相）には意味がなく、計算ごとに全体が反転することがある。
開殻（UKS）では α と β の軌道を別々に扱う。
"""
import os
import re
import uuid

import numpy as np

from .esp import GRID_SPACING, MARGIN, _batches, cube_text, downsample, grid_quality, make_grid
from .ui import ColabUMAError, ProgressDisplay, bar_row, ok, warn

EV_PER_HARTREE = 27.211386245988

MAX_POINTS = 500_000  # 格子の点の数の上限。超えるなら格子を粗くする
DISPLAY_POINTS = 80_000  # 3D表示に渡す1軌道あたりの点の数。多いとノートの出力が重くなる
MAX_ORBITALS = 8  # 一度に表示する軌道の数の上限（開殻では α と β で倍になる）
TABLE_RANGE = 5  # エネルギーの表に出す、HOMO より下と LUMO より上の軌道の数
DIAGRAM_RANGE = 8  # 準位図に出す、HOMO より下と LUMO より上の軌道の数
CORE_GAP_EV = 40.0  # HOMO よりこれだけ低い軌道（内殻）は準位図に出さない。フォームで指定すればボタンから表示できる
AO_CACHE_BYTES = 400e6  # 表示用の格子の AO の値を覚えておく上限
DISPLAY_CUTOFF = 1e-3  # 3D表示に渡す値で、これより小さいものは 0 にする（等値面の選択肢の最小 0.005 より十分小さい）
POSITIVE_COLOR = "#1f4fd8"  # 正の等値面（青）
NEGATIVE_COLOR = "#d62728"  # 負の等値面（赤）

_SPEC = re.compile(r"^(HOMO|LUMO)\s*(?:([+-])\s*(\d+))?$|^(\d+)$")


def parse_orbitals(text):
    """「HOMO, LUMO」「HOMO-1」「LUMO+2」「12」（1始まりの軌道番号）をカンマ区切りで受け取る。
    [(基準, ずれ), ...] を返す。基準は "HOMO" / "LUMO" / "index"。"""
    text = text.replace("＋", "+").replace("－", "-").replace("−", "-")
    tokens = [re.sub(r"\s+", "", t) for t in re.split(r"[,、，]", text) if t.strip()]
    if not tokens:
        raise ColabUMAError("No orbitals were given.", "For example: HOMO, LUMO",
                            ja="表示する軌道が指定されていません。", ja_hint="例：HOMO, LUMO")
    specs = []
    for t in tokens:
        m = _SPEC.match(t.upper())
        if not m:
            raise ColabUMAError(f"Could not read the orbital \"{t}\".",
                                "Write HOMO, LUMO, HOMO-1, LUMO+1 or orbital numbers (from 1), separated by commas.",
                                ja=f"軌道の指定「{t}」が読めません。",
                                ja_hint="HOMO、LUMO、HOMO-1、LUMO+1、または軌道の番号（1始まり）をカンマで区切って書いてください。")
        if m.group(4):
            if int(m.group(4)) < 1:
                raise ColabUMAError(f"The orbital number \"{t}\" must be 1 or more.", "Orbitals are numbered from 1, lowest energy first.",
                                    ja=f"軌道の番号「{t}」は 1 以上にしてください。", ja_hint="番号はエネルギーの低い順に 1 から数えます。")
            spec = ("index", int(m.group(4)) - 1)
        else:
            shift = int(m.group(3) or 0) * (-1 if m.group(2) == "-" else 1)
            if (m.group(1) == "HOMO" and shift > 0) or (m.group(1) == "LUMO" and shift < 0):
                raise ColabUMAError(f"The orbital \"{t}\" cannot be used.",
                                    "Write orbitals above HOMO as LUMO+n, and orbitals below LUMO as HOMO-n.",
                                    ja=f"軌道の指定「{t}」は使えません。",
                                    ja_hint="HOMO より上は LUMO+n、LUMO より下は HOMO-n と書いてください。")
            spec = (m.group(1), shift)
        if spec not in specs:
            specs.append(spec)
    if len(specs) > MAX_ORBITALS:
        raise ColabUMAError(f"At most {MAX_ORBITALS} orbitals can be shown at once.", "Give fewer orbitals.",
                            ja=f"一度に表示できる軌道は {MAX_ORBITALS} 個までです。", ja_hint="軌道の数を減らしてください。")
    return specs


def check_form(orbitals, isovalue, quality):
    specs = parse_orbitals(orbitals)
    if isovalue <= 0 or isovalue > 0.2:
        raise ColabUMAError("The isovalue must be greater than 0 and at most 0.2.", "Common values are 0.02–0.05 (Bohr^-3/2).",
                            ja="等値面の値は 0 より大きく 0.2 以下にしてください。",
                            ja_hint="よく使うのは 0.02〜0.05（Bohr^-3/2）です。")
    grid_quality(quality)
    return specs


def _spin_sets(point):
    """[(スピンの名前, mo_coeff, mo_energy, mo_occ), ...]。閉殻は1つ、開殻は α と β の2つ。"""
    c, e, n = (np.asarray(point[k]) for k in ("mo_coeff", "mo_energy", "mo_occ"))
    if e.ndim == 1:
        return [("", c, e, n)]
    return [("α", c[0], e[0], n[0]), ("β", c[1], e[1], n[1])]


def _label(i, homo):
    if i <= homo:
        return "HOMO" if i == homo else f"HOMO-{homo - i}"
    return "LUMO" if i == homo + 1 else f"LUMO+{i - homo - 1}"


def select(point, specs):
    """指定に合う軌道を選ぶ。[{"spin", "index"(0始まり), "label", "energy", "occ", "coeff"}, ...]。"""
    chosen = []
    for spin, c, e, n in _spin_sets(point):
        occupied = np.nonzero(n > 1e-6)[0]
        homo = int(occupied.max()) if occupied.size else -1
        nmo = len(e)
        for base, shift in specs:
            i = shift if base == "index" else (homo + shift if base == "HOMO" else homo + 1 + shift)
            name = f"#{shift + 1}" if base == "index" else (base if shift == 0 else f"{base}{shift:+d}")
            if not 0 <= i < nmo:
                raise ColabUMAError(f"There is no {spin + ' ' if spin else ''}orbital {name} "
                                    f"(orbitals are 1–{nmo}, HOMO is {homo + 1}).", "Change the orbitals to show.",
                                    ja=f"{spin}軌道「{name}」はありません（軌道は 1〜{nmo} 番、HOMO は {homo + 1} 番です）。",
                                    ja_hint="表示する軌道の指定を変えてください。")
            chosen.append({"spin": spin, "index": i, "label": _label(i, homo), "energy": float(e[i]),
                           "occ": float(n[i]), "coeff": c[:, i]})
    return chosen


def compute(point, atoms, specs, isovalue, quality):
    """point は dft.run(..., keep_orbitals=True) の1構造分の結果。選んだ軌道の格子上の値を計算して dict で返す。"""
    if point.get("mo_coeff") is None:
        raise ColabUMAError("There are no DFT orbitals.", "Run again from the DFT cell.",
                            ja="DFT の軌道がありません。", ja_hint="「DFT」のセルから実行し直してください。")
    quality = grid_quality(quality)
    mol = point["mol"]
    orbitals = select(point, specs)
    grid = make_grid(mol.atom_coords(), GRID_SPACING[quality], MAX_POINTS, MARGIN)
    coords = grid.coords()
    print(f"Grid: {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]} ({grid.size:,} points, spacing {grid.spacing:.3f} Bohr)")
    if grid.spacing > GRID_SPACING[quality]:
        print(f"The molecule is large, so the grid is coarser than the requested {GRID_SPACING[quality]} Bohr.")

    coeff = np.stack([o["coeff"] for o in orbitals], axis=1)
    values = np.empty((len(coords), len(orbitals)))
    progress = ProgressDisplay()
    progress.show([bar_row("Orbitals", 0.0, "0%")], force=True)
    batches = _batches(len(coords), 5000)
    try:
        for k, (p0, p1) in enumerate(batches):
            ao = mol.eval_gto("GTOval_cart" if mol.cart else "GTOval_sph", coords[p0:p1])
            values[p0:p1] = ao @ coeff
            f = (k + 1) / len(batches)
            progress.show([bar_row("Orbitals", f, f"{f * 100:.0f}%")], force=k + 1 == len(batches))
    except MemoryError:
        raise ColabUMAError("The orbital calculation ran out of memory.", "Set the grid quality to coarse.",
                            ja="軌道の計算でメモリが足りません。", ja_hint="格子の細かさを「coarse」にしてください。") from None

    for j, o in enumerate(orbitals):
        o["values"] = values[:, j]
        o.pop("coeff")
    edge = [_name(o) for o in orbitals if _touches_box(o["values"], grid, isovalue)]
    if edge:
        warn(f"The isosurface ({isovalue}) of {', '.join(edge)} reaches the edge of the grid. Part of it may be cut off.",
             "Use a larger isovalue (for example 0.05).")
    ok("Orbital calculation finished.")
    return {"grid": grid, "orbitals": orbitals, "isovalue": isovalue, "point": point,
            "atom_coords": mol.atom_coords(), "numbers": atoms.numbers.copy(), "atoms": atoms}


def _name(o):
    return f"{o['label']} ({o['spin']})" if o["spin"] else o["label"]


def _file_name(o):
    label = o["label"].lower().replace("+", "p").replace("-", "m")
    spin = {"α": "_alpha", "β": "_beta"}.get(o["spin"], "")
    return f"mo{o['index'] + 1}_{label}{spin}.cube"


def _touches_box(field, grid, isovalue):
    r = np.abs(field.reshape(grid.shape))
    faces = (r[0], r[-1], r[:, 0], r[:, -1], r[:, :, 0], r[:, :, -1])
    return any(f.max() > isovalue for f in faces)


def _viewer_url():
    # py3Dmol が使う 3Dmol.js の URL（py3Dmol の版と合わせる）
    import py3Dmol

    m = re.search(r"https://[^'\"]+3Dmol-min\.js", py3Dmol.view().startjs)
    return m.group(0) if m else "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"


class _DisplayGrid:
    """3D表示用の間引いた格子と、クリックした軌道をその格子で計算する関数。"""

    def __init__(self, result):
        self.result = result
        self.mol = result["point"]["mol"]
        _, self.grid = downsample(np.zeros(result["grid"].size), result["grid"], DISPLAY_POINTS)
        self._ao = None

    def _ao_values(self, coords):
        return self.mol.eval_gto("GTOval_cart" if self.mol.cart else "GTOval_sph", coords)

    def values(self, coeff):
        coords = self.grid.coords()
        if len(coords) * self.mol.nao * 8 <= AO_CACHE_BYTES:  # 小さければ AO の値を覚えておいて、2回目から速くする
            if self._ao is None:
                self._ao = self._ao_values(coords)
            return self._ao @ coeff
        return np.concatenate([self._ao_values(coords[p0:p1]) @ coeff for p0, p1 in _batches(len(coords), 5000)])

    def cube(self, values, title):
        # 表示に使わない小さい値は 0 にして短く書き、ページを軽くする
        r = self.result
        values = np.where(np.abs(values) < DISPLAY_CUTOFF, 0.0, values)
        return cube_text(values, self.grid, r["numbers"], r["atom_coords"], title, " {:.3g}")


_explorer = None  # クリックで呼ばれる関数が使う、最後に表示した軌道の格子


def _cube_for_click(spin, index):
    from IPython.display import JSON

    if _explorer is None:
        return JSON({"error": "No orbital results. Run again from the orbital calculation cell."})
    sets = _spin_sets(_explorer.result["point"])
    spin, index = int(spin), int(index)
    if not (0 <= spin < len(sets) and 0 <= index < len(sets[spin][2])):
        return JSON({"error": "That orbital does not exist."})
    try:
        values = _explorer.values(sets[spin][1][:, index])
    except MemoryError:
        return JSON({"error": "Out of memory."})
    return JSON({"cube": _explorer.cube(values, f"orbital {index + 1}")})


def explorer(result):
    """エネルギー準位図と 3D 表示を並べて表示する。準位をクリックすると、その軌道を表示する。

    フォームで指定した軌道はページに埋め込むので、ランタイムがなくても表示できる。ほかの軌道は、クリックしたときに
    Colab のランタイムで計算する（google.colab.kernel.invokeFunction）。
    """
    import json

    from IPython.display import HTML, display

    from .structure import xyz_block

    global _explorer
    _explorer = _DisplayGrid(result)
    try:
        from google.colab import output

        output.register_callback("colabchem.mo_cube", _cube_for_click)
    except ImportError:
        pass

    point = result["point"]
    spins = []
    for spin, _, e, n in _spin_sets(point):
        occupied = np.nonzero(n > 1e-6)[0]
        homo = int(occupied.max()) if occupied.size else -1
        lo, hi = max(0, homo - DIAGRAM_RANGE), min(len(e) - 1, homo + 1 + DIAGRAM_RANGE)
        while lo < homo and e[homo] - e[lo] > CORE_GAP_EV / EV_PER_HARTREE:  # 内殻の軌道は準位図に出さない
            lo += 1
        spins.append({"name": spin, "orbitals": [
            {"i": i, "label": _label(i, homo), "e": float(e[i]) * EV_PER_HARTREE, "occ": float(n[i])}
            for i in range(lo, hi + 1)]})
    names = [s["name"] for s in spins]
    cubes, chips = {}, []
    for o in result["orbitals"]:
        small, _ = downsample(o["values"], result["grid"], DISPLAY_POINTS)
        key = f"{names.index(o['spin'])}:{o['index']}"
        cubes[key] = _explorer.cube(small, _name(o))
        chips.append({"key": key, "name": _name(o), "label": o["label"], "e": o["energy"] * EV_PER_HARTREE, "occ": o["occ"],
                      "i": o["index"]})
    atoms = result["atoms"]
    data = {"uid": f"mo_{uuid.uuid4().hex[:12]}", "spins": spins, "cubes": cubes, "chips": chips,
            "iso": result["isovalue"], "pos": POSITIVE_COLOR, "neg": NEGATIVE_COLOR,
            "xyz": xyz_block(atoms.get_chemical_symbols(), atoms.positions, "")}
    with open(os.path.join(os.path.dirname(__file__), "mo_explorer.html"), encoding="utf-8") as f:
        page = f.read()
    page = page.replace("__URL__", _viewer_url()).replace("__UID__", data["uid"]).replace(
        "__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    display(HTML(page))


def energy_table(point):
    """全軌道のエネルギーの表（pandas.DataFrame）。"""
    import pandas as pd

    rows = []
    for spin, _, e, n in _spin_sets(point):
        occupied = np.nonzero(n > 1e-6)[0]
        homo = int(occupied.max()) if occupied.size else -1
        for i, (ei, ni) in enumerate(zip(e, n)):
            row = {"Spin": spin} if spin else {}
            row.update({"No.": i + 1, "Orbital": _label(i, homo), "Occupation": float(ni),
                        "Energy (Hartree)": round(float(ei), 6), "Energy (eV)": round(float(ei) * EV_PER_HARTREE, 4)})
            rows.append(row)
    return pd.DataFrame(rows)


def gaps(point):
    """[(スピンの名前, HOMO の eV, LUMO の eV, ギャップの eV), ...]。"""
    out = []
    for spin, _, e, n in _spin_sets(point):
        occupied = np.nonzero(n > 1e-6)[0]
        if occupied.size and occupied.max() + 1 < len(e):
            h = int(occupied.max())
            out.append((spin, e[h] * EV_PER_HARTREE, e[h + 1] * EV_PER_HARTREE, (e[h + 1] - e[h]) * EV_PER_HARTREE))
    return out


def report(result):
    """軌道を表示して (files, log) を返す。"""
    from IPython.display import display

    point = result["point"]
    lines = [f"Isovalue: ±{result['isovalue']} Bohr^-3/2"]
    for spin, h, l, g in gaps(point):
        s = f" ({spin})" if spin else ""
        lines.append(f"HOMO{s}: {h:.3f} eV, LUMO{s}: {l:.3f} eV, HOMO-LUMO gap{s}: {g:.3f} eV")
    print("\n".join(lines))
    print("Blue is positive and red is negative. The overall sign of an orbital has no meaning, so the colors can swap.")

    explorer(result)

    table = energy_table(point)
    near = table[table["Orbital"].str.match(r"^(HOMO(-[0-9]+)?|LUMO(\+[0-9]+)?)$")
                 & table["Orbital"].map(lambda s: int(re.sub(r"\D", "", s) or 0) < TABLE_RANGE)]
    print("Orbital energies near HOMO and LUMO (all orbitals are in mo_energies.csv in the ZIP):")
    display(near.style.hide(axis="index"))

    grid = result["grid"]
    log = [f"[Molecular orbitals] grid {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]}, spacing {grid.spacing:.4f} Bohr"] + lines
    log += [f"{_name(o)}: No. {o['index'] + 1}, E = {o['energy']:.6f} Ha ({o['energy'] * EV_PER_HARTREE:.4f} eV), "
            f"occupation {o['occ']:g} → {_file_name(o)}" for o in result["orbitals"]] + [""]
    numbers, coords = result["numbers"], result["atom_coords"]
    files = {"mo_energies.csv": table.to_csv(index=False)}
    for o in result["orbitals"]:
        files[_file_name(o)] = cube_text(o["values"], grid, numbers, coords,
                                         f"{_name(o)} orbital {o['index'] + 1} (Bohr^-3/2)")
    return files, log
