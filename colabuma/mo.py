"""分子軌道（HOMO・LUMO など）：DFT の軌道を格子上で計算し、正と負の等値面を 3D 表示する。

軌道の値の単位は Bohr^(-3/2)。軌道の符号（位相）には意味がなく、計算ごとに全体が反転することがある。
開殻（UKS）では α と β の軌道を別々に扱う。
"""
import re

import numpy as np

from .esp import GRID_SPACING, MARGIN, _batches, cube_text, downsample, make_grid
from .ui import ColabUMAError, ProgressDisplay, bar_row, ok, warn

EV_PER_HARTREE = 27.211386245988

MAX_POINTS = 500_000  # 格子の点の数の上限。超えるなら格子を粗くする
DISPLAY_POINTS = 80_000  # 3D表示に渡す1軌道あたりの点の数。多いとノートの出力が重くなる
MAX_ORBITALS = 8  # 一度に表示する軌道の数の上限（開殻では α と β で倍になる）
TABLE_RANGE = 5  # エネルギーの表に出す、HOMO より下と LUMO より上の軌道の数
POSITIVE_COLOR = "#1f4fd8"  # 正の等値面（青）
NEGATIVE_COLOR = "#d62728"  # 負の等値面（赤）

_SPEC = re.compile(r"^(HOMO|LUMO)\s*(?:([+-])\s*(\d+))?$|^(\d+)$")


def parse_orbitals(text):
    """「HOMO, LUMO」「HOMO-1」「LUMO+2」「12」（1始まりの軌道番号）をカンマ区切りで受け取る。
    [(基準, ずれ), ...] を返す。基準は "HOMO" / "LUMO" / "index"。"""
    text = text.replace("＋", "+").replace("－", "-").replace("−", "-")
    tokens = [re.sub(r"\s+", "", t) for t in re.split(r"[,、，]", text) if t.strip()]
    if not tokens:
        raise ColabUMAError("表示する軌道が指定されていません。", "例：HOMO, LUMO")
    specs = []
    for t in tokens:
        m = _SPEC.match(t.upper())
        if not m:
            raise ColabUMAError(f"軌道の指定「{t}」が読めません。",
                                "HOMO、LUMO、HOMO-1、LUMO+1、または軌道の番号（1始まり）をカンマで区切って書いてください。")
        if m.group(4):
            if int(m.group(4)) < 1:
                raise ColabUMAError(f"軌道の番号「{t}」は 1 以上にしてください。", "番号はエネルギーの低い順に 1 から数えます。")
            spec = ("index", int(m.group(4)) - 1)
        else:
            shift = int(m.group(3) or 0) * (-1 if m.group(2) == "-" else 1)
            if (m.group(1) == "HOMO" and shift > 0) or (m.group(1) == "LUMO" and shift < 0):
                raise ColabUMAError(f"軌道の指定「{t}」は使えません。",
                                    "HOMO より上は LUMO+n、LUMO より下は HOMO-n と書いてください。")
            spec = (m.group(1), shift)
        if spec not in specs:
            specs.append(spec)
    if len(specs) > MAX_ORBITALS:
        raise ColabUMAError(f"一度に表示できる軌道は {MAX_ORBITALS} 個までです。", "軌道の数を減らしてください。")
    return specs


def check_form(orbitals, isovalue, quality):
    specs = parse_orbitals(orbitals)
    if isovalue <= 0 or isovalue > 0.2:
        raise ColabUMAError("等値面の値は 0 より大きく 0.2 以下にしてください。",
                            "よく使うのは 0.02〜0.05（Bohr^-3/2）です。")
    if quality not in GRID_SPACING:
        raise ColabUMAError(f"格子の細かさ「{quality}」には対応していません。", f"{', '.join(GRID_SPACING)} から選んでください。")
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
            name = f"{shift + 1}番" if base == "index" else (base if shift == 0 else f"{base}{shift:+d}")
            if not 0 <= i < nmo:
                raise ColabUMAError(f"{spin}軌道「{name}」はありません（軌道は 1〜{nmo} 番、HOMO は {homo + 1} 番です）。",
                                    "表示する軌道の指定を変えてください。")
            chosen.append({"spin": spin, "index": i, "label": _label(i, homo), "energy": float(e[i]),
                           "occ": float(n[i]), "coeff": c[:, i]})
    return chosen


def compute(point, atoms, specs, isovalue, quality):
    """point は dft.run(..., keep_orbitals=True) の1構造分の結果。選んだ軌道の格子上の値を計算して dict で返す。"""
    if point.get("mo_coeff") is None:
        raise ColabUMAError("DFT の軌道がありません。", "「DFT 一点計算」のセルから実行し直してください。")
    mol = point["mol"]
    orbitals = select(point, specs)
    grid = make_grid(mol.atom_coords(), GRID_SPACING[quality], MAX_POINTS, MARGIN)
    coords = grid.coords()
    print(f"格子：{grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]}（{grid.size:,} 点、間隔 {grid.spacing:.3f} Bohr）")
    if grid.spacing > GRID_SPACING[quality]:
        print(f"分子が大きいため、格子の間隔を指定の {GRID_SPACING[quality]} Bohr より粗くしました。")

    coeff = np.stack([o["coeff"] for o in orbitals], axis=1)
    values = np.empty((len(coords), len(orbitals)))
    progress = ProgressDisplay()
    progress.show([bar_row("軌道", 0.0, "0%")], force=True)
    batches = _batches(len(coords), 5000)
    try:
        for k, (p0, p1) in enumerate(batches):
            ao = mol.eval_gto("GTOval_cart" if mol.cart else "GTOval_sph", coords[p0:p1])
            values[p0:p1] = ao @ coeff
            f = (k + 1) / len(batches)
            progress.show([bar_row("軌道", f, f"{f * 100:.0f}%")], force=k + 1 == len(batches))
    except MemoryError:
        raise ColabUMAError("軌道の計算でメモリが足りません。", "格子の細かさを「粗い」にしてください。") from None

    for j, o in enumerate(orbitals):
        o["values"] = values[:, j]
        o.pop("coeff")
    edge = [_name(o) for o in orbitals if _touches_box(o["values"], grid, isovalue)]
    if edge:
        warn(f"{', '.join(edge)} の等値面（{isovalue}）が計算範囲の端に届いています。面の一部が切れている可能性があります。",
             "等値面の値を大きくしてください（例：0.05）。")
    ok("軌道の計算が終わりました。")
    return {"grid": grid, "orbitals": orbitals, "isovalue": isovalue, "point": point,
            "atom_coords": mol.atom_coords(), "numbers": atoms.numbers.copy(), "atoms": atoms}


def _name(o):
    return f"{o['label']}（{o['spin']}）" if o["spin"] else o["label"]


def _file_name(o):
    label = o["label"].lower().replace("+", "p").replace("-", "m")
    spin = {"α": "_alpha", "β": "_beta"}.get(o["spin"], "")
    return f"mo{o['index'] + 1}_{label}{spin}.cube"


def _touches_box(field, grid, isovalue):
    r = np.abs(field.reshape(grid.shape))
    faces = (r[0], r[-1], r[:, 0], r[:, -1], r[:, :, 0], r[:, :, -1])
    return any(f.max() > isovalue for f in faces)


def show_orbital(result, orbital):
    """1つの軌道の正（青）と負（赤）の等値面を表示する。"""
    import py3Dmol

    from .structure import xyz_block
    from .ui import heading

    small_field, small = downsample(orbital["values"], result["grid"], DISPLAY_POINTS)
    atoms = result["atoms"]
    cube = cube_text(small_field, small, result["numbers"], result["atom_coords"], _name(orbital), " {:.2E}")
    heading(f"{_name(orbital)}　{orbital['index'] + 1} 番、{orbital['energy'] * EV_PER_HARTREE:.3f} eV、"
            f"占有数 {orbital['occ']:g}")
    view = py3Dmol.view(width=640, height=420)
    view.addModel(xyz_block(atoms.get_chemical_symbols(), atoms.positions, ""), "xyz")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.2}})
    iso = result["isovalue"]
    view.addVolumetricData(cube, "cube", {"isoval": iso, "color": POSITIVE_COLOR, "opacity": 0.85, "smoothness": 3})
    view.addVolumetricData(cube, "cube", {"isoval": -iso, "color": NEGATIVE_COLOR, "opacity": 0.85, "smoothness": 3})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.show()


def energy_table(point):
    """全軌道のエネルギーの表（pandas.DataFrame）。"""
    import pandas as pd

    rows = []
    for spin, _, e, n in _spin_sets(point):
        occupied = np.nonzero(n > 1e-6)[0]
        homo = int(occupied.max()) if occupied.size else -1
        for i, (ei, ni) in enumerate(zip(e, n)):
            row = {"スピン": spin} if spin else {}
            row.update({"番号": i + 1, "軌道": _label(i, homo), "占有数": float(ni),
                        "エネルギー (Hartree)": round(float(ei), 6), "エネルギー (eV)": round(float(ei) * EV_PER_HARTREE, 4)})
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
    lines = [f"等値面: ±{result['isovalue']} Bohr^-3/2"]
    for spin, h, l, g in gaps(point):
        s = f"（{spin}）" if spin else ""
        lines.append(f"HOMO{s}: {h:.3f} eV、LUMO{s}: {l:.3f} eV、HOMO-LUMO ギャップ{s}: {g:.3f} eV")
    print("\n".join(lines))
    print("青が正、赤が負の等値面です。軌道の符号（位相）には意味がないので、青と赤が入れ替わることがあります。")

    for o in result["orbitals"]:
        show_orbital(result, o)

    table = energy_table(point)
    near = table[table["軌道"].str.match(r"^(HOMO(-[0-9]+)?|LUMO(\+[0-9]+)?)$")
                 & table["軌道"].map(lambda s: int(re.sub(r"\D", "", s) or 0) < TABLE_RANGE)]
    print("HOMO・LUMO の近くの軌道のエネルギー（全軌道は ZIP の mo_energies.csv）：")
    display(near.style.hide(axis="index"))

    grid = result["grid"]
    log = [f"[分子軌道] 格子 {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]}、間隔 {grid.spacing:.4f} Bohr"] + lines
    log += [f"{_name(o)}: {o['index'] + 1} 番、E = {o['energy']:.6f} Ha（{o['energy'] * EV_PER_HARTREE:.4f} eV）、"
            f"占有数 {o['occ']:g} → {_file_name(o)}" for o in result["orbitals"]] + [""]
    numbers, coords = result["numbers"], result["atom_coords"]
    files = {"mo_energies.csv": table.to_csv(index=False)}
    for o in result["orbitals"]:
        files[_file_name(o)] = cube_text(o["values"], grid, numbers, coords,
                                         f"{_name(o)} orbital {o['index'] + 1} (Bohr^-3/2)")
    return files, log
