"""静電ポテンシャル（ESP）：DFT の密度から格子上の電子密度と ESP を計算し、等密度面に色を付けて表示する。

ESP は +1 の点電荷が感じるポテンシャル V(r) = Σ Z_A/|r-R_A| - ∫ρ(r')/|r-r'| dr'。
ECP を使う元素では、Z_A は有効核電荷、ρ は価電子の密度になる（内殻の分が両方から抜けるので打ち消し合う）。
"""
from dataclasses import dataclass

import numpy as np

from .ui import ColabUMAError, ProgressDisplay, bar_row, ok, warn

BOHR_PER_ANGSTROM = 1.8897261246
KCALMOL_PER_AU = 627.5094740631

GRID_SPACING = {"粗い": 0.5, "標準": 0.35, "細かい": 0.25}  # 格子の間隔（Bohr）
MARGIN = 7.0  # 分子のまわりの余白（Bohr）。0.001 の等密度面まで入る広さ
MAX_POINTS = 1_000_000  # GPU のとき。超えるなら格子を粗くする
MAX_POINTS_CPU = 150_000  # GPU が使えないとき
DISPLAY_POINTS = 200_000  # 3D表示に渡す点の数。多いとノートの出力が重くなる
SHELL_RATIO = 1.25  # 等密度面の上の値を調べる範囲（密度が等値の 1/1.25〜1.25 倍の格子点）
GPU_CHECK_TOLERANCE = 1e-5  # GPU と CPU の結果の許容差（a.u.）


@dataclass
class Grid:
    origin: np.ndarray  # Bohr
    spacing: float  # Bohr
    shape: tuple

    @property
    def size(self):
        return int(np.prod(self.shape))

    def coords(self):
        # cube の順番（x が一番外側、z が一番内側）
        axes = [self.origin[i] + np.arange(self.shape[i]) * self.spacing for i in range(3)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def check_form(isovalue, esp_range, quality):
    if isovalue <= 0 or isovalue > 0.1:
        raise ColabUMAError("等密度面の値は 0 より大きく 0.1 以下にしてください。",
                            "よく使うのは 0.001 または 0.002（e/Bohr³）です。")
    if esp_range < 0:
        raise ColabUMAError("ESP の色の範囲は 0 以上にしてください。", "0 にすると自動で決めます。")
    if quality not in GRID_SPACING:
        raise ColabUMAError(f"格子の細かさ「{quality}」には対応していません。", f"{', '.join(GRID_SPACING)} から選んでください。")


def make_grid(atom_coords, spacing, max_points=MAX_POINTS, margin=MARGIN):
    lo = atom_coords.min(axis=0) - margin
    extent = atom_coords.max(axis=0) + margin - lo
    spacing = max(spacing, float(np.prod(extent) / max_points) ** (1 / 3))
    shape = tuple(int(np.ceil(e / spacing)) + 1 for e in extent)
    return Grid(lo, spacing, shape)


def _batches(n, size):
    return [(p, min(p + size, n)) for p in range(0, n, size)]


def electron_density(mol, dm, coords, on_progress=None):
    """全電子密度（e/Bohr³）。"""
    rho = np.empty(len(coords))
    batches = _batches(len(coords), 5000)
    for k, (p0, p1) in enumerate(batches):
        ao = mol.eval_gto("GTOval_cart" if mol.cart else "GTOval_sph", coords[p0:p1])
        rho[p0:p1] = np.einsum("pi,pi->p", ao @ dm, ao)
        if on_progress:
            on_progress((k + 1) / len(batches))
    return rho


def nuclear_potential(mol, coords):
    v = np.zeros(len(coords))
    for z, r in zip(mol.atom_charges(), mol.atom_coords()):
        d = np.linalg.norm(coords - r, axis=1)
        v += z / np.maximum(d, 1e-8)  # 格子点が原子核に重なっても無限大にしない
    return v


def _electron_potential_cpu(mol, dm, coords, on_progress=None):
    from pyscf import df, gto

    v = np.empty(len(coords))
    batches = _batches(len(coords), 600)
    for k, (p0, p1) in enumerate(batches):
        ints = df.incore.aux_e2(mol, gto.fakemol_for_charges(coords[p0:p1]))
        v[p0:p1] = np.einsum("ijp,ij->p", ints, dm)
        if on_progress:
            on_progress((k + 1) / len(batches))
    return v


def _electron_potential_gpu(mol, dm, coords, on_progress=None):
    import cupy
    from gpu4pyscf.gto.int3c1e import VHFOpt, int1e_grids

    intopt = VHFOpt(mol)
    intopt.build(1e-13, aosym=True)
    dm_gpu = cupy.asarray(dm)
    v = np.empty(len(coords))
    batches = _batches(len(coords), 50000)
    for k, (p0, p1) in enumerate(batches):
        v[p0:p1] = cupy.asnumpy(int1e_grids(mol, coords[p0:p1], dm=dm_gpu, intopt=intopt))
        if on_progress:
            on_progress((k + 1) / len(batches))
    return v


def electron_potential(mol, dm, coords, use_gpu, on_progress=None):
    """電子の作るポテンシャル ∫ρ(r')/|r-r'| dr'（正の値）と、使った計算の名前を返す。

    GPU の結果は、最初に数点だけ CPU の結果と比べてから使う。合わないか失敗したら CPU で計算する。
    """
    if use_gpu:
        try:
            probe = coords[:: max(1, len(coords) // 16)]
            diff = np.abs(_electron_potential_gpu(mol, dm, probe) - _electron_potential_cpu(mol, dm, probe)).max()
            if diff > GPU_CHECK_TOLERANCE:
                raise ValueError(f"CPU の結果と {diff:.1e} a.u. 違います")
            return _electron_potential_gpu(mol, dm, coords, on_progress), "gpu4pyscf"
        except Exception as e:
            warn(f"GPU での ESP の計算に失敗したため、CPU で計算し直します（{type(e).__name__}: {e}）。",
                 "CPU ではかなり時間がかかります。格子の細かさを「粗い」にすると速くなります。")
    return _electron_potential_cpu(mol, dm, coords, on_progress), "PySCF (CPU)"


def compute(point, atoms, isovalue, quality):
    """point は dft.run(..., keep_density=True) の1構造分の結果。格子上の密度と ESP を計算して dict で返す。"""
    if point.get("dm") is None:
        raise ColabUMAError("DFT の密度がありません。", "「DFT 一点計算」のセルから実行し直してください。")
    mol, dm = point["mol"], point["dm"]
    use_gpu = point["backend"].startswith("gpu")
    grid = make_grid(mol.atom_coords(), GRID_SPACING[quality], MAX_POINTS if use_gpu else MAX_POINTS_CPU)
    coords = grid.coords()
    print(f"格子：{grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]}（{grid.size:,} 点、間隔 {grid.spacing:.3f} Bohr）")
    if grid.spacing > GRID_SPACING[quality]:
        print(f"分子が大きいため、格子の間隔を指定の {GRID_SPACING[quality]} Bohr より粗くしました。")

    progress = ProgressDisplay()
    rows = lambda rho_f, esp_f: [bar_row("密度", rho_f, f"{rho_f * 100:.0f}%"), bar_row("ESP", esp_f, f"{esp_f * 100:.0f}%")]
    progress.show(rows(0.0, 0.0), force=True)
    try:
        rho = electron_density(mol, dm, coords, lambda f: progress.show(rows(f, 0.0)))
        progress.show(rows(1.0, 0.0), force=True)
        v_el, backend = electron_potential(mol, dm, coords, use_gpu, lambda f: progress.show(rows(1.0, f)))
    except MemoryError:
        raise ColabUMAError("ESP の計算でメモリが足りません。", "格子の細かさを「粗い」にしてください。") from None
    esp = nuclear_potential(mol, coords) - v_el
    progress.show(rows(1.0, 1.0), force=True)

    if use_gpu:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    result = {"grid": grid, "density": rho, "esp": esp, "isovalue": isovalue, "backend": backend,
              "atom_coords": mol.atom_coords(), "numbers": atoms.numbers.copy(), "atoms": atoms}
    result.update(surface_stats(rho, esp, coords, mol.atom_coords(), isovalue))
    if _touches_box(rho, grid, isovalue):
        warn(f"等密度面（{isovalue}）が計算範囲の端に届いています。面の一部が切れている可能性があります。",
             "等密度面の値を大きくしてください（例：0.002）。")
    ok("ESP の計算が終わりました。")
    return result


def surface_stats(rho, esp, coords, atom_coords, isovalue):
    # 等密度面の近くの格子点での ESP の最小・最大と、それに一番近い原子（1始まり）
    mask = (rho > isovalue / SHELL_RATIO) & (rho < isovalue * SHELL_RATIO)
    if not mask.any():
        raise ColabUMAError(f"密度 {isovalue} の等密度面が格子上に見つかりません。",
                            "等密度面の値を 0.001〜0.002 にするか、格子を細かくしてください。")
    values, points = esp[mask], coords[mask]
    i_min, i_max = int(values.argmin()), int(values.argmax())
    nearest = lambda p: int(np.linalg.norm(atom_coords - p, axis=1).argmin()) + 1
    return {"v_min": float(values[i_min]), "v_max": float(values[i_max]),
            "atom_min": nearest(points[i_min]), "atom_max": nearest(points[i_max]),
            "auto_range": float(max(abs(np.percentile(values, 1)), abs(np.percentile(values, 99)))) or 0.05}


def _touches_box(rho, grid, isovalue):
    r = rho.reshape(grid.shape)
    faces = (r[0], r[-1], r[:, 0], r[:, -1], r[:, :, 0], r[:, :, -1])
    return any(f.max() > isovalue for f in faces)


def cube_text(field, grid, numbers, atom_coords, comment, fmt="{:13.5E}"):
    """Gaussian の cube 形式（単位 Bohr）。"""
    nx, ny, nz = grid.shape
    d = grid.spacing
    lines = [comment, "OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z",
             f"{len(numbers):5d}{grid.origin[0]:12.6f}{grid.origin[1]:12.6f}{grid.origin[2]:12.6f}",
             f"{nx:5d}{d:12.6f}{0.0:12.6f}{0.0:12.6f}",
             f"{ny:5d}{0.0:12.6f}{d:12.6f}{0.0:12.6f}",
             f"{nz:5d}{0.0:12.6f}{0.0:12.6f}{d:12.6f}"]
    lines += [f"{int(z):5d}{float(z):12.6f}{x:12.6f}{y:12.6f}{w:12.6f}" for z, (x, y, w) in zip(numbers, atom_coords)]
    values = np.asarray(field).ravel()
    for start in range(0, values.size, nz):  # z の1列ごとに改行する
        column = values[start:start + nz]
        lines += ["".join(fmt.format(v) for v in column[i:i + 6]) for i in range(0, nz, 6)]
    return "\n".join(lines) + "\n"


def downsample(field, grid, max_points=DISPLAY_POINTS):
    """3D表示用に間引く。(間引いた配列, 間引いた格子) を返す。"""
    step = max(1, int(np.ceil((grid.size / max_points) ** (1 / 3))))
    if step == 1:
        return field.reshape(grid.shape), grid
    small = field.reshape(grid.shape)[::step, ::step, ::step]
    return small, Grid(grid.origin, grid.spacing * step, small.shape)


def show(result, color_range):
    """等密度面に ESP の色を付けて表示する。color_range は色の範囲（a.u.）で、負が赤、正が青。"""
    import py3Dmol
    from IPython.display import HTML, display

    from .structure import xyz_block

    rho_small, small = downsample(result["density"], result["grid"])
    esp_small, _ = downsample(result["esp"], result["grid"])
    numbers, coords = result["numbers"], result["atom_coords"]
    atoms = result["atoms"]
    view = py3Dmol.view(width=640, height=480)
    view.addModel(xyz_block(atoms.get_chemical_symbols(), atoms.positions, ""), "xyz")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.2}})
    view.addVolumetricData(cube_text(rho_small, small, numbers, coords, "density", " {:.3E}"), "cube", {
        "isoval": result["isovalue"], "opacity": 0.85, "smoothness": 3,
        "voldata": cube_text(esp_small, small, numbers, coords, "esp", " {:.3E}"), "volformat": "cube",
        "volscheme": {"gradient": "rwb", "min": -color_range, "max": color_range},
    })
    view.setBackgroundColor("white")
    view.zoomTo()
    view.show()
    r = color_range * KCALMOL_PER_AU
    display(HTML(
        "<div style='width:640px;max-width:100%;font-size:13px;margin-top:6px'>"
        "<div style='height:14px;border-radius:3px;background:linear-gradient(to right,#ff0000,#ffffff,#0000ff)'></div>"
        f"<div style='display:flex;justify-content:space-between'><span>{-r:.1f}（負：赤）</span><span>0</span>"
        f"<span>+{r:.1f}（正：青）kcal/mol</span></div></div>"))


def report(result, esp_range_kcal=0.0):
    """表示して (files, log) を返す。esp_range_kcal が 0 なら色の範囲は自動。"""
    color_range = esp_range_kcal / KCALMOL_PER_AU if esp_range_kcal > 0 else result["auto_range"]
    show(result, color_range)
    v_min, v_max = result["v_min"], result["v_max"]
    lines = [f"等密度面: {result['isovalue']} e/Bohr³",
             f"面の上の最小 ESP: {v_min * KCALMOL_PER_AU:.2f} kcal/mol（{v_min:.5f} a.u.、原子 {result['atom_min']} の近く）",
             f"面の上の最大 ESP: {v_max * KCALMOL_PER_AU:.2f} kcal/mol（{v_max:.5f} a.u.、原子 {result['atom_max']} の近く）",
             f"色の範囲: ±{color_range * KCALMOL_PER_AU:.2f} kcal/mol"]
    print("\n".join(lines))
    print("最小・最大は、面の近くの格子点の値です（格子の細かさで少し変わります）。")
    grid = result["grid"]
    log = [f"[ESP] {result['backend']}、格子 {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]}、間隔 {grid.spacing:.4f} Bohr"] + lines + [""]
    numbers, coords = result["numbers"], result["atom_coords"]
    files = {"density.cube": cube_text(result["density"], grid, numbers, coords, "Electron density (e/Bohr^3)"),
             "esp.cube": cube_text(result["esp"], grid, numbers, coords, "Electrostatic potential (a.u.)")}
    return files, log
