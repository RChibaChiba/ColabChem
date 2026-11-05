"""DFT の一点計算（gpu4pyscf。GPU で使えなければ CPU の PySCF）。

UMA で求めた反応物・TS・生成物の構造で DFT のエネルギーを計算し、活性化エネルギーを比べる。
"""
import time

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


def parse_method(functional, basis):
    key = functional.split("（")[0].strip()
    if key not in FUNCTIONALS:
        raise ColabUMAError(f"汎関数「{functional}」には対応していません。", f"{', '.join(FUNCTIONALS)} から選んでください。")
    if basis not in BASIS_SETS:
        raise ColabUMAError(f"基底関数「{basis}」には対応していません。", f"{', '.join(BASIS_SETS)} から選んでください。")
    return key, basis


def _backend():
    try:
        import cupy  # noqa: F401
        from gpu4pyscf import dft as gdft

        cupy.zeros(1)  # GPU が実際に使えるか
        return "gpu4pyscf", gdft
    except Exception:
        from pyscf import dft as cdft

        return "pyscf (CPU)", cdft


def single_point(atoms, charge, multiplicity, functional, basis, max_cycle=100):
    """1構造の DFT エネルギー（Hartree）。dict を返す。"""
    import pyscf

    name, mod = _backend()
    xc, disp = FUNCTIONALS[functional]
    mol = pyscf.M(atom=[(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                  basis=basis, ecp=basis if atoms.numbers.max() > 36 else None,  # def2 の ECP は Rb 以降
                  charge=int(charge), spin=int(multiplicity) - 1, unit="Angstrom", verbose=0)
    mf = (mod.RKS(mol, xc=xc) if multiplicity == 1 else mod.UKS(mol, xc=xc)).density_fit()
    if disp:
        mf.disp = disp
    mf.max_cycle = max_cycle
    t0 = time.time()
    e = float(mf.kernel())
    out = {"energy": e, "converged": bool(mf.converged), "seconds": time.time() - t0, "backend": name,
           "nao": int(mol.nao)}
    if name.startswith("gpu"):
        import cupy

        del mf
        cupy.get_default_memory_pool().free_all_blocks()
    return out


def run(points, charge, multiplicity, functional, basis):
    """points は [(名前, atoms), ...]。各構造の一点計算の結果を返す。"""
    functional, basis = parse_method(functional, basis)
    name, _ = _backend()
    if not name.startswith("gpu"):
        warn("gpu4pyscf が使えないため、CPU の PySCF で計算します。かなり時間がかかります。",
             "ランタイムのタイプで T4 GPU を選んでいるか確認してください。")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # UMA が使った GPU メモリを空ける
    except ImportError:
        pass
    print(f"DFT 一点計算: {functional}/{basis}（{name}、密度フィッティング）")
    progress = ProgressDisplay()
    results = []
    for k, (label, atoms) in enumerate(points):
        progress.show([bar_row("DFT", k / len(points), f"{label} を計算中（{k}/{len(points)} 完了）")], force=True)
        try:
            r = single_point(atoms, charge, multiplicity, functional, basis)
        except MemoryError:
            raise ColabUMAError("DFT の計算でメモリが足りません。", "基底関数を def2-SVP にしてください。") from None
        except Exception as e:
            if "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower():
                raise ColabUMAError("DFT の計算で GPU のメモリが足りません。", "基底関数を def2-SVP にしてください。") from None
            raise
        r.update(label=label, atoms=atoms)
        results.append(r)
        print(f"  {label}: E = {r['energy']:.8f} Ha  {'収束' if r['converged'] else 'SCF未収束'}"
              f"（基底関数 {r['nao']} 個、{r['seconds']:.1f} 秒）")
        if not r["converged"]:
            warn(f"{label}の SCF が収束しませんでした。このエネルギーは信頼できません。")
    progress.show([bar_row("DFT", 1.0, f"{len(points)}/{len(points)} 完了")], force=True)
    ok("DFT の一点計算が終わりました。")
    return {"functional": functional, "basis": basis, "backend": name, "points": results}


def report(dft_res, uma_rows):
    """UMA と DFT のエネルギー差を並べて表示する。(files, log) を返す。"""
    import pandas as pd
    from IPython.display import display

    pts = dft_res["points"]
    uma = dict(uma_rows) if uma_rows else {}
    base = pts[0]["energy"]
    ubase = uma.get(pts[0]["label"])
    rows = []
    for p in pts:
        row = {"": p["label"], "DFT (Hartree)": round(p["energy"], 8),
               f"ΔE DFT（{pts[0]['label']}基準, kcal/mol）": round((p["energy"] - base) * KCALMOL_PER_HARTREE, 2)}
        if p["label"] in uma and ubase is not None:
            row[f"ΔE UMA（kcal/mol）"] = round((uma[p["label"]] - ubase) / EV_PER_HARTREE * KCALMOL_PER_HARTREE, 2)
        row["SCF収束"] = "はい" if p["converged"] else "いいえ"
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"DFT 一点計算（{dft_res['functional']}/{dft_res['basis']}、UMA の構造）：")
    display(df.style.hide(axis="index"))
    log = [f"[DFT一点計算] {dft_res['functional']}/{dft_res['basis']}（{dft_res['backend']}、密度フィッティング、UMAの構造）"]
    log += [f"{p['label']}: E = {p['energy']:.10f} Ha  converged = {p['converged']}  nao = {p['nao']}  "
            f"{p['seconds']:.1f} s" for p in pts]
    if len(pts) == 3 and pts[1]["label"] == "TS":
        e_r, e_ts, e_p = (p["energy"] for p in pts)
        fwd = (e_ts - e_r) * KCALMOL_PER_HARTREE
        rev = (e_ts - e_p) * KCALMOL_PER_HARTREE
        rxn = (e_p - e_r) * KCALMOL_PER_HARTREE
        print(f"活性化エネルギー（DFT）: 正反応 {fwd:.2f} kcal/mol、逆反応 {rev:.2f} kcal/mol、反応エネルギー {rxn:.2f} kcal/mol")
        log += [f"活性化エネルギー 正反応: {fwd:.4f} kcal/mol", f"活性化エネルギー 逆反応: {rev:.4f} kcal/mol",
                f"反応エネルギー: {rxn:.4f} kcal/mol"]
    log.append("")
    files = {"dft.csv": df.to_csv(index=False)}
    return files, log
