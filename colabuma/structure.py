"""ファイルのアップロード、構造を描く入力（JSME）、xyz・gjf の読み込み、元素と電荷・多重度の確認、3D表示。"""
import os
import re
import tempfile

import numpy as np

from .ui import ColabUMAError, warn

GJF_EXTENSIONS = (".gjf", ".com", ".gjc")
INPUT_MODES = ("ファイル", "描く")
DRAWN_NAME = "drawn.mol"  # 描いた構造は、この名前の molfile として読み込みに渡す
JSME_URL = "https://cdn.jsdelivr.net/npm/jsme-editor@2024.4.29/jsme.nocache.js"
N_CONFORMERS = 10  # 描いた構造から作る配座の数。MMFF のエネルギーが一番低いものを使う
MAX_Z = 83  # UMAのomolタスク（OMol25）が扱える元素は H〜Bi

_uploads = {}  # 前回アップロードしたファイル（Cancel upload で使い回す）。キーごとに覚える


# --- アップロード ---
def upload(key="input", what="xyz または gjf（.gjf / .com）ファイル"):
    """ファイルを1つ選ばせて (名前, 中身) を返す。Cancel upload なら前回のファイルを使う。"""
    from google.colab import files

    previous = _uploads.get(key)
    print(f"{what}を選んでください。")
    if previous:
        print(f"前回の「{previous[0]}」をそのまま使うときは Cancel upload を押してください。")
    uploaded = files.upload()
    if uploaded:
        if len(uploaded) > 1:
            warn("複数のファイルが選ばれたため、最初の1つだけを使います。")
        _uploads[key] = next(iter(uploaded.items()))
    elif previous:
        print(f"前回のファイル「{previous[0]}」を使います。")
    else:
        raise ColabUMAError("ファイルがアップロードされませんでした。", f"もう一度すべて実行して、{what}を選んでください。")
    return _uploads[key]


def get_input(mode):
    """フォームの「入力」に合わせて、ファイルを選ばせるか構造を描かせる。(名前, 中身) を返す。"""
    if mode == "ファイル":
        return upload()
    if mode == "描く":
        return draw()
    raise ColabUMAError(f"入力「{mode}」には対応していません。", f"{', '.join(INPUT_MODES)} から選んでください。")


def draw(key="draw"):
    """JSME で構造を描かせ、「この構造で計算」が押されるまで待つ。(名前, molfile) を返す。前回の構造を最初に表示する。"""
    import json

    from IPython.display import HTML, display

    try:
        from google.colab import output
    except ImportError:
        raise ColabUMAError("構造を描く入力は Google Colab でだけ使えます。", "入力を「ファイル」にしてください。") from None
    previous = _uploads.get(key)
    with open(os.path.join(os.path.dirname(__file__), "draw.html"), encoding="utf-8") as f:
        page = f.read()
    page = page.replace("__JSME_URL__", JSME_URL).replace(
        "__PREVIOUS__", json.dumps(previous[1].decode() if previous else "").replace("</", "<\\/"))
    print("構造を描いて「この構造で計算」を押してください。水素は描かなくても自動で付けます。")
    display(HTML(page))
    molfile = output.eval_js("new Promise((resolve) => { const wait = () => window.colabchemDrawn"
                             " ? window.colabchemDrawn.then(resolve) : setTimeout(wait, 100); wait(); })")
    if not molfile or not molfile.strip():
        raise ColabUMAError("構造が描かれていません。", "もう一度すべて実行して、構造を描いてください。")
    _uploads[key] = (DRAWN_NAME, molfile.encode())
    return _uploads[key]


def molfile_to_atoms(molfile):
    """描いた構造（2D の molfile）に水素を付けて 3D にする。(atoms, SMILES, 形式電荷の合計) を返す。

    RDKit の ETKDG で配座をいくつか作り、MMFF94（使えなければ UFF）で最適化して、エネルギーが一番低いものを使う。
    """
    from ase import Atoms
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromMolBlock(molfile, removeHs=False)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ColabUMAError("描いた構造を読み取れませんでした。", "原子の価数（結合の数）がおかしくないか確認して、描き直してください。")
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 20240429
    ids = list(AllChem.EmbedMultipleConfs(mol, N_CONFORMERS, params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, N_CONFORMERS, params))
    if not ids:
        raise ColabUMAError("描いた構造から 3D の座標を作れませんでした。", "ひずみの大きい構造かもしれません。xyz か gjf で入力してください。")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)]
    else:
        energies = [e for _, e in AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)]
    best = ids[int(np.argmin(energies))]
    atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer(best).GetPositions())
    return atoms, smiles, Chem.GetFormalCharge(mol)


def quick_atom_count(name, data):
    # インストール前に原子番号の範囲だけ調べる。読めなければ None（あとで詳しく調べる）
    text = data.decode("utf-8", "replace").strip()
    if os.path.splitext(name)[1].lower() in GJF_EXTENSIONS:
        blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
        return len(blocks[2].splitlines()) - 1 if len(blocks) >= 3 else None
    try:
        return int(text.split()[0])
    except (IndexError, ValueError):
        return None


# --- 読み込み ---
def check_elements(numbers):
    from ase.data import chemical_symbols

    bad = sorted({chemical_symbols[z] if 0 < z < len(chemical_symbols) else "X"
                  for z in numbers if z < 1 or z > MAX_Z})
    if bad:
        raise ColabUMAError(f"UMAが扱えない元素が含まれています：{', '.join(bad)}",
                            "UMA（omol）が扱えるのは H〜Bi（原子番号1〜83）です。該当する元素を除くか、別の手法を使ってください。")


_FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _gjf_cartesian_line(line):
    # 「元素 [凍結フラグ] x y z [ONIOMの層など]」を (元素記号, xyz) にする。読めなければ None
    from ase.data import chemical_symbols

    tok = line.replace(",", " ").split()
    m = re.match(r"[A-Za-z]+|\d+", tok[0])  # "C(Fragment=1)", "C-CA--0.1", "C12" → "C"、"6" → 原子番号
    if not m:
        return None
    label = m.group(0)
    if label.isdigit():
        z = int(label)
        sym = chemical_symbols[z] if 0 < z < len(chemical_symbols) else label
    elif label.lower().startswith("bq") or label.upper() == "X":
        sym = "X"  # ゴースト原子・ダミー原子（後でエラーにする）
    elif label.lower().startswith("tv"):
        sym = "Tv"
    else:
        sym = label[:2].capitalize() if label[:2].capitalize() in chemical_symbols else label[:1].upper()
    nums = tok[1:]
    is_f = [bool(_FLOAT.match(t)) for t in nums]
    if len(nums) >= 4 and re.fullmatch(r"-?\d+", nums[0]) and all(is_f[1:4]) and (len(nums) == 4 or not is_f[4]):
        xyz = nums[1:4]
    elif len(nums) >= 3 and all(is_f[:3]) and (len(nums) == 3 or not is_f[3]):
        xyz = nums[:3]
    else:
        return None
    return tok[0], sym, [float(t.replace("D", "E").replace("d", "e")) for t in xyz]


def read_gjf(path, fname):
    # Gaussianの入力ファイルから構造だけを読む。電荷・多重度は確認用に返すだけで、計算にはフォームの値を使う
    from ase import Atoms
    from ase.data import chemical_symbols
    from ase.io import read as ase_read

    with open(path, encoding="utf-8", errors="replace") as f:
        text = re.split(r"(?im)^\s*--link1--\s*$", f.read())[0]
    lines = [l.split("!")[0].rstrip() for l in text.splitlines()]
    i, n = 0, len(lines)
    while i < n and (not lines[i].strip() or lines[i].lstrip().startswith("%")):
        i += 1
    if i >= n or not lines[i].lstrip().startswith("#"):
        raise ColabUMAError(f"「{fname}」にルートセクション（# で始まる行）が見つかりませんでした。",
                            "Gaussianの入力ファイル（.gjf / .com）か確認してください。")
    for _ in range(2):  # ルートセクションとタイトルを読み飛ばす
        while i < n and lines[i].strip():
            i += 1
        while i < n and not lines[i].strip():
            i += 1
    cm = lines[i].replace(",", " ").split() if i < n else []
    try:
        gjf_cm = (int(cm[0]), int(cm[1]))
    except (IndexError, ValueError):
        raise ColabUMAError(f"「{fname}」の電荷・多重度の行を読み取れませんでした。",
                            "ルートセクションとタイトルの後に空行があり、その次の行が「0 1」のような電荷・多重度になっているか確認してください。") from None
    i += 1
    atom_lines = []
    while i < n and lines[i].strip():
        atom_lines.append(lines[i])
        i += 1
    parsed = [_gjf_cartesian_line(l) for l in atom_lines]
    if atom_lines and all(parsed):
        for raw, sym, _ in parsed:
            if sym == "Tv":
                raise ColabUMAError("gjfに並進ベクトル（Tv）があります。周期系の最適化には対応していません。",
                                    "Tv の行を消して、分子として計算してください。")
            if sym not in chemical_symbols or sym == "X":
                raise ColabUMAError(f"gjfの原子「{raw}」を元素として読み取れませんでした。",
                                    "ゴースト原子（Bq）やダミー原子（X）は使えません。その行を消してください。")
        return Atoms([p[1] for p in parsed], positions=[p[2] for p in parsed]), gjf_cm
    try:  # デカルト座標でなければ Z-matrix とみなして ASE で読む
        atoms = ase_read(path, format="gaussian-in")
        atoms.calc = None
        return atoms, gjf_cm
    except Exception:
        raise ColabUMAError(f"「{fname}」の分子構造を読み取れませんでした。",
                            "GaussViewで開いて保存し直すか、「元素記号 x y z」のデカルト座標で書かれているか確認してください。") from None


def read_frames(fname, data, charge, multiplicity, uma=True):
    """アップロードされたファイルの構造をすべて読む。gjf は1構造、xyz は複数構造のこともある。"""
    from ase.io import read as ase_read

    path = os.path.join(tempfile.mkdtemp(), os.path.basename(fname))
    with open(path, "wb") as f:
        f.write(data)
    if fname == DRAWN_NAME:
        atoms, smiles, formal = molfile_to_atoms(data.decode())
        if formal != charge:
            warn(f"描いた構造の形式電荷の合計（{formal:+d}）とフォームの電荷（{charge:+d}）が違います。計算にはフォームの値を使います。")
        return [prepare(atoms, charge, multiplicity, uma)], f"描いた構造: {smiles}（RDKit で 3D 化）"
    if os.path.splitext(fname)[1].lower() in GJF_EXTENSIONS:
        atoms, gjf_cm = read_gjf(path, fname)
        if gjf_cm != (charge, multiplicity):
            warn(f"「{fname}」に書かれた電荷・多重度（{gjf_cm[0]} {gjf_cm[1]}）とフォームの値（{charge} {multiplicity}）が違います。"
                 "計算にはフォームの値を使います。")
        frames, kind = [atoms], "gjf"
    else:
        frames = None
        for fmt in ("extxyz", "xyz"):
            try:
                frames = ase_read(path, index=":", format=fmt)
                break
            except Exception:
                pass
        if not frames:
            raise ColabUMAError(f"「{fname}」をxyzファイルとして読み込めませんでした。",
                                "1行目が原子数、2行目がコメント、3行目以降が「元素記号 x y z」（単位Å）の形式か確認してください。"
                                "Gaussianの入力ファイルなら拡張子を .gjf か .com にしてください。")
        kind = "xyz"
    return [prepare(a, charge, multiplicity, uma) for a in frames], f"{kind}: {fname}"


def read_structure(fname, data, charge, multiplicity, uma=True):
    """先頭の1構造だけを使う。(atoms, 入力の説明) を返す。"""
    frames, desc = read_frames(fname, data, charge, multiplicity, uma)
    if len(frames) > 1:
        warn(f"「{fname}」には {len(frames)} 個の構造が入っています。先頭の構造だけを使います。")
    return frames[0], desc


def prepare(atoms, charge, multiplicity, uma=True):
    # 分子として扱う設定にして、元素・電荷・多重度を確認する
    atoms = atoms.copy()
    atoms.calc = None
    atoms.set_constraint()
    atoms.set_pbc(False)
    atoms.set_cell(np.zeros((3, 3)))
    atoms.info = {}
    if uma:  # UMA を使わないときの元素の範囲は dft.run で確かめる
        check_elements(atoms.numbers)
    check_charge_multiplicity(atoms.numbers, charge, multiplicity)
    atoms.info.update({"charge": int(charge), "spin": int(multiplicity)})
    return atoms


def check_form(fmax, max_steps, multiplicity):
    if fmax <= 0:
        raise ColabUMAError("収束判定 fmax は正の数にしてください。", "通常は 0.02 eV/Å 程度です。")
    if max_steps < 1:
        raise ColabUMAError("最大ステップ数は1以上にしてください。", "通常は 500 程度です。")
    check_multiplicity(multiplicity)


def check_multiplicity(multiplicity):
    if multiplicity < 1:
        raise ColabUMAError("スピン多重度は1以上の整数にしてください。", "閉殻分子なら1（一重項）です。")


def check_charge_multiplicity(numbers, charge, multiplicity):
    n_electrons = int(np.sum(numbers)) - int(charge)
    n_unpaired = int(multiplicity) - 1
    if n_electrons <= 0 or n_unpaired > n_electrons or (n_electrons - n_unpaired) % 2:
        suggest = "1, 3, …" if n_electrons % 2 == 0 else "2, 4, …"
        raise ColabUMAError(
            f"電荷 {charge:+d}・スピン多重度 {multiplicity} の組み合わせはこの分子では成り立ちません（電子数 {n_electrons}）。",
            f"多重度は 2S+1 です。電子数が偶数なら多重度は奇数、奇数なら偶数になります。この電荷なら多重度は {suggest} が候補です。")


def out_prefix(name):
    return re.sub(r"[^\w\-.]", "_", name.strip()) or "result"


# --- xyz と 3D表示 ---
def xyz_block(symbols, positions, comment):
    lines = [str(len(symbols)), comment]
    lines += [f"{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}" for s, (x, y, z) in zip(symbols, positions)]
    return "\n".join(lines) + "\n"


def show_numbered(atoms, highlight=()):
    # 原子番号（1始まり）付きの3D表示。highlight の原子（0始まり）は赤
    import py3Dmol

    hl = set(highlight)
    view = py3Dmol.view(width=640, height=420)
    view.addModel(xyz_block(atoms.get_chemical_symbols(), atoms.positions, ""), "xyz")
    view.setStyle({"stick": {"radius": 0.12}, "sphere": {"scale": 0.2}})
    for i, (x, y, z) in enumerate(atoms.positions):
        view.addLabel(str(i + 1), {"position": {"x": float(x), "y": float(y), "z": float(z)}, "fontSize": 11,
                                   "fontColor": "white", "backgroundOpacity": 0.75,
                                   "backgroundColor": "#d62728" if i in hl else "#555555"})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.show()


def show_structure(xyz_text):
    import py3Dmol

    view = py3Dmol.view(width=640, height=420)
    view.addModel(xyz_text, "xyz")
    view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.24}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.show()


def show_animation(multi_xyz, interval=150):
    import py3Dmol

    view = py3Dmol.view(width=640, height=420)
    view.addModelsAsFrames(multi_xyz, "xyz")
    view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.24}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.animate({"loop": "backAndForth", "interval": interval})
    view.show()


def show_mode(atoms, mode, amplitude=0.6):
    # 振動モードのアニメーション（py3Dmol の vibrate）
    import py3Dmol

    lines = [str(len(atoms)), ""]
    for s, (x, y, z), (dx, dy, dz) in zip(atoms.get_chemical_symbols(), atoms.positions, mode):
        lines.append(f"{s} {x:.6f} {y:.6f} {z:.6f} {dx:.6f} {dy:.6f} {dz:.6f}")
    view = py3Dmol.view(width=640, height=420)
    view.addModel("\n".join(lines) + "\n", "xyz", {"vibrate": {"frames": 10, "amplitude": amplitude}})
    view.setStyle({"stick": {"radius": 0.14}, "sphere": {"scale": 0.24}})
    view.setBackgroundColor("white")
    view.animate({"loop": "backAndForth"})
    view.zoomTo()
    view.show()
