"""出力ファイルをまとめて ZIP にし、ダウンロードする。"""
import datetime
import os
import tempfile
import zipfile

from . import env


def log_text(title, source_desc, job, sections):
    """log.txt の中身。sections は各計算が返したログの行のリスト。"""
    lines = [title, f"Date: {datetime.datetime.now().isoformat(timespec='seconds')}", "",
             "[Input]", f"Input: {source_desc}"] + job.condition_lines()
    lines += ["", "[Software]"] + env.software_lines()
    for sec in sections:
        lines += [""] + sec
    return "\n".join(lines) + "\n"


def save_zip(prefix, files, download=True):
    """files は {"opt.xyz": 中身, ...}。<prefix>_<名前> で書いて ZIP にまとめる。ZIP のパスを返す。"""
    out_dir = os.path.join(tempfile.mkdtemp(), prefix)
    os.makedirs(out_dir)
    paths = []
    for name, text in files.items():
        p = os.path.join(out_dir, f"{prefix}_{name}")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(p)
    zip_path = os.path.join(os.path.dirname(out_dir), f"{prefix}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, os.path.basename(p))
    print("Files in the ZIP:", ", ".join(os.path.basename(p) for p in paths))
    if download:
        try:
            from google.colab import files as colab_files
            colab_files.download(zip_path)
        except ImportError:
            print("ZIP saved:", zip_path)
    return zip_path
