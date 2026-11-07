"""利用者に見せるメッセージ、エラー表示、プログレスバー。"""
import html
import time

import numpy as np
from IPython.display import HTML, display


class ColabUMAError(Exception):
    # 利用者に見せるエラー。トレースバックの代わりに、英語の一文と対処法を表示し、その下に日本語も添える
    def __init__(self, message, hint="", ja="", ja_hint=""):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.ja = ja
        self.ja_hint = ja_hint


def _box(kind, message, hint="", ja="", ja_hint=""):
    bg, fg, label = {
        "error": ("#fdecea", "#b3261e", "Error"),
        "warn": ("#fff4e5", "#8a5300", "Note"),
        "ok": ("#e8f5e9", "#1b5e20", "OK"),
    }[kind]
    sub = "style='margin-top:2px;font-size:13px;opacity:.8'"
    body = f"<b>{label}</b>　{html.escape(message)}"
    if ja:
        body += f"<div {sub}>{html.escape(ja)}</div>"
    if hint:
        body += f"<div style='margin-top:4px'>Fix: {html.escape(hint)}</div>"
    if ja_hint:
        body += f"<div {sub}>対処法：{html.escape(ja_hint)}</div>"
    display(HTML(
        f"<div style='background:{bg};color:{fg};padding:10px 14px;border-radius:6px;"
        f"margin:6px 0;font-size:14px;line-height:1.6'>{body}</div>"
    ))


def warn(message, hint=""):
    _box("warn", message, hint)


def ok(message):
    _box("ok", message)


def heading(text):
    display(HTML(f"<div style='font-size:15px;font-weight:bold;margin:14px 0 4px'>{html.escape(text)}</div>"))


def _show_colabuma_error(shell, etype, evalue, tb, tb_offset=None):
    _box("error", evalue.message, evalue.hint, evalue.ja, evalue.ja_hint)
    return []


def install_error_handler():
    # ColabUMAError をトレースバックではなく箱で表示する
    from IPython import get_ipython

    shell = get_ipython()
    if shell is not None:
        shell.set_custom_exc((ColabUMAError,), _show_colabuma_error)


# --- プログレスバー ---
def bar_row(label, fraction, text):
    fraction = min(max(fraction, 0.0), 1.0)
    color = "#34a853" if fraction >= 1.0 else "#1a73e8"
    return (f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px'>"
            f"<div style='flex:none;width:7em'>{html.escape(label)}</div>"
            f"<div style='flex:1;max-width:320px;height:14px;background:#dadce0;border-radius:4px;overflow:hidden'>"
            f"<div style='width:{fraction * 100:.1f}%;height:100%;background:{color}'></div></div>"
            f"<div style='flex:none'>{html.escape(text)}</div></div>")


def convergence_fraction(start, current, target):
    # 最初の値から収束判定までに、桁でどこまで近づいたか（0〜1）。値が上がればバーも戻る
    if current < target:
        return 1.0
    if start <= target:
        return 0.0
    return float(np.log(start / current) / np.log(start / target))


class ProgressDisplay:
    # 同じ場所を書き換えて表示する。更新は 0.2 秒に1回まで（force なら必ず）
    def __init__(self):
        self._handle = display(HTML(""), display_id=True)
        self._last = 0.0

    def show(self, rows, force=False):
        now = time.time()
        if self._handle is not None and (force or now - self._last >= 0.2):
            self._handle.update(HTML("".join(rows)))
            self._last = now
