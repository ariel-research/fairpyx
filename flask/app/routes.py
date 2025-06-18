"""
Flask routes for the RFAOII demo site
-------------------------------------
Only *parsing / encoding* is done here; the algorithms module is
*never* touched.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import io, json, logging, urllib.parse
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from flask import (
    Blueprint, render_template, request, redirect, url_for, abort
)

import fairpyx.algorithms.repeated_Fair_Allocation_of_Indivisible_Items as rfaoi

bp = Blueprint("main", __name__)
log = logging.getLogger("fairpyx.web")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------ helpers ----
def _coerce_utils(raw: str) -> dict[int, dict[int, float]]:
    """
    Accept *either* Python-dict syntax *or* JSON and coerce all keys to int.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: user pasted a Python literal
        data = eval(raw, {})          # noqa – only because user gave the code
    out: dict[int, dict[int, float]] = {}
    for a_lbl, items in data.items():
        a = int(a_lbl)
        out[a] = {int(it): float(val) for it, val in items.items()}
    if set(out) != {0, 1}:
        abort(400, "The current demo supports *exactly* two agents (0/1).")
    return out


def _run_with_log_capture(func, *args, **kw):
    """
    Run *func* while capturing

      •  anything written to stdout / stderr
      •  *all* logging messages  (any level, any logger)

    The function’s return-value is passed through unchanged, the collected
    text is returned as the second tuple element.
    """
    buf = io.StringIO()

    # ① hook stdout / stderr  →  our in-memory buffer
    with redirect_stdout(buf), redirect_stderr(buf):

        # ② attach a temporary logging handler that writes into the same buffer
        root              = logging.getLogger()
        prev_level        = root.level
        stream_handler    = logging.StreamHandler(buf)
        stream_handler.setFormatter(
            logging.Formatter("%(levelname).1s %(name)s: %(message)s")
        )
        root.setLevel(logging.DEBUG)         # grab DEBUG too
        root.addHandler(stream_handler)

        try:
            result = func(*args, **kw)
        finally:
            # cleanup – remove the temporary handler again
            root.removeHandler(stream_handler)
            root.setLevel(prev_level)

    return result, buf.getvalue()


# ---------------------------------------------------------------- main page ----
@bp.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        utils_raw = request.form["utilities"]
        k         = int(request.form.get("k", 2))

        # keep the raw-text utilities in the query string
        q = {
            "k": k,
            "utilities": urllib.parse.quote_plus(utils_raw),
        }

        # which algorithm page?
        algo  = request.form.get("algo", "1")
        dest  = "main.algo1" if algo == "1" else "main.algo2"   # <── fixed line
        return redirect(url_for(dest, **q))

    return render_template("index.html")


# -------------------------------------------------------- algorithm 1 page ----
@bp.route("/algo1", endpoint="algo1")
def algo1_view():
    utils = _coerce_utils(urllib.parse.unquote_plus(request.args["utilities"]))
    k     = int(request.args.get("k", 2))
    if k != 2:
        abort(400, "Algorithm 1 only works for k = 2 rounds.")

    rounds, log_txt = _run_with_log_capture(rfaoi.algorithm1, utils)
    checks = [
        [rfaoi.EF1_holds(rd, a, utils) for a in (0, 1)]
        for rd in rounds
    ]
    return render_template(
        "algo1.html",
        utils=utils, rounds=rounds, checks=checks, log_txt=log_txt
    )


# -------------------------------------------------------- algorithm 2 page ----
@bp.route("/algo2", endpoint="algo2")
def algo2_view():
    utils = _coerce_utils(urllib.parse.unquote_plus(request.args["utilities"]))
    k     = int(request.args.get("k", 4))
    if k % 2:
        abort(400, "Algorithm 2 needs *even* k.")

    rounds, log_txt = _run_with_log_capture(rfaoi.algorithm2, k, utils)
    checks = [
        [rfaoi.weak_EF1_holds(rd, a, utils) for a in (0, 1)]
        for rd in rounds
    ]
    return render_template(
        "algo2.html",
        utils=utils, rounds=rounds, checks=checks, k=k, log_txt=log_txt
    )

# ------------------------------------------------------------- error handler ----
@bp.route("/about")
def about():
    return render_template("about.html")
