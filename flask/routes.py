"""
Flask routes for the Repeated-Fair-Allocation demo website.
Everything here is **integer-only** (the algorithms already expect ints).
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import json
from typing import Dict

from flask import (
    Blueprint, render_template, request, redirect, url_for, flash
)
from werkzeug.exceptions import BadRequestKeyError

from fairpyx.algorithms.repeated_Fair_Allocation_of_Indivisible_Items import (
    algorithm1,
    algorithm2,
    weak_EF1_holds,
    EF1_holds,
)

main_bp = Blueprint("main", __name__)

# --------------------------------------------------------------------- #
# util – parse the textarea input (expects *valid* Python/JSON literal) #
# --------------------------------------------------------------------- #
def _parse_utilities(txt: str) -> Dict[int, Dict[int, int]]:
    """
    Accepts either JSON like:  {"0": {"0": 11, "1": 22}, "1": {"0": 22, "1": 11}}
    or Python-ish without quotes: {0:{0:11,1:22}, 1:{0:22,1:11}}
    and returns a **proper** utilities dict with *int* keys and values.
    """
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # fallback – eval in a tiny, safe environment
        try:
            data = eval(txt, {"__builtins__": {}})
        except Exception as exc:   # noqa: BLE001
            raise ValueError(f"Cannot parse utilities: {exc}") from exc

    # convert keys to int
    utils: Dict[int, Dict[int, int]] = {}
    for a_str, item_dict in data.items():
        agent = int(a_str)
        utils[agent] = {int(o_str): int(v) for o_str, v in item_dict.items()}
    return utils


# --------------------------------------------------------------------- #
# routes                                                                #
# --------------------------------------------------------------------- #

@main_bp.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            util_text = request.form["utilities"]
            k         = int(request.form["k"])
            algo      = request.form["algo"]    # "1" or "2"
            utils = _parse_utilities(util_text)
        except (BadRequestKeyError, ValueError) as exc:
            flash(f"Input error: {exc}", "danger")
            return redirect(url_for("main.home"))

        if algo == "1":
            return redirect(url_for("main.algo1_view",
                                    utilities=json.dumps(utils), k=2))
        else:
            return redirect(url_for("main.algo2_view",
                                    utilities=json.dumps(utils), k=k))

    return render_template("home.html")


@main_bp.route("/algo1")
def algo1_view():
    utils   = _parse_utilities(request.args["utilities"])
    rounds  = algorithm1(utils)                   # k == 2 internally

    # checks per round
    checks = [
        {a: EF1_holds(r, a, utils) for a in (0, 1)}
        for r in rounds
    ]
    return render_template("algo1.html",
                           rounds=rounds,
                           utils=utils,
                           checks=checks)


@main_bp.route("/algo2")
def algo2_view():
    utils = _parse_utilities(request.args["utilities"])
    k     = int(request.args.get("k", 4))
    rounds = algorithm2(k, utils)                # we refactored algo2(k, utils)

    checks = [
        {a: weak_EF1_holds(r, a, utils) for a in (0, 1)}
        for r in rounds
    ]
    return render_template("algo2.html",
                           rounds=rounds,
                           utils=utils,
                           checks=checks,
                           k=k)
