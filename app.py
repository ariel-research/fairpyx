import sys
import os
import json
import subprocess
from io import StringIO

# Add current directory to Python path so local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from flask import Flask, render_template, request, render_template_string
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.algorithms.leximin_primal import leximin_primal

# Initialize the Flask app
app = Flask(__name__)

import logging

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # === Parse form inputs ===
            valuations_raw = json.loads(request.form["valuations"])
            agent_capacities_raw = json.loads(request.form["agent_capacities"])
            item_capacities_raw = json.loads(request.form["item_capacities"])

            # === Show raw input for debugging ===
            print("[RAW] Valuations:", valuations_raw)
            print("[RAW] Agent Capacities:", agent_capacities_raw)
            print("[RAW] Item Capacities:", item_capacities_raw)

            # === Normalize key types ===
            valuations = {
                int(agent_id): {str(item): int(val) for item, val in items.items()}
                for agent_id, items in valuations_raw.items()
            }
            agent_capacities = {int(k): int(v) for k, v in agent_capacities_raw.items()}
            item_capacities = {str(k): int(v) for k, v in item_capacities_raw.items()}

            # === Show cleaned input ===
            print("[CLEANED] Valuations:", valuations)
            print("[CLEANED] Agent Capacities:", agent_capacities)
            print("[CLEANED] Item Capacities:", item_capacities)

            # === Setup logging capture ===
            log_stream = StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

            # === Run LeximinPrimal algorithm ===
            inst = Instance(
                valuations=valuations,
                agent_capacities=agent_capacities,
                item_capacities=item_capacities
            )
            alloc = AllocationBuilder(inst)
            leximin_primal(alloc)

            # === Collect logs ===
            handler.flush()
            logger.removeHandler(handler)
            logs = log_stream.getvalue()

            input_data = json.dumps({
                "valuations": valuations,
                "agent_capacities": agent_capacities,
                "item_capacities": item_capacities
            }, indent=2)

            return render_template(
                "result.html",
                input_data=input_data,
                distribution=alloc.distribution,
                logs=logs
            )

        except Exception as e:
            logger.removeHandler(handler)
            return render_template("index.html", error=str(e))

    return render_template("index.html")


@app.route("/run-tests")
def run_tests():
    """
    Run pytest for LeximinPrimal-related tests and show results.
    """
    try:
        result = subprocess.run(
            ["pytest", "-v", "-s", "tests/test_leximin_primal.py"],
            capture_output=True,
            text=True
        )
        output = result.stdout + result.stderr
    except Exception as e:
        output = f"Error running tests: {str(e)}"

    return render_template_string("""
    <html>
    <head><title>Test Results</title></head>
    <body style="font-family: monospace; background: #f4f6f8; padding: 20px;">
        <h1>🧪 LeximinPrimal Test Results</h1>
        <pre style="background:white; padding:20px; border-radius:8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);">{{ output }}</pre>
        <a href="/"><button>⬅ Back</button></a>
    </body>
    </html>
    """, output=output)


# Entry point for local dev server
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app.run(debug=True, host="0.0.0.0", port=5050)
