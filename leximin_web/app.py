from flask import Flask, render_template, request, render_template_string
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder
from fairpyx.algorithms.leximin_primal import leximin_primal
import json
import subprocess

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            valuations = json.loads(request.form["valuations"])
            agent_capacities = json.loads(request.form["agent_capacities"])
            item_capacities = json.loads(request.form["item_capacities"])

            inst = Instance(valuations=valuations,
                            agent_capacities=agent_capacities,
                            item_capacities=item_capacities)
            alloc = AllocationBuilder(inst)
            leximin_primal(alloc)

            return render_template("result.html", bundles=alloc.bundles, distribution=alloc.distribution)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

@app.route("/run-tests")
def run_tests():
    try:
        # Run only LeximinPrimal-related tests with prints and verbose output
        result = subprocess.run(
            ["pytest", "-v", "-s", "../tests/test_leximin_primal.py"],
            capture_output=True,
            text=True
        )
        output = result.stdout + result.stderr
    except Exception as e:
        output = f"Error running tests: {str(e)}"

    # Show result inline using simple HTML
    return render_template_string("""
    <html>
    <head><title>Test Results</title></head>
    <body style="font-family: monospace; background: #f4f6f8; padding: 20px;">
        <h1>🧪 LeximinPrimal Test Results</h1>
        <pre style="background:white; padding:20px; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">{{ output }}</pre>
        <a href="/"><button>⬅ Back</button></a>
    </body>
    </html>
    """, output=output)


if __name__ == "__main__":
    app.run(debug=True)
