import json
from flask import Flask, render_template, request, redirect, url_for
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms.Optimization_Matching.FaSt import FaSt
from fairpyx.algorithms.Optimization_Matching.FaStGen import FaStGen
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Helper functions to create student and college lists
def make_s_list(n):
    return {f"s{i + 1}" for i in range(int(n))}


def make_c_list(m):
    return {f"c{i + 1}" for i in range(int(m))}


def fastgen_make_s_list(n):
    return [f"s{i + 1}" for i in range(int(n))]


def fastgen_make_c_list(m):
    return [f"c{i + 1}" for i in range(int(m))]


# Route for the homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('firstpage.html')


# Route for the 'Try Demo' page
@app.route('/demo', methods=['GET'])
def try_demo():
    return render_template('tryDemo.html')


# Route for the 'Results Demo' page
@app.route('/results', methods=['POST'])
def results_demo():
    pass

if __name__ == "__main__":
    app.run(debug=True)
