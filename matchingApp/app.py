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
    # Get the algorithm type
    algorithm = request.form['algorithm']

    # Get number of students and colleges
    num_students = request.form['num_students']
    num_colleges = request.form['num_colleges']

    # Get the students' valuations (nested form data)
    student_valuations = {}
    for student in range(1, int(num_students) + 1):
        student_id = f"S{student}"
        student_valuations[student_id] = {}
        for college in range(1, int(num_colleges) + 1):
            college_id = f"C{college}"
            student_valuations[student_id][college_id] = int(request.form.get(f'valuation[{student_id}][{college_id}]'))

    # Get the colleges' valuations (for FaStGen algorithm only)
    college_valuations = {}
    if algorithm == 'FaStGen':
        for college in range(1, int(num_colleges) + 1):
            college_id = f"C{college}"
            college_valuations[college_id] = {}
            for student in range(1, int(num_students) + 1):
                student_id = f"S{student}"
                college_valuations[college_id][student_id] = int(request.form.get(f'colleges_valuation[{college_id}][{student_id}]'))

    # Log for debugging
    logger.debug(request.form)
    logger.debug(f"Algorithm: {algorithm}")
    logger.debug(f"Number of Students: {num_students}")
    logger.debug(f"Number of Colleges: {num_colleges}")
    logger.debug(f"Student Valuations: {json.dumps(student_valuations, indent=4)}")
    if algorithm == 'FaStGen':
        logger.debug(f"College Valuations: {json.dumps(college_valuations, indent=4)}")

    # Process the data with the selected algorithm
    # FaSt or FaStGen processing can go here

    # Pass all data to the resultsDemo.html template
    return render_template('resultsDemo.html', algorithm=algorithm, num_students=num_students, num_colleges=num_colleges,
                           student_valuations=student_valuations, college_valuations=college_valuations)


if __name__ == "__main__":
    app.run(debug=True)
