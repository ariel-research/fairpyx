import json
from flask import Flask, render_template, request
from fairpyx import Instance, AllocationBuilder
from fairpyx.algorithms.Optimization_Matching.FaSt import FaSt
from fairpyx.algorithms.Optimization_Matching.FaStGen import FaStGen
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def make_s_list(n):
    """Converts a number into a set of student identifiers."""
    return {f"s{i+1}" for i in range(int(n))}

def make_c_list(m):
    """Converts a number into a set of college identifiers."""
    return {f"c{i+1}" for i in range(int(m))}

@app.route('/', methods=['GET', 'POST'])
def index():
    logger.debug("Entered index function")
    error = None
    result = None

    if request.method == 'POST':
        try:
            # Parse input data from the form
            agents = int(request.form['k'])
            items = int(request.form['m'])
            list_students = make_s_list(agents)
            list_colleges = make_c_list(items)

            logger.info("Received number of students (n): %s", agents)
            logger.info("Received number of colleges (m): %s", items)
            logger.info("List of students: %s", list_students)
            logger.info("List of colleges: %s", list_colleges)

            algorithm = request.form['algorithm']
            logger.info("Selected algorithm: %s", algorithm)

            # Select and run the appropriate algorithm
            if algorithm == 'FaSt':
                valuation = request.form['valuation']
                logger.debug("Received valuation (JSON): %s", valuation)
                valuation_dict = json.loads(valuation)
                logger.debug("Converted valuation to dict: %s", valuation_dict)
                
                ins = Instance(agents=list_students, items=list_colleges, valuations=valuation_dict)
                alloc = AllocationBuilder(instance=ins)
                logger.debug("Instance and AllocationBuilder created")
                
                result = FaSt(alloc)
                logger.debug("FaSt algorithm result: %s", result)
                
            elif algorithm == 'FaStGen':
                agents_valuation = request.form['agentsValues']
                logger.debug("Received agentsValues (JSON): %s", agents_valuation)
                agents_valuation_dict = json.loads(agents_valuation)
                logger.debug("Converted agentsValues to dict: %s", agents_valuation_dict)
                ins = Instance(agents=list_students, items=list_colleges, valuations=agents_valuation_dict)
                alloc = AllocationBuilder(instance=ins)
                logger.debug("Instance and AllocationBuilder created for FaStGen")

                items_valuation=request.form['itemsValues']
                logger.debug("Received itemsValues (JSON): %s", items_valuation)
                items_valuation_dict=json.loads(items_valuation)
                logger.debug("Converted itemsValues to dict: %s", items_valuation_dict)
                result = FaStGen(alloc, items_valuations=items_valuation_dict)
                logger.debug("FaStGen algorithm result: %s", result)
                
            else:
                error = "Invalid algorithm selected."
                logger.warning("Invalid algorithm selected by user")

        except json.JSONDecodeError as jde:
            error = "Invalid JSON format. Please check your input."
            logger.error("JSON decode error: %s", jde)
        except Exception as e:
            error = str(e)
            logger.error("An error occurred: %s", error)

    return render_template('index.html', error=error, result=result)

if __name__ == '__main__':
    app.run(debug=True)