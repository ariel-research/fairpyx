from flask_wtf import FlaskForm
from wtforms import TextAreaField, IntegerField, SubmitField
from wtforms.validators import InputRequired, NumberRange

class UtilitiesForm(FlaskForm):
    """
    A simple text-area where the user types a JSON object of the form:
        {
          "0": {"0": 3, "1": -2, "2": 5},
          "1": {"0": 1, "1": -3, "2": 2}
        }
    plus an integer k (even, for Algorithm 2).
    """
    utilities_json = TextAreaField(
        "Utilities (JSON dict of dicts)",
        validators=[InputRequired()],
        render_kw={"rows": 6, "placeholder": '{"0":{"0":3,"1":-2},"1":{"0":1,"1":4}}'}
    )
    k = IntegerField(
        "k (even, Algorithm 2 only)",
        default=4,
        validators=[NumberRange(min=2)]
    )
    submit = SubmitField("Compute")
