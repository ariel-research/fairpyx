# app/forms.py
from flask_wtf import FlaskForm
from wtforms import TextAreaField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class UtilitiesForm(FlaskForm):
    utilities_json = TextAreaField(
        "Utilities (JSON or Python-dict)",
        default='{\n  "0": {"0":11,"1":22},\n  "1": {"0":11,"1":22}\n}',
        validators=[DataRequired()],
        render_kw={"rows":5}
    )
    k = IntegerField(
        "k (number of rounds)",
        default=2,
        validators=[DataRequired(), NumberRange(min=2)]
    )
    algo = SelectField(
        "Algorithm",
        choices=[("1", "Algorithm 1 (k=2)"), ("2", "Algorithm 2 (k even)")],
        default="2"
    )
    submit = SubmitField("Run")
