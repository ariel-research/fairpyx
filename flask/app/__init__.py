from pathlib import Path
from flask import Flask
from . import routes   # registers blue-prints


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev-key"          # WTForms CSRF

    # serve bootstrap from local file instead of CDN
    static_dir = Path(__file__).with_name("static")
    app.static_folder = static_dir

    app.register_blueprint(routes.bp)
    return app
