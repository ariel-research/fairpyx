from flask import Flask

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev-key"    # WTForms needs it

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app
