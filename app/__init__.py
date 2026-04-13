from flask import Flask

from .routes import bp
from .services.store import PatientStore
from .services.sepsis_engine import SepsisEngine
from .services.simulator import PatientSimulator


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "sepsis-demo-secret"

    store = PatientStore()
    engine = SepsisEngine()
    simulator = PatientSimulator(store)

    app.config["STORE"] = store
    app.config["ENGINE"] = engine
    app.config["SIMULATOR"] = simulator

    # Start real-time simulation
    simulator.start()

    app.register_blueprint(bp)
    return app
