from flask import Flask

from .routes import bp
from .routes_human_loop import human_loop_bp
from .services.store import PatientStore
from .services.sepsis_engine import SepsisEngine
from .services.simulator import PatientSimulator


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "sepsis-demo-secret-key-change-in-production"
    
    # Enable session management
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_PERMANENT"] = False

    store = PatientStore()
    engine = SepsisEngine()
    simulator = PatientSimulator(store)

    app.config["STORE"] = store
    app.config["ENGINE"] = engine
    app.config["SIMULATOR"] = simulator

    # Start real-time simulation
    simulator.start()

    app.register_blueprint(bp)
    app.register_blueprint(human_loop_bp)
    return app
