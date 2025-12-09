from flask import Flask, render_template
from flask_cors import CORS
from config import Config
from extensions import db
import os

def create_app():
    app = Flask(__name__,
                template_folder='../frontend/templates',
                static_folder='../frontend/static')
    app.config.from_object(Config)

    db.init_app(app)
    CORS(app)

    from routes.news import news_bp
    from routes.sport import sport_bp
    from routes.economic import economic_bp

    app.register_blueprint(news_bp, url_prefix='/news')
    app.register_blueprint(sport_bp, url_prefix='/sport')
    app.register_blueprint(economic_bp, url_prefix='/economic')

    @app.route('/')
    def index():
        return render_template('index.html')

    with app.app_context():
        db.create_all()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
