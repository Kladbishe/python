# News Portal - Flask Application

A simple news portal built with Flask featuring three categories: News, Sport, and Economic.

## Features

- Three separate categories with dedicated blueprints
- SQLite/MySQL database support via SQLAlchemy
- Create and view articles
- RESTful API endpoints
- Responsive design
- CORS enabled

## Project Structure

```
9.12/
├── backend/              # Backend application
│   ├── routes/          # Route blueprints
│   │   ├── __init__.py
│   │   ├── news.py
│   │   ├── sport.py
│   │   └── economic.py
│   ├── app.py          # Flask application factory
│   ├── config.py       # Configuration settings
│   ├── models.py       # Database models
│   ├── extensions.py   # Flask extensions
│   └── add_sample_data.py  # Script to populate database
├── frontend/            # Frontend assets
│   ├── templates/      # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── news/
│   │   ├── sport/
│   │   └── economic/
│   └── static/         # Static files (CSS, JS, images)
│       └── css/
│           └── style.css
├── venv/               # Virtual environment
├── instance/           # SQLite database files
├── run.py             # Application entry point
├── requirements.txt   # Python dependencies
└── MYSQL_SETUP.md    # MySQL setup guide
```

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add sample data (optional):
```bash
python3 backend/add_sample_data.py
```

4. Run the application:
```bash
python3 run.py
```

The application will be available at `http://localhost:5000`

## Quick Start (Next Time)

If you've already installed dependencies:

```bash
source venv/bin/activate  # On macOS/Linux
python3 run.py
```

Or on Windows:
```bash
venv\Scripts\activate
python run.py
```

## Database Configuration

### SQLite (Default)
No additional setup required. Database file will be created automatically.

### MySQL
See detailed MySQL setup instructions in `MYSQL_SETUP.md`

## API Endpoints

Each category has the following endpoints:

- `GET /news/` - List all news articles
- `GET /news/<id>` - View single article
- `POST /news/create` - Create new article
- `GET /news/api` - JSON API endpoint

Same pattern for `/sport/` and `/economic/`

## Usage

1. Visit the homepage to see all categories
2. Click on a category to view articles
3. Create new articles using the "Create Article" button
4. Each article can be viewed individually

## Production Deployment

For production, use gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:create_app()
```

## Technologies Used

- Flask 3.1.2
- Flask-SQLAlchemy 3.1.1
- Flask-CORS 6.0.1
- Flask-MySQLdb 2.0.0
- Gunicorn 23.0.0
