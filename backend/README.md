# Trendify Backend

The backend for Trendify is a Django-based REST API that aggregates and analyzes tech trends from various platforms like GitHub, Hacker News, Product Hunt, and more.

## Setup Instructions

### Prerequisites
- Python 3.10+
- Redis (for Celery task queue)
- Virtual Environment (recommended)

### Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env.development` file in the `trendify/backend/` directory with the following variables:
   ```env
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=sqlite:///db.sqlite3
   REDIS_URL=redis://localhost:6379/0
   
   # External API Tokens (Optional but recommended for full data)
   PRODUCT_HUNT_API_TOKEN=your_product_hunt_token
   GITHUB_TOKEN=your_github_personal_access_token
   ```

4. **Run Migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Start the Development Server:**
   ```bash
   python manage.py runserver
   ```

## Data Fetching

To populate the database with current trend data, use the Django management command:

```bash
python manage.py fetch_data --platform all
```

### Options:
- `--platform {all,hn,reddit,github}`: Specify a platform to fetch from (default: all).
- `--pipeline`: Run the full pipeline including processing, metrics, and alerts.
- `--async`: Run as an asynchronous Celery task.

This command fetches data from:
- GitHub Trending
- Hacker News (Show HN)
- Reddit (if configured)
- And other integrated sources

## Background Tasks

Trendify uses Celery for periodic data fetching and analysis.

1. **Start Celery Worker:**
   ```bash
   celery -A backend worker -l info
   ```

2. **Start Celery Beat (Scheduler):**
   ```bash
   celery -A backend beat -l info
   ```

## API Documentation
Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/api/schema/swagger-ui/`
- Redoc: `http://localhost:8000/api/schema/redoc/`
