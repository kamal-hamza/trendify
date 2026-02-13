.PHONY: help install install-dev migrate makemigrations runserver shell test test-cov lint format check clean superuser

# Default target
help:
	@echo "Available commands:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make migrate          - Run database migrations"
	@echo "  make makemigrations   - Create new migrations"
	@echo "  make runserver        - Start Django development server"
	@echo "  make shell            - Start Django shell"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make lint             - Run linters (flake8, pylint)"
	@echo "  make format           - Format code with black and isort"
	@echo "  make check            - Run all checks (format, lint, test)"
	@echo "  make clean            - Remove Python cache files"
	@echo "  make superuser        - Create Django superuser"
	@echo "  make pre-commit       - Install pre-commit hooks"
	@echo "  make pre-commit-run   - Run pre-commit on all files"

# Install production dependencies
install:
	uv pip install -r requirements.txt

# Install development dependencies
install-dev:
	uv pip install -r requirements.txt
	uv pip install black isort flake8 flake8-django pylint pylint-django mypy django-stubs djangorestframework-stubs pytest pytest-django pytest-cov pytest-xdist factory-boy faker ipython ipdb pre-commit

# Run database migrations
migrate:
	python manage.py migrate

# Create new migrations
makemigrations:
	python manage.py makemigrations

# Start development server
runserver:
	python manage.py runserver

# Start Django shell
shell:
	python manage.py shell

# Run tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=api --cov=backend --cov-report=html --cov-report=term-missing

# Run tests in parallel
test-parallel:
	pytest -n auto

# Run linters
lint:
	@echo "Running flake8..."
	flake8 api backend --max-line-length=100 --extend-ignore=E203,W503 --exclude=migrations
	@echo "Running pylint..."
	pylint api backend --load-plugins=pylint_django --django-settings-module=backend.settings --ignore=migrations

# Format code
format:
	@echo "Running black..."
	black api backend --line-length=100 --exclude=migrations
	@echo "Running isort..."
	isort api backend --profile=black --line-length=100 --skip=migrations

# Check formatting without making changes
format-check:
	@echo "Checking black..."
	black api backend --check --line-length=100 --exclude=migrations
	@echo "Checking isort..."
	isort api backend --check-only --profile=black --line-length=100 --skip=migrations

# Type checking
type-check:
	mypy api backend --exclude=migrations

# Run Django checks
django-check:
	python manage.py check
	python manage.py makemigrations --check --dry-run --no-input

# Run all checks
check: format-check lint django-check test

# Clean Python cache files
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build

# Create superuser
superuser:
	python manage.py createsuperuser

# Install pre-commit hooks
pre-commit:
	pre-commit install

# Run pre-commit on all files
pre-commit-run:
	pre-commit run --all-files

# Collect static files
collectstatic:
	python manage.py collectstatic --no-input

# Database backup (SQLite only)
backup-db:
	@echo "Backing up database..."
	cp db.sqlite3 db.sqlite3.backup.$(shell date +%Y%m%d_%H%M%S)

# Load fixtures
loaddata:
	python manage.py loaddata $(FIXTURE)

# Dump data to fixture
dumpdata:
	python manage.py dumpdata --indent=2 -o $(FIXTURE)

# Show migrations status
showmigrations:
	python manage.py showmigrations

# Flush database
flush:
	python manage.py flush --no-input

# Reset database (flush + migrate)
reset-db: flush migrate

# Development setup (install + migrate + superuser)
setup-dev: install-dev migrate
	@echo "Development environment setup complete!"
	@echo "Run 'make superuser' to create an admin user"
	@echo "Run 'make runserver' to start the development server"

# Production setup
setup-prod: install migrate collectstatic
	@echo "Production environment setup complete!"

# Check Python version
check-python:
	@python --version | grep "Python 3.12" || (echo "Error: Python 3.12 required" && exit 1)
	@echo "Python version check passed"

# Security check
security:
	bandit -r api backend -c pyproject.toml

# Update dependencies
update-deps:
	uv pip install --upgrade -r requirements.txt

# Show Django settings
show-settings:
	python manage.py diffsettings

# Generate SECRET_KEY
generate-secret:
	@python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"