"""
Management command for manually fetching data from platforms
Can be used for testing or one-off data collection
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from api.tasks import (
    fetch_all_platforms,
    fetch_github,
    fetch_hacker_news,
    fetch_reddit,
    full_pipeline_daily,
)


class Command(BaseCommand):
    help = "Manually fetch trending data from platforms (HN, Reddit, GitHub)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--platform",
            type=str,
            choices=["all", "hn", "reddit", "github"],
            default="all",
            help="Platform to fetch from (default: all)",
        )
        parser.add_argument(
            "--async",
            action="store_true",
            dest="use_async",
            help="Run as async Celery task instead of synchronously",
        )
        parser.add_argument(
            "--pipeline",
            action="store_true",
            help="Run full pipeline (fetch + process + metrics + alerts)",
        )

    def handle(self, *args, **options):
        platform = options["platform"]
        use_async = options["use_async"]
        run_pipeline = options["pipeline"]

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("TRENDIFY DATA FETCHER")
        self.stdout.write("=" * 70)

        if run_pipeline:
            self._handle_pipeline(use_async)
        else:
            self._handle_fetch(platform, use_async)

    def _handle_pipeline(self, use_async):
        """Run the full daily pipeline"""
        self.stdout.write("\nRunning full daily pipeline...")
        self.stdout.write("   Steps: Fetch -> Sentiment -> Topics -> Metrics -> Alerts")

        if use_async:
            self.stdout.write("\nQueuing pipeline as async Celery task...")
            result = full_pipeline_daily.delay()
            self.stdout.write(
                self.style.SUCCESS(f"[SUCCESS] Pipeline queued! Task ID: {result.id}")
            )
            self.stdout.write("\nMonitor progress with:")
            self.stdout.write(f"  celery -A backend inspect active")
            self.stdout.write(f"  # Or check task result:")
            self.stdout.write(f"  python manage.py shell")
            self.stdout.write(f"  >>> from celery.result import AsyncResult")
            self.stdout.write(f"  >>> result = AsyncResult('{result.id}')")
            self.stdout.write(f"  >>> result.status")
        else:
            self.stdout.write("\nRunning pipeline synchronously...")
            self.stdout.write("(This may take several minutes)\n")

            try:
                result = full_pipeline_daily()
                self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Pipeline complete!"))
                self.stdout.write(f"Chain ID: {result.get('chain_id')}")
                self.stdout.write(f"Started at: {result.get('timestamp')}")
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"\n[ERROR] Pipeline failed: {e}"))

    def _handle_fetch(self, platform, use_async):
        """Fetch data from specified platform(s)"""
        self.stdout.write(f"\nFetching from: {platform.upper()}")

        # Select appropriate task
        if platform == "all":
            task = fetch_all_platforms
            task_args = []
        elif platform == "hn":
            task = fetch_hacker_news
            task_args = []
        elif platform == "reddit":
            task = fetch_reddit
            task_args = []
        elif platform == "github":
            task = fetch_github
            task_args = []
        else:
            raise CommandError(f"Unknown platform: {platform}")

        if use_async:
            self.stdout.write("\nQueuing fetch task...")
            result = task.delay(*task_args)
            self.stdout.write(
                self.style.SUCCESS(f"[SUCCESS] Task queued! Task ID: {result.id}")
            )
            self.stdout.write("\nCheck task status:")
            self.stdout.write(f"  python manage.py shell")
            self.stdout.write(f"  >>> from celery.result import AsyncResult")
            self.stdout.write(f"  >>> result = AsyncResult('{result.id}')")
            self.stdout.write(f"  >>> result.status")
            self.stdout.write(f"  >>> result.result  # Once complete")
        else:
            self.stdout.write("\nFetching data synchronously...\n")

            try:
                result = task(*task_args)
                self._display_results(result)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"\n[ERROR] Fetch failed: {e}"))
                raise

    def _display_results(self, result):
        """Display fetch results in a nice format"""
        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS("FETCH COMPLETE"))
        self.stdout.write("=" * 70)

        self.stdout.write(f"\nResults:")
        self.stdout.write(f"  - Total fetched: {result.get('total_fetched', 0)}")
        self.stdout.write(f"  - Saved to DB: {result.get('saved', 0)}")
        self.stdout.write(f"  - Skipped (duplicates): {result.get('skipped', 0)}")
        self.stdout.write(f"  - Topics created: {result.get('topics_created', 0)}")

        sources = result.get("sources", {})
        if sources:
            self.stdout.write(f"\nSource breakdown:")
            for source, count in sources.items():
                self.stdout.write(f"  - {source}: {count} posts")

        errors = result.get("errors", [])
        if errors:
            self.stdout.write(f"\n[WARNING] Errors encountered:")
            for error in errors:
                self.stdout.write(self.style.WARNING(f"  - {error}"))

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("Next steps:")
        self.stdout.write("  1. Calculate metrics: python manage.py calculate_metrics")
        self.stdout.write("  2. View in admin: http://localhost:8000/admin/api/post/")
        self.stdout.write("=" * 70)