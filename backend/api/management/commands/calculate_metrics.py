from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from api.tasks import (
    calculate_all_historical_metrics,
    calculate_daily_metrics,
    check_watchlist_alerts,
    cleanup_old_data,
    detect_exploding_trends,
    generate_daily_report,
)


class Command(BaseCommand):
    help = "Calculate daily metrics and perform maintenance tasks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--action",
            type=str,
            default="daily",
            choices=[
                "daily",
                "historical",
                "exploding",
                "alerts",
                "cleanup",
                "report",
            ],
            help="Action to perform (default: daily)",
        )
        parser.add_argument(
            "--date",
            type=str,
            help="Specific date to calculate metrics for (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--start-date",
            type=str,
            help="Start date for historical metrics (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--end-date",
            type=str,
            help="End date for historical metrics (YYYY-MM-DD)",
        )
        parser.add_argument(
            "--topic-id",
            type=int,
            help="Calculate metrics for a specific topic ID",
        )
        parser.add_argument(
            "--momentum-threshold",
            type=float,
            default=2.0,
            help="Momentum threshold for exploding trends (default: 2.0)",
        )
        parser.add_argument(
            "--cleanup-days",
            type=int,
            default=90,
            help="Number of days to keep data during cleanup (default: 90)",
        )

    def handle(self, *args, **options):
        action = options["action"]
        date_str = options.get("date")
        start_date_str = options.get("start_date")
        end_date_str = options.get("end_date")
        topic_id = options.get("topic_id")
        momentum_threshold = options["momentum_threshold"]
        cleanup_days = options["cleanup_days"]

        # Parse dates if provided
        date = self._parse_date(date_str) if date_str else None
        start_date = self._parse_date(start_date_str) if start_date_str else None
        end_date = self._parse_date(end_date_str) if end_date_str else None

        if action == "daily":
            self._handle_daily(date, topic_id)
        elif action == "historical":
            self._handle_historical(start_date, end_date)
        elif action == "exploding":
            self._handle_exploding(momentum_threshold)
        elif action == "alerts":
            self._handle_alerts()
        elif action == "cleanup":
            self._handle_cleanup(cleanup_days)
        elif action == "report":
            self._handle_report()

    def _handle_daily(self, date, topic_id):
        """Calculate daily metrics for a specific date or today."""
        if date is None:
            date = (timezone.now() - timedelta(days=1)).date()

        self.stdout.write(f"Calculating daily metrics for {date}...")

        if topic_id:
            self.stdout.write(f"Filtering for topic ID: {topic_id}")

        result = calculate_daily_metrics(date=date, topic_id=topic_id)

        self.stdout.write(
            self.style.SUCCESS(
                f"[SUCCESS] Calculated metrics for {result} topics on {date}"
            )
        )

    def _handle_historical(self, start_date, end_date):
        """Calculate historical metrics for a date range."""
        if start_date is None:
            start_date = (timezone.now() - timedelta(days=30)).date()

        if end_date is None:
            end_date = timezone.now().date()

        self.stdout.write(
            f"Calculating historical metrics from {start_date} to {end_date}..."
        )

        total_days = (end_date - start_date).days + 1
        self.stdout.write(f"Processing {total_days} days...")

        result = calculate_all_historical_metrics(
            start_date=start_date, end_date=end_date
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"[SUCCESS] Calculated {result} total metrics across {total_days} days"
            )
        )

    def _handle_exploding(self, momentum_threshold):
        """Detect exploding trends."""
        self.stdout.write(
            f"Detecting exploding trends (threshold: {momentum_threshold})..."
        )

        result = detect_exploding_trends(momentum_threshold=momentum_threshold)

        if result:
            self.stdout.write(
                self.style.SUCCESS(f"\n[SUCCESS] Found {len(result)} exploding trends:")
            )
            for item in result:
                self.stdout.write(
                    f"  - {item['topic']}: momentum {item['momentum']:.2f}"
                )
        else:
            self.stdout.write(
                self.style.WARNING("No exploding trends found at this threshold")
            )

    def _handle_alerts(self):
        """Check watchlists and send alerts."""
        self.stdout.write("Checking watchlist alerts...")

        result = check_watchlist_alerts()

        self.stdout.write(
            self.style.SUCCESS(f"[SUCCESS] Sent {result} alert(s) to users")
        )

    def _handle_cleanup(self, cleanup_days):
        """Clean up old data."""
        self.stdout.write(f"Cleaning up data older than {cleanup_days} days...")

        result = cleanup_old_data(days=cleanup_days)

        self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Cleanup complete:"))
        self.stdout.write(f"  - Posts deleted: {result.get('posts', 0)}")
        self.stdout.write(f"  - Metrics deleted: {result.get('metrics', 0)}")

    def _handle_report(self):
        """Generate daily report."""
        self.stdout.write("Generating daily report...\n")

        result = generate_daily_report()

        self.stdout.write(self.style.SUCCESS(f"Daily Report for {result['date']}"))
        self.stdout.write(f"\nOverall Statistics:")
        self.stdout.write(f"  - Total posts: {result['total_posts']}")
        self.stdout.write(f"  - Active topics: {result['total_active_topics']}")

        self.stdout.write(f"\nTop Trending Topics:")
        for i, topic in enumerate(result["top_trending"][:5], 1):
            self.stdout.write(
                f"  {i}. {topic['topic']}: {topic['mentions']} mentions "
                f"(momentum: {topic['momentum']:.2f})"
            )

        self.stdout.write(f"\nTop Engagement Topics:")
        for i, topic in enumerate(result["top_engagement"][:5], 1):
            self.stdout.write(
                f"  {i}. {topic['topic']}: {topic['engagement']} engagement "
                f"({topic['mentions']} mentions)"
            )

    def _parse_date(self, date_str):
        """Parse date string in YYYY-MM-DD format."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise CommandError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")