import json
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from api.models import Post, Topic, TopicMention
from api.tasks import calculate_sentiment_score, extract_topics_from_post


class Command(BaseCommand):
    help = "Import posts from JSON file and process them for trend tracking"

    def add_arguments(self, parser):
        parser.add_argument(
            "json_file",
            type=str,
            help="Path to JSON file containing posts data",
        )
        parser.add_argument(
            "--source",
            type=str,
            default="HN",
            help="Source platform (HN, REDDIT_LOCALLLAMA, etc.)",
        )
        parser.add_argument(
            "--calculate-sentiment",
            action="store_true",
            help="Calculate sentiment scores after import",
        )
        parser.add_argument(
            "--extract-topics",
            action="store_true",
            help="Extract topics from posts after import",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of posts to process in each batch (default: 100)",
        )
        parser.add_argument(
            "--skip-existing",
            action="store_true",
            help="Skip posts that already exist in the database",
        )

    def handle(self, *args, **options):
        json_file = options["json_file"]
        source = options["source"]
        calculate_sentiment = options["calculate_sentiment"]
        extract_topics = options["extract_topics"]
        batch_size = options["batch_size"]
        skip_existing = options["skip_existing"]

        # Validate file exists
        file_path = Path(json_file)
        if not file_path.exists():
            raise CommandError(f"File not found: {json_file}")

        self.stdout.write(f"Loading data from {json_file}...")

        # Load JSON data
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise CommandError(f"Invalid JSON file: {e}")

        # Handle both list and dict with "posts" key
        if isinstance(data, dict):
            if "posts" in data:
                posts_data = data["posts"]
            elif "items" in data:
                posts_data = data["items"]
            else:
                raise CommandError(
                    "JSON must be a list or dict with 'posts' or 'items' key"
                )
        elif isinstance(data, list):
            posts_data = data
        else:
            raise CommandError("JSON must be a list or dict")

        self.stdout.write(f"Found {len(posts_data)} posts to import")

        # Import posts in batches
        imported = 0
        skipped = 0
        errors = 0
        post_ids = []

        for i in range(0, len(posts_data), batch_size):
            batch = posts_data[i : i + batch_size]

            with transaction.atomic():
                for post_data in batch:
                    try:
                        post_id = self._import_post(post_data, source, skip_existing)
                        if post_id:
                            imported += 1
                            post_ids.append(post_id)
                        else:
                            skipped += 1
                    except Exception as e:
                        errors += 1
                        self.stderr.write(
                            self.style.ERROR(f"Error importing post: {e}")
                        )

            # Progress update
            self.stdout.write(
                f"Progress: {min(i + batch_size, len(posts_data))}/{len(posts_data)} "
                f"(imported: {imported}, skipped: {skipped}, errors: {errors})"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"\nImport complete: {imported} imported, {skipped} skipped, {errors} errors"
            )
        )

        # Optional post-processing
        if calculate_sentiment and post_ids:
            self.stdout.write("\nCalculating sentiment scores...")
            for post_id in post_ids:
                calculate_sentiment_score(post_id)
            self.stdout.write(self.style.SUCCESS("Sentiment calculation complete"))

        if extract_topics and post_ids:
            self.stdout.write("\nExtracting topics from posts...")
            for post_id in post_ids:
                extract_topics_from_post(post_id)
            self.stdout.write(self.style.SUCCESS("Topic extraction complete"))

    def _import_post(self, post_data, source, skip_existing):
        """Import a single post from JSON data."""
        # Extract required fields with fallbacks
        external_id = post_data.get("id") or post_data.get("external_id")
        if not external_id:
            raise ValueError("Post missing 'id' or 'external_id'")

        # Make external_id unique per source
        external_id = f"{source}:{external_id}"

        # Check if already exists
        if skip_existing and Post.objects.filter(external_id=external_id).exists():
            return None

        title = post_data.get("title", "")
        url = post_data.get("url") or post_data.get("link", "")

        if not url:
            # Generate a fallback URL if not provided
            url = f"https://example.com/post/{external_id}"

        # Parse timestamp
        published_at = self._parse_timestamp(
            post_data.get("created_at")
            or post_data.get("timestamp")
            or post_data.get("published_at")
        )

        # Extract metrics
        engagement_score = int(post_data.get("score", 0) or 0)
        comment_count = int(post_data.get("num_comments", 0) or post_data.get("comments", 0) or 0)

        # Optional fields
        author = post_data.get("author") or post_data.get("by")
        content = post_data.get("text") or post_data.get("content") or post_data.get("selftext")
        sentiment_score = float(post_data.get("sentiment_score", 0.0))

        # Create or update post
        post, created = Post.objects.update_or_create(
            external_id=external_id,
            defaults={
                "source": source,
                "title": title,
                "url": url,
                "engagement_score": engagement_score,
                "comment_count": comment_count,
                "sentiment_score": sentiment_score,
                "published_at": published_at,
                "author": author,
                "content": content,
            },
        )

        return post.id if created else None

    def _parse_timestamp(self, timestamp):
        """Parse various timestamp formats."""
        if not timestamp:
            return timezone.now()

        # Handle Unix timestamp (int or float)
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        # Handle string timestamps
        if isinstance(timestamp, str):
            # Try ISO format first
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass

            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return timezone.make_aware(dt)
                except ValueError:
                    continue

        # Fallback to now
        return timezone.now()