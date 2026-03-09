"""
Management command for fetching emerging/early-stage content
Focuses on new product launches, Show HN, rising Reddit posts, etc.
"""

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.conf import settings
import os

from api.services.fetchers import TrendifyAggregator
from api.models import Post, Topic, TopicMention
from api.tasks import extract_topics_from_post, batch_calculate_sentiment


class Command(BaseCommand):
    help = "Fetch emerging/early-stage content from Product Hunt, Show HN, Dev.to, etc."

    def add_arguments(self, parser):
        parser.add_argument(
            "--sources",
            type=str,
            default="all",
            help="Comma-separated sources: all,ph,devto,showhn,reddit,github (default: all)",
        )
        parser.add_argument(
            "--process",
            action="store_true",
            help="Also extract topics and calculate sentiment after fetching",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch data but don't save to database",
        )

    def handle(self, *args, **options):
        sources = options["sources"]
        process = options["process"]
        dry_run = options["dry_run"]

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("TRENDIFY EMERGING CONTENT FETCHER")
        self.stdout.write("=" * 70)
        
        if dry_run:
            self.stdout.write(self.style.WARNING("\n[DRY RUN MODE - No data will be saved]\n"))

        # Parse sources
        source_list = [s.strip().lower() for s in sources.split(",")]
        include_all = "all" in source_list
        
        include_ph = include_all or "ph" in source_list or "producthunt" in source_list
        include_devto = include_all or "devto" in source_list or "dev.to" in source_list
        include_showhn = include_all or "showhn" in source_list or "show" in source_list
        include_reddit = include_all or "reddit" in source_list
        include_github = include_all or "github" in source_list
        include_ih = include_all or "ih" in source_list or "indiehackers" in source_list

        self.stdout.write(f"\nSources to fetch:")
        if include_ph:
            self.stdout.write("  ✓ Product Hunt")
        if include_devto:
            self.stdout.write("  ✓ Dev.to")
        if include_showhn:
            self.stdout.write("  ✓ Show HN / Launches")
        if include_reddit:
            self.stdout.write("  ✓ Reddit (Rising)")
        if include_github:
            self.stdout.write("  ✓ GitHub (Recently Created)")
        if include_ih:
            self.stdout.write("  ✓ Indie Hackers")

        # Get Product Hunt token from environment
        ph_token = getattr(settings, 'PRODUCT_HUNT_API_TOKEN', None) or os.getenv('PRODUCT_HUNT_API_TOKEN')
        
        if include_ph and not ph_token:
            self.stdout.write(
                self.style.WARNING(
                    "\n[WARNING] Product Hunt API token not found. Skipping Product Hunt."
                )
            )
            self.stdout.write("  Set PRODUCT_HUNT_API_TOKEN in environment or settings.py")
            include_ph = False

        # Initialize aggregator
        self.stdout.write("\n" + "-" * 70)
        self.stdout.write("Fetching data...")
        self.stdout.write("-" * 70 + "\n")

        aggregator = TrendifyAggregator(product_hunt_token=ph_token)

        try:
            result = aggregator.fetch_emerging_only(
                include_product_hunt=include_ph,
                include_devto=include_devto,
                include_show_hn=include_showhn,
                include_reddit_rising=include_reddit,
                include_github_new=include_github,
                include_indiehackers=include_ih,
            )

            all_posts = result.get("all_posts", [])
            discovered_topics = result.get("discovered_topics", [])
            stats = result.get("stats", {})

            self.stdout.write(self.style.SUCCESS("\n[SUCCESS] Fetch complete!"))
            self._display_stats(stats)

            if dry_run:
                self.stdout.write(
                    self.style.WARNING(
                        "\n[DRY RUN] Skipping database save. Fetched data preview:"
                    )
                )
                self._display_preview(all_posts[:5])
                return

            # Save to database
            self.stdout.write("\n" + "-" * 70)
            self.stdout.write("Saving to database...")
            self.stdout.write("-" * 70 + "\n")

            saved_count = 0
            skipped_count = 0
            topics_created = 0

            for post_data in all_posts:
                try:
                    # Check if post already exists
                    external_id = post_data.get("external_id")
                    if Post.objects.filter(external_id=external_id).exists():
                        skipped_count += 1
                        continue

                    # Create post
                    post = Post.objects.create(
                        external_id=external_id,
                        source=post_data.get("source"),
                        title=post_data.get("title"),
                        url=post_data.get("url"),
                        engagement_score=post_data.get("score", 0),
                        comment_count=post_data.get("num_comments", 0),
                        author=post_data.get("author", ""),
                        content=post_data.get("content", ""),
                        published_at=post_data.get("published_at"),
                    )

                    saved_count += 1

                    # Create topics from metadata if available
                    post_topics = post_data.get("topics", [])
                    for topic_name in post_topics:
                        if topic_name and len(topic_name) > 1:
                            topic, created = Topic.objects.get_or_create(
                                name=topic_name,
                                defaults={"category": "OTHER"},
                            )
                            if created:
                                topics_created += 1

                            # Create mention
                            TopicMention.objects.get_or_create(
                                topic=topic,
                                post=post,
                                defaults={"relevance_score": 1.0, "is_primary": False},
                            )

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"  [ERROR] Failed to save post: {e}")
                    )
                    continue

            self.stdout.write(self.style.SUCCESS(f"\n✓ Saved {saved_count} new posts"))
            self.stdout.write(f"✓ Skipped {skipped_count} duplicates")
            self.stdout.write(f"✓ Created {topics_created} new topics")

            # Process posts if requested
            if process and saved_count > 0:
                self.stdout.write("\n" + "-" * 70)
                self.stdout.write("Processing posts...")
                self.stdout.write("-" * 70 + "\n")

                # Extract topics from content
                self.stdout.write("Extracting topics from content...")
                new_post_ids = [
                    p.id
                    for p in Post.objects.filter(
                        external_id__in=[
                            pd.get("external_id") for pd in all_posts[-saved_count:]
                        ]
                    )
                ]

                for post_id in new_post_ids:
                    try:
                        extract_topics_from_post(post_id)
                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"  [ERROR] Topic extraction failed: {e}")
                        )

                self.stdout.write(self.style.SUCCESS("✓ Topic extraction complete"))

                # Calculate sentiment
                self.stdout.write("Calculating sentiment scores...")
                try:
                    batch_calculate_sentiment(new_post_ids)
                    self.stdout.write(self.style.SUCCESS("✓ Sentiment calculation complete"))
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"  [ERROR] Sentiment calculation failed: {e}")
                    )

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write("FETCH COMPLETE")
            self.stdout.write("=" * 70)
            self.stdout.write("\nNext steps:")
            self.stdout.write("  1. Calculate metrics: python manage.py calculate_metrics")
            self.stdout.write("  2. View emerging trends: /api/topics/emerging/")
            self.stdout.write("  3. View in admin: http://localhost:8000/admin/api/post/")
            self.stdout.write("=" * 70)

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n[ERROR] Fetch failed: {e}"))
            raise

    def _display_stats(self, stats):
        """Display fetch statistics"""
        self.stdout.write("\nFetch Statistics:")
        self.stdout.write(f"  Total posts: {stats.get('total_posts', 0)}")
        self.stdout.write(f"  Discovered topics: {stats.get('discovered_topics', 0)}")

        sources = stats.get("sources", {})
        if sources:
            self.stdout.write("\n  Source breakdown:")
            for source, count in sources.items():
                self.stdout.write(f"    - {source}: {count}")

        errors = stats.get("errors", [])
        if errors:
            self.stdout.write(self.style.WARNING("\n  Errors:"))
            for error in errors:
                self.stdout.write(self.style.WARNING(f"    - {error}"))

    def _display_preview(self, posts):
        """Display preview of fetched posts"""
        self.stdout.write("\nSample posts fetched:")
        for i, post in enumerate(posts, 1):
            self.stdout.write(f"\n  {i}. [{post.get('source')}] {post.get('title')[:60]}...")
            self.stdout.write(f"     Score: {post.get('score', 0)} | URL: {post.get('url', 'N/A')[:50]}")