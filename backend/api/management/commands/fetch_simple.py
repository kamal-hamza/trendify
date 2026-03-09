"""
Simple management command for fetching data without Celery
Use this for local development when Redis is not available
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from datetime import datetime
import os

from api.models import Post, Topic, TopicMention
from api.services.fetchers import TrendifyAggregator


class Command(BaseCommand):
    help = "Fetch trending data from platforms (no Celery required)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--platform",
            type=str,
            choices=["all", "hn", "reddit", "github"],
            default="all",
            help="Platform to fetch from (default: all)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=30,
            help="Number of posts to fetch per platform (default: 30)",
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["normal", "emerging"],
            default="normal",
            help="Fetch mode: 'normal' for established sources, 'emerging' for new/rising content (default: normal)",
        )

    def handle(self, *args, **options):
        platform = options["platform"]
        limit = options["limit"]
        mode = options["mode"]

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS("TRENDIFY SIMPLE DATA FETCHER"))
        self.stdout.write("=" * 70)
        self.stdout.write(f"\nPlatform: {platform.upper()}")
        self.stdout.write(f"Mode: {mode.upper()}")
        self.stdout.write(f"Limit: {limit} posts per source\n")

        # Determine which platforms to fetch from
        include_hn = platform in ["all", "hn"]
        include_reddit = platform in ["all", "reddit"]
        include_github = platform in ["all", "github"]

        try:
            # Get Product Hunt token from settings or environment
            ph_token = getattr(settings, 'PRODUCT_HUNT_API_TOKEN', None) or os.getenv('PRODUCT_HUNT_API_TOKEN')
            
            # Initialize aggregator
            self.stdout.write("Initializing aggregator...")
            aggregator = TrendifyAggregator(product_hunt_token=ph_token)

            # Fetch data based on mode
            self.stdout.write("Fetching data from platforms...")
            
            if mode == "emerging":
                # Fetch from emerging sources
                if platform != "all":
                    self.stdout.write(
                        self.style.WARNING(
                            "\n[Note] Emerging mode fetches from specific emerging sources. Platform filter may be ignored."
                        )
                    )
                
                result = aggregator.fetch_emerging_only(
                    include_product_hunt=bool(ph_token),
                    include_devto=True,
                    include_show_hn=include_hn or platform == "all",
                    include_reddit_rising=include_reddit or platform == "all",
                    include_github_new=include_github or platform == "all",
                    include_indiehackers=True,
                )
                
                if not ph_token:
                    self.stdout.write(
                        self.style.WARNING(
                            "\n[Warning] Product Hunt API token not found. Set PRODUCT_HUNT_API_TOKEN to fetch from Product Hunt."
                        )
                    )
            else:
                # Normal fetch
                result = aggregator.fetch_all(
                    include_hn=include_hn,
                    include_reddit=include_reddit,
                    include_github=include_github,
                    use_topic_search=True,
                )

            all_posts = result["all_posts"]
            discovered_topics = result.get("discovered_topics", [])
            stats = result["stats"]

            self.stdout.write(
                self.style.SUCCESS(
                    f"✓ Fetched {stats['total_posts']} posts from {len(stats['sources'])} sources"
                )
            )

            # Display source breakdown
            self.stdout.write("\nSource breakdown:")
            for source, count in stats["sources"].items():
                self.stdout.write(f"  - {source}: {count} posts")

            if stats["errors"]:
                self.stdout.write(
                    self.style.WARNING(f"\n⚠ {len(stats['errors'])} errors encountered")
                )
                for error in stats["errors"][:3]:
                    self.stdout.write(f"  - {error}")

            # Save posts to database
            self.stdout.write("\nSaving posts to database...")
            saved_count = 0
            skipped_count = 0

            for post_data in all_posts:
                try:
                    # Create unique external_id with source prefix
                    external_id = f"{post_data['source']}:{post_data['external_id']}"

                    # Check if post already exists
                    if Post.objects.filter(external_id=external_id).exists():
                        skipped_count += 1
                        continue

                    # Create post
                    post = Post.objects.create(
                        external_id=external_id,
                        source=post_data["source"],
                        title=post_data["title"],
                        url=post_data["url"],
                        engagement_score=post_data["score"],
                        comment_count=post_data.get("num_comments", 0),
                        author=post_data.get("author", ""),
                        content=post_data.get("content", ""),
                        published_at=post_data["published_at"],
                    )

                    saved_count += 1

                    # Print progress every 10 posts
                    if saved_count % 10 == 0:
                        self.stdout.write(f"  Saved {saved_count} posts...", ending="\r")

                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f"\n  Error saving post: {e}")
                    )
                    continue

            self.stdout.write(
                self.style.SUCCESS(
                    f"\n✓ Saved {saved_count} new posts, skipped {skipped_count} duplicates"
                )
            )

            # Create topics from discovered platform topics
            self.stdout.write("\nCreating topics...")
            topics_created = 0
            topic_map = {}

            for topic_name in discovered_topics:
                topic, created = Topic.objects.get_or_create(
                    name=topic_name,
                    defaults={
                        "category": "OTHER",
                        "is_active": True,
                        "description": f"Auto-discovered topic from platform data",
                    },
                )
                topic_map[topic_name] = topic

                if created:
                    topics_created += 1

            # Extract keywords as additional topics
            self.stdout.write("Extracting keywords from posts...")
            keywords = aggregator.extract_keywords(all_posts)

            for keyword, count in keywords[:20]:  # Top 20 keywords
                if keyword not in topic_map:
                    topic, created = Topic.objects.get_or_create(
                        name=keyword,
                        defaults={
                            "category": "OTHER",
                            "is_active": True,
                            "description": f"Keyword extracted from {count} posts",
                        },
                    )
                    topic_map[keyword] = topic
                    if created:
                        topics_created += 1

            self.stdout.write(
                self.style.SUCCESS(f"✓ Created {topics_created} new topics")
            )

            # Create TopicMentions for posts with explicit topics
            self.stdout.write("\nLinking posts to topics...")
            mentions_created = 0

            for post_data in all_posts:
                external_id = f"{post_data['source']}:{post_data['external_id']}"

                try:
                    post = Post.objects.get(external_id=external_id)
                    post_topics = post_data.get("topics", [])

                    # Link post to its explicit topics
                    for topic_name in post_topics:
                        if topic_name in topic_map:
                            topic = topic_map[topic_name]
                            mention, created = TopicMention.objects.get_or_create(
                                topic=topic,
                                post=post,
                                defaults={
                                    "relevance_score": 1.0,
                                    "is_primary": len(post_topics) == 1,
                                },
                            )
                            if created:
                                mentions_created += 1

                    # Also link to keyword topics found in title
                    title_lower = post.title.lower()
                    for keyword in topic_map.keys():
                        if (
                            keyword.lower() in title_lower
                            and keyword not in post_topics
                        ):
                            topic = topic_map[keyword]
                            mention, created = TopicMention.objects.get_or_create(
                                topic=topic,
                                post=post,
                                defaults={
                                    "relevance_score": 0.5,
                                    "is_primary": False,
                                },
                            )
                            if created:
                                mentions_created += 1

                except Post.DoesNotExist:
                    continue

            self.stdout.write(
                self.style.SUCCESS(f"✓ Created {mentions_created} topic mentions")
            )

            # Final summary
            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.SUCCESS("FETCH COMPLETE"))
            self.stdout.write("=" * 70)
            self.stdout.write(f"\nSummary:")
            self.stdout.write(f"  • Posts fetched: {stats['total_posts']}")
            self.stdout.write(f"  • Posts saved: {saved_count}")
            self.stdout.write(f"  • Posts skipped: {skipped_count}")
            self.stdout.write(f"  • Topics created: {topics_created}")
            self.stdout.write(f"  • Topic mentions: {mentions_created}")
            self.stdout.write(f"  • Discovered topics: {len(discovered_topics)}")
            self.stdout.write(f"  • Top keywords: {len(keywords)}")

            # Display database totals
            total_posts = Post.objects.count()
            total_topics = Topic.objects.count()
            total_mentions = TopicMention.objects.count()

            self.stdout.write(f"\nDatabase totals:")
            self.stdout.write(f"  • Total posts: {total_posts}")
            self.stdout.write(f"  • Total topics: {total_topics}")
            self.stdout.write(f"  • Total mentions: {total_mentions}")

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write("Next steps:")
            self.stdout.write("  1. Calculate metrics:")
            self.stdout.write("     python manage.py calculate_metrics")
            self.stdout.write("\n  2. View data in admin:")
            self.stdout.write("     http://localhost:8000/admin/api/post/")
            self.stdout.write("\n  3. Test API endpoints:")
            self.stdout.write("     python test_api.py")
            self.stdout.write("\n  4. View trending topics:")
            self.stdout.write(
                "     curl http://localhost:8000/api/topics/trending/?limit=10"
            )
            if mode == "emerging":
                self.stdout.write("\n  5. View emerging topics:")
                self.stdout.write(
                    "     curl http://localhost:8000/api/topics/emerging/?limit=10"
                )
            self.stdout.write("=" * 70 + "\n")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"\n✗ Error: {e}"))
            import traceback

            traceback.print_exc()
            raise