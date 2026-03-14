"""
Management command for backfilling historical data day-by-day
Fetches data from Jan 1-30, 2026 to build time series and see growth without waiting
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
from django.conf import settings
from datetime import datetime, timedelta
import os
import time

from api.services.historical_fetchers import HistoricalAggregator
from api.models import Post, Topic, TopicMention
from api.tasks import extract_topics_from_post, batch_calculate_sentiment


class Command(BaseCommand):
    help = "Backfill historical data day-by-day to build growth patterns (Jan 1-30, 2026)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--start-date",
            type=str,
            default="2026-01-01",
            help="Start date (YYYY-MM-DD, default: 2026-01-01)",
        )
        parser.add_argument(
            "--end-date",
            type=str,
            default="2026-01-30",
            help="End date (YYYY-MM-DD, default: 2026-01-30)",
        )
        parser.add_argument(
            "--sources",
            type=str,
            default="all",
            help="Comma-separated sources: all,ph,devto,showhn,reddit,github,ih",
        )
        parser.add_argument(
            "--process",
            action="store_true",
            help="Extract topics and calculate sentiment after each day",
        )
        parser.add_argument(
            "--delay",
            type=int,
            default=2,
            help="Delay in seconds between each day's fetch (to respect rate limits)",
        )
        parser.add_argument(
            "--skip-metrics",
            action="store_true",
            help="Skip metric calculation (faster, but you'll need to run calculate_metrics later)",
        )

    def handle(self, *args, **options):
        start_date_str = options["start_date"]
        end_date_str = options["end_date"]
        sources = options["sources"]
        process = options["process"]
        delay = options["delay"]
        skip_metrics = options["skip_metrics"]

        # Parse dates
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        except ValueError as e:
            self.stdout.write(self.style.ERROR(f"Invalid date format: {e}"))
            return

        if start_date > end_date:
            self.stdout.write(self.style.ERROR("Start date must be before end date"))
            return

        total_days = (end_date - start_date).days + 1

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("🚀 TRENDIFY HISTORICAL BACKFILL 🚀"))
        self.stdout.write("=" * 80)
        self.stdout.write(f"\n📅 Date range: {start_date} to {end_date}")
        self.stdout.write(f"📊 Total days: {total_days}")
        self.stdout.write(f"⏱️  Delay between fetches: {delay}s")
        self.stdout.write(f"🔄 Post-processing: {'Enabled' if process else 'Disabled'}")
        self.stdout.write(f"📈 Metrics calculation: {'Skipped (run later)' if skip_metrics else 'Per-day'}")

        # Parse sources
        source_list = [s.strip().lower() for s in sources.split(",")]
        include_all = "all" in source_list
        
        include_ph = include_all or "ph" in source_list or "producthunt" in source_list
        include_devto = include_all or "devto" in source_list
        include_showhn = include_all or "showhn" in source_list or "show" in source_list
        include_reddit = include_all or "reddit" in source_list
        include_github = include_all or "github" in source_list
        include_ih = include_all or "ih" in source_list or "indiehackers" in source_list

        self.stdout.write(f"\n📡 Sources enabled:")
        if include_ph:
            self.stdout.write("  ✓ Product Hunt (launches)")
        if include_devto:
            self.stdout.write("  ✓ Dev.to (articles)")
        if include_showhn:
            self.stdout.write("  ✓ Show HN (product launches)")
        if include_reddit:
            self.stdout.write("  ✓ Reddit (rising posts)")
        if include_github:
            self.stdout.write("  ✓ GitHub (newly created repos)")
        if include_ih:
            self.stdout.write("  ✓ Indie Hackers (indie launches)")

        # Get Product Hunt token
        ph_token = getattr(settings, 'PRODUCT_HUNT_API_TOKEN', None) or os.getenv('PRODUCT_HUNT_API_TOKEN')
        if include_ph and not ph_token:
            self.stdout.write(
                self.style.WARNING(
                    "\n⚠️  Product Hunt token not found. Skipping Product Hunt."
                )
            )
            self.stdout.write("   Set PRODUCT_HUNT_API_TOKEN in environment or settings.py")
            include_ph = False

        # Initialize historical aggregator
        aggregator = HistoricalAggregator(product_hunt_token=ph_token)

        # Track overall stats
        total_saved = 0
        total_skipped = 0
        total_topics_created = 0
        daily_results = []
        start_time = time.time()

        # Iterate through each day
        current_date = start_date
        day_num = 1
        
        while current_date <= end_date:
            self.stdout.write("\n" + "─" * 80)
            self.stdout.write(f"📅 DAY {day_num}/{total_days}: {current_date.strftime('%B %d, %Y')}")
            self.stdout.write("─" * 80)

            try:
                # Fetch data for this specific day
                result = aggregator.fetch_for_date(
                    target_date=current_date,
                    include_product_hunt=include_ph,
                    include_devto=include_devto,
                    include_show_hn=include_showhn,
                    include_reddit_rising=include_reddit,
                    include_github_new=include_github,
                    include_indiehackers=include_ih,
                )

                all_posts = result.get("all_posts", [])
                stats = result.get("stats", {})

                self.stdout.write(f"   Fetched {len(all_posts)} posts")
                
                # Show source breakdown
                sources_breakdown = stats.get("sources", {})
                if sources_breakdown:
                    for source, count in sources_breakdown.items():
                        if count > 0:
                            self.stdout.write(f"     • {source}: {count}")

                # Save to database
                saved_count = 0
                skipped_count = 0
                topics_created = 0
                new_post_ids = []

                for post_data in all_posts:
                    try:
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
                        
                        # Override created_at to match the historical date
                        # This ensures metrics are calculated correctly for that day
                        post.created_at = timezone.make_aware(
                            datetime.combine(current_date, datetime.min.time())
                        )
                        post.save()

                        saved_count += 1
                        new_post_ids.append(post.id)

                        # Create topics from metadata
                        post_topics = post_data.get("topics", [])
                        for topic_name in post_topics:
                            if topic_name and len(topic_name) > 1:
                                topic, created = Topic.objects.get_or_create(
                                    name=topic_name,
                                    defaults={"category": "OTHER"},
                                )
                                if created:
                                    topics_created += 1

                                TopicMention.objects.get_or_create(
                                    topic=topic,
                                    post=post,
                                    defaults={"relevance_score": 1.0, "is_primary": False},
                                )

                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(f"     ❌ Failed to save post: {e}")
                        )
                        continue

                # Process posts if requested
                if process and new_post_ids:
                    self.stdout.write("   🔄 Processing posts...")
                    for post_id in new_post_ids:
                        try:
                            extract_topics_from_post(post_id)
                        except Exception as e:
                            pass  # Silent fail for individual posts

                    try:
                        batch_calculate_sentiment(new_post_ids)
                    except Exception as e:
                        pass  # Silent fail for batch sentiment
                    
                    self.stdout.write("   ✓ Post processing complete")

                self.stdout.write(
                    f"   💾 Saved: {saved_count} | Skipped: {skipped_count} | New topics: {topics_created}"
                )

                total_saved += saved_count
                total_skipped += skipped_count
                total_topics_created += topics_created

                daily_results.append({
                    "date": current_date,
                    "saved": saved_count,
                    "skipped": skipped_count,
                    "topics": topics_created,
                    "sources": sources_breakdown,
                })

                # Calculate metrics for this day (unless skipped)
                if not skip_metrics:
                    self.stdout.write(f"   📊 Calculating metrics...")
                    from django.core.management import call_command
                    try:
                        call_command("calculate_metrics", action="daily", date=current_date.isoformat(), verbosity=0)
                        self.stdout.write("   ✓ Metrics calculated")
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f"   ⚠️  Metrics calculation failed: {e}"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"\n❌ Day {day_num} failed: {e}"))

            # Move to next day
            current_date += timedelta(days=1)
            day_num += 1

            # Rate limiting delay
            if current_date <= end_date and delay > 0:
                self.stdout.write(f"   ⏳ Waiting {delay}s...")
                time.sleep(delay)

        # Final summary
        elapsed_time = time.time() - start_time
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("✨ BACKFILL COMPLETE! ✨"))
        self.stdout.write("=" * 80)
        self.stdout.write(f"\n📊 Summary:")
        self.stdout.write(f"   Total saved: {total_saved} posts")
        self.stdout.write(f"   Total skipped: {total_skipped} duplicates")
        self.stdout.write(f"   Total topics created: {total_topics_created}")
        self.stdout.write(f"   Time elapsed: {elapsed_time:.1f}s")
        
        self.stdout.write("\n\n📈 Daily Breakdown:")
        self.stdout.write("─" * 80)
        for result in daily_results:
            date_str = result['date'].strftime('%b %d')
            saved = result['saved']
            topics = result['topics']
            self.stdout.write(f"   {date_str}: {saved:3d} posts, {topics:2d} topics")
        
        # Show top products/topics discovered
        self.stdout.write("\n\n🔥 Most Active Sources:")
        source_totals = {}
        for result in daily_results:
            for source, count in result.get('sources', {}).items():
                source_totals[source] = source_totals.get(source, 0) + count
        
        for source, total in sorted(source_totals.items(), key=lambda x: x[1], reverse=True):
            self.stdout.write(f"   {source}: {total} total posts")

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("🎯 Next steps:")
        self.stdout.write("=" * 80)
        if skip_metrics:
            self.stdout.write("   1. Calculate metrics: python manage.py calculate_metrics --action=historical \\")
            self.stdout.write(f"                           --start-date={start_date} --end-date={end_date}")
        self.stdout.write("   2. View emerging trends: http://localhost:8000/api/topics/emerging/")
        self.stdout.write("   3. Check growth patterns: http://localhost:8000/admin/api/topicdailymetric/")
        self.stdout.write("   4. View in frontend: Toggle to 'Emerging' mode")
        self.stdout.write("=" * 80 + "\n")
