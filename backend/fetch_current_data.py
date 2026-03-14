#!/usr/bin/env python
"""
Fetch current data from all sources for March 8-9, 2026
"""
import os
import sys
import django
from datetime import datetime, timedelta

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.models import Post, Topic, TopicMention
from api.services.fetchers import (
    TrendifyAggregator,
    DevToFetcher,
    GitHubTrendingFetcher,
    LobstersFetcher,
    TAAFTFetcher,
    HackerNewsFetcher,
    ProductHuntFetcher,
)
from django.utils import timezone

def fetch_all_sources():
    """Fetch data from all available sources"""
    print("=" * 80)
    print("FETCHING DATA FROM ALL SOURCES")
    print("=" * 80)
    
    total_saved = 0
    total_skipped = 0
    total_topics_created = 0
    source_stats = {}
    
    # Initialize fetchers
    ph_token = os.getenv('PRODUCT_HUNT_API_TOKEN')
    github_token = os.getenv('GITHUB_TOKEN')
    
    # 1. Dev.to #showdev
    print("\n[1/6] Fetching from Dev.to (#showdev)...")
    try:
        devto_fetcher = DevToFetcher()
        devto_result = devto_fetcher.fetch(days_ago=3, tag="showdev")
        saved, skipped, topics = save_posts(devto_result, "DEVTO")
        total_saved += saved
        total_skipped += skipped
        total_topics_created += topics
        source_stats["Dev.to"] = saved
        print(f"   ✓ Saved {saved} posts, skipped {skipped}, created {topics} topics")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        source_stats["Dev.to"] = 0
    
    # 2. GitHub Trending
    print("\n[2/6] Fetching from GitHub Trending...")
    try:
        github_trending_fetcher = GitHubTrendingFetcher()
        trending_result = github_trending_fetcher.fetch(days_ago=2)
        saved, skipped, topics = save_posts(trending_result, "GITHUB_TRENDING")
        total_saved += saved
        total_skipped += skipped
        total_topics_created += topics
        source_stats["GitHub Trending"] = saved
        print(f"   ✓ Saved {saved} posts, skipped {skipped}, created {topics} topics")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        source_stats["GitHub Trending"] = 0
    
    # 3. Lobste.rs (show tag)
    print("\n[3/6] Fetching from Lobste.rs (show tag)...")
    try:
        lobsters_fetcher = LobstersFetcher()
        lobsters_result = lobsters_fetcher.fetch(tag="show")
        saved, skipped, topics = save_posts(lobsters_result, "LOBSTERS")
        total_saved += saved
        total_skipped += skipped
        total_topics_created += topics
        source_stats["Lobste.rs"] = saved
        print(f"   ✓ Saved {saved} posts, skipped {skipped}, created {topics} topics")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        source_stats["Lobste.rs"] = 0
    
    # 4. TAAFT (AI Tools)
    print("\n[4/6] Fetching from There's An API For That...")
    try:
        taaft_fetcher = TAAFTFetcher()
        taaft_result = taaft_fetcher.fetch(limit=50)
        saved, skipped, topics = save_posts(taaft_result, "TAAFT")
        total_saved += saved
        total_skipped += skipped
        total_topics_created += topics
        source_stats["TAAFT"] = saved
        print(f"   ✓ Saved {saved} posts, skipped {skipped}, created {topics} topics")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        source_stats["TAAFT"] = 0
    
    # 5. Product Hunt (refresh recent)
    print("\n[5/6] Fetching from Product Hunt...")
    try:
        if ph_token:
            ph_fetcher = ProductHuntFetcher(ph_token)
            ph_result = ph_fetcher.fetch(days_ago=3)
            saved, skipped, topics = save_posts(ph_result, "PRODUCT_HUNT")
            total_saved += saved
            total_skipped += skipped
            total_topics_created += topics
            source_stats["Product Hunt"] = saved
            print(f"   ✓ Saved {saved} posts, skipped {skipped}, created {topics} topics")
        else:
            print("   ⚠ Skipped: No PRODUCT_HUNT_API_TOKEN found")
            source_stats["Product Hunt"] = 0
    except Exception as e:
        print(f"   ✗ Error: {e}")
        source_stats["Product Hunt"] = 0
    
    # 6. Hacker News (Show HN)
    print("\n[6/6] Fetching from Hacker News (Show HN)...")
    try:
        hn_fetcher = HackerNewsFetcher()
        show_hn = hn_fetcher.fetch_show_hn(limit=30, hours_ago=72)
        saved, skipped, topics = save_posts_list(show_hn, "HN")
        total_saved += saved
        total_skipped += skipped
        source_stats["Hacker News"] = saved
        print(f"   ✓ Saved {saved} posts, skipped {skipped}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        source_stats["Hacker News"] = 0
    
    # Summary
    print("\n" + "=" * 80)
    print("FETCH COMPLETE")
    print("=" * 80)
    print(f"\nTotal posts saved: {total_saved}")
    print(f"Total posts skipped (duplicates): {total_skipped}")
    print(f"Total topics created: {total_topics_created}")
    print(f"\nBreakdown by source:")
    for source, count in source_stats.items():
        print(f"  - {source}: {count} posts")
    
    # Show date range of new data
    if total_saved > 0:
        latest = Post.objects.order_by('-published_at').first()
        earliest_new = Post.objects.order_by('-created_at').first()
        print(f"\nLatest post date: {latest.published_at.date()}")
        print(f"Data now covers: {Post.objects.order_by('published_at').first().published_at.date()} to {latest.published_at.date()}")
    
    print("\n" + "=" * 80)

def save_posts(result, source):
    """Save posts from a fetcher result"""
    posts = result.get("posts", [])
    topics_data = result.get("topics", [])
    
    saved_count = 0
    skipped_count = 0
    topics_created = 0
    topic_map = {}
    
    # Create topics
    for topic_name in topics_data:
        if topic_name and len(topic_name) > 1:
            topic, created = Topic.objects.get_or_create(
                name=topic_name,
                defaults={"category": "TECH", "is_active": True},
            )
            topic_map[topic_name] = topic
            if created:
                topics_created += 1
    
    # Save posts
    for post_data in posts:
        try:
            external_id = post_data.get("external_id")
            
            # Skip if already exists
            if Post.objects.filter(external_id=external_id).exists():
                skipped_count += 1
                continue
            
            # Create post
            post = Post.objects.create(
                external_id=external_id,
                source=source,
                title=post_data.get("title", ""),
                url=post_data.get("url", ""),
                engagement_score=post_data.get("score", 0),
                comment_count=post_data.get("num_comments", 0),
                author=post_data.get("author", ""),
                content=post_data.get("content", ""),
                published_at=post_data.get("published_at", timezone.now()),
            )
            
            saved_count += 1
            
            # Link topics
            post_topics = post_data.get("topics", [])
            for topic_name in post_topics:
                if topic_name in topic_map:
                    TopicMention.objects.get_or_create(
                        topic=topic_map[topic_name],
                        post=post,
                        defaults={"relevance_score": 1.0, "is_primary": False},
                    )
        
        except Exception as e:
            print(f"      Error saving post: {e}")
            continue
    
    return saved_count, skipped_count, topics_created

def save_posts_list(posts, source):
    """Save posts from a simple list"""
    saved_count = 0
    skipped_count = 0
    
    for post_data in posts:
        try:
            external_id = post_data.get("external_id")
            
            if Post.objects.filter(external_id=external_id).exists():
                skipped_count += 1
                continue
            
            post = Post.objects.create(
                external_id=external_id,
                source=source,
                title=post_data.get("title", ""),
                url=post_data.get("url", ""),
                engagement_score=post_data.get("score", 0),
                comment_count=post_data.get("num_comments", 0),
                author=post_data.get("author", ""),
                content=post_data.get("content", ""),
                published_at=post_data.get("published_at", timezone.now()),
            )
            
            saved_count += 1
            
            # Extract topics from Show HN titles
            post_topics = post_data.get("topics", [])
            for topic_name in post_topics:
                if topic_name:
                    topic, _ = Topic.objects.get_or_create(
                        name=topic_name,
                        defaults={"category": "TECH", "is_active": True},
                    )
                    TopicMention.objects.get_or_create(
                        topic=topic,
                        post=post,
                        defaults={"relevance_score": 1.0, "is_primary": True},
                    )
        
        except Exception as e:
            print(f"      Error saving post: {e}")
            continue
    
    return saved_count, skipped_count, 0

if __name__ == "__main__":
    fetch_all_sources()