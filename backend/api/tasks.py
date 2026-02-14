from datetime import datetime, timedelta

from celery import shared_task
from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.db import transaction
from django.db.models import Avg, Count, F, Sum
from django.utils import timezone

from .models import (
    Entity,
    EntityNode,
    Post,
    Topic,
    TopicDailyMetric,
    TopicEntityLink,
    TopicMention,
    Watchlist,
)
from .services.embeddings import (
    get_embedding_service,
    get_hybrid_resolver,
    get_llm_resolver,
)
from .services.fetchers import TrendifyAggregator


@shared_task
def calculate_sentiment_score(post_id):
    """
    Calculate sentiment score for a post using VADER sentiment analysis.
    
    Args:
        post_id: ID of the post to analyze
    
    Returns:
        The calculated sentiment score
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        post = Post.objects.get(id=post_id)
        analyzer = SentimentIntensityAnalyzer()

        # Analyze title and content if available
        text = post.title
        if post.content:
            text += " " + post.content

        scores = analyzer.polarity_scores(text)
        sentiment_score = scores["compound"]

        # Update post with sentiment score
        post.sentiment_score = sentiment_score
        post.save(update_fields=["sentiment_score"])

        return sentiment_score

    except Post.DoesNotExist:
        return None
    except ImportError:
        print("vaderSentiment not installed. Run: pip install vaderSentiment")
        return None


@shared_task
def batch_calculate_sentiment(post_ids):
    """
    Calculate sentiment scores for multiple posts in batch.
    
    Args:
        post_ids: List of post IDs to analyze
    
    Returns:
        Number of posts processed
    """
    processed = 0
    for post_id in post_ids:
        result = calculate_sentiment_score(post_id)
        if result is not None:
            processed += 1

    return processed


@shared_task
def calculate_daily_metrics(date=None, topic_id=None):
    """
    Calculate daily metrics for all topics or a specific topic.
    This is the core task for velocity/momentum tracking.
    
    Args:
        date: Date to calculate metrics for (defaults to yesterday)
        topic_id: Optional specific topic ID to calculate for
    
    Returns:
        Number of metrics calculated
    """
    if date is None:
        # Default to yesterday
        date = (timezone.now() - timedelta(days=1)).date()
    elif isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()

    topics = Topic.objects.filter(is_active=True)
    if topic_id:
        topics = topics.filter(id=topic_id)

    metrics_created = 0

    for topic in topics:
        # Get all mentions for this topic on this date
        mentions = TopicMention.objects.filter(
            topic=topic, post__published_at__date=date
        ).select_related("post")

        if not mentions.exists():
            continue

        # Calculate aggregated metrics
        total_mentions = mentions.count()
        posts = [m.post for m in mentions]

        total_engagement = sum(p.engagement_score for p in posts)
        avg_sentiment = sum(p.sentiment_score for p in posts) / len(posts)

        # Calculate source breakdown
        source_breakdown = {}
        for post in posts:
            source = post.source
            source_breakdown[source] = source_breakdown.get(source, 0) + 1

        # Get previous day's metrics for momentum calculation
        previous_date = date - timedelta(days=1)
        previous_metric = TopicDailyMetric.objects.filter(
            topic=topic, date=previous_date
        ).first()

        # Calculate momentum (change in mentions)
        momentum_score = 0.0
        engagement_momentum = 0.0

        if previous_metric:
            momentum_score = total_mentions - previous_metric.total_mentions
            engagement_momentum = total_engagement - previous_metric.total_engagement
        else:
            # First day, momentum is just the count
            momentum_score = float(total_mentions)
            engagement_momentum = float(total_engagement)

        # Create or update the daily metric
        metric, created = TopicDailyMetric.objects.update_or_create(
            topic=topic,
            date=date,
            defaults={
                "total_mentions": total_mentions,
                "total_engagement": total_engagement,
                "avg_sentiment": avg_sentiment,
                "momentum_score": momentum_score,
                "engagement_momentum": engagement_momentum,
                "source_breakdown": source_breakdown,
            },
        )

        if created:
            metrics_created += 1

    return metrics_created


@shared_task
def calculate_all_historical_metrics(start_date=None, end_date=None):
    """
    Calculate daily metrics for a date range. Useful for backfilling data.
    
    Args:
        start_date: Start date (YYYY-MM-DD string or date object)
        end_date: End date (YYYY-MM-DD string or date object)
    
    Returns:
        Total number of metrics calculated
    """
    if start_date is None:
        # Default to 30 days ago
        start_date = (timezone.now() - timedelta(days=30)).date()
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    if end_date is None:
        end_date = timezone.now().date()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    total_metrics = 0
    current_date = start_date

    while current_date <= end_date:
        metrics = calculate_daily_metrics(date=current_date)
        total_metrics += metrics
        current_date += timedelta(days=1)

    return total_metrics


@shared_task
def detect_exploding_trends(momentum_threshold=2.0):
    """
    Detect "exploding" trends - topics with high momentum scores.
    This powers the "Exploding" tab in the UI.
    
    Args:
        momentum_threshold: Minimum momentum score to be considered "exploding"
    
    Returns:
        List of exploding topic names
    """
    today = timezone.now().date()

    exploding = (
        TopicDailyMetric.objects.filter(date=today, momentum_score__gte=momentum_threshold)
        .select_related("topic")
        .order_by("-momentum_score")
    )

    exploding_topics = [
        {"topic": metric.topic.name, "momentum": metric.momentum_score}
        for metric in exploding
    ]

    return exploding_topics


@shared_task
def check_watchlist_alerts():
    """
    Check all user watchlists and send alerts for topics that have crossed
    their momentum threshold.
    
    Returns:
        Number of alerts sent
    """
    today = timezone.now().date()
    alerts_sent = 0

    # Get all enabled watchlists
    watchlists = Watchlist.objects.filter(enabled=True).select_related("user", "topic")

    for watchlist in watchlists:
        # Get today's metric for this topic
        metric = TopicDailyMetric.objects.filter(
            topic=watchlist.topic, date=today
        ).first()

        if not metric:
            continue

        # Check if momentum exceeds threshold
        if metric.momentum_score >= watchlist.momentum_threshold:
            # Check if we haven't alerted today
            if (
                watchlist.last_alerted_at is None
                or watchlist.last_alerted_at.date() < today
            ):
                # Send alert
                send_watchlist_alert.delay(watchlist.id, metric.id)
                alerts_sent += 1

                # Update last alerted timestamp
                watchlist.last_alerted_at = timezone.now()
                watchlist.save(update_fields=["last_alerted_at"])

    return alerts_sent


@shared_task
def send_watchlist_alert(watchlist_id, metric_id):
    """
    Send an alert email to a user about a trending topic on their watchlist.
    
    Args:
        watchlist_id: ID of the watchlist entry
        metric_id: ID of the TopicDailyMetric that triggered the alert
    """
    try:
        watchlist = Watchlist.objects.select_related("user", "topic").get(
            id=watchlist_id
        )
        metric = TopicDailyMetric.objects.get(id=metric_id)

        subject = f"🚀 Alert: {watchlist.topic.name} is trending!"
        message = f"""
Hello {watchlist.user.username},

The topic "{watchlist.topic.name}" on your Trendify watchlist is trending!

Today's metrics:
- Mentions: {metric.total_mentions}
- Momentum Score: {metric.momentum_score:.2f} (threshold: {watchlist.momentum_threshold})
- Total Engagement: {metric.total_engagement}
- Average Sentiment: {metric.avg_sentiment:.2f}

View more details at: {settings.FRONTEND_URL}/topics/{watchlist.topic.id}

--
Trendify - Track What Matters
        """

        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[watchlist.user.email],
            fail_silently=False,
        )

        return f"Alert sent to {watchlist.user.email}"

    except (Watchlist.DoesNotExist, TopicDailyMetric.DoesNotExist):
        return None


@shared_task
def extract_topics_from_post(post_id, keywords=None):
    """
    Extract topics from a post based on keywords and create TopicMention entries.
    
    Args:
        post_id: ID of the post to process
        keywords: Optional list of keywords to search for. If None, uses all active topics.
    
    Returns:
        Number of topics extracted
    """
    try:
        post = Post.objects.get(id=post_id)
        text = (post.title + " " + (post.content or "")).lower()

        if keywords is None:
            # Use all active topics as keywords
            topics = Topic.objects.filter(is_active=True)
        else:
            topics = Topic.objects.filter(name__in=keywords, is_active=True)

        mentions_created = 0

        for topic in topics:
            topic_name_lower = topic.name.lower()

            # Simple keyword matching (can be enhanced with NLP)
            if topic_name_lower in text:
                # Calculate relevance based on frequency
                count = text.count(topic_name_lower)
                relevance = min(count * 0.25, 1.0)  # Cap at 1.0

                # Check if this is the primary topic (appears in title)
                is_primary = topic_name_lower in post.title.lower()

                # Create mention if it doesn't exist
                mention, created = TopicMention.objects.get_or_create(
                    topic=topic,
                    post=post,
                    defaults={"relevance_score": relevance, "is_primary": is_primary},
                )

                if created:
                    mentions_created += 1

        return mentions_created

    except Post.DoesNotExist:
        return 0


@shared_task
def cleanup_old_data(days=90):
    """
    Clean up old data from the database to manage storage.
    
    Args:
        days: Number of days to keep data (default: 90)
    
    Returns:
        Dictionary with counts of deleted items
    """
    cutoff_date = timezone.now() - timedelta(days=days)

    deleted_counts = {}

    # Delete old posts
    posts_deleted = Post.objects.filter(published_at__lt=cutoff_date).delete()
    deleted_counts["posts"] = posts_deleted[0] if posts_deleted else 0

    # Delete old daily metrics
    metrics_deleted = TopicDailyMetric.objects.filter(date__lt=cutoff_date.date()).delete()
    deleted_counts["metrics"] = metrics_deleted[0] if metrics_deleted else 0

    return deleted_counts


@shared_task
def generate_daily_report():
    """
    Generate a daily summary report of trending topics.
    Can be scheduled to run every day via celery-beat.
    
    Returns:
        Report data as dictionary
    """
    today = timezone.now().date()
    yesterday = today - timedelta(days=1)

    # Get top trending topics
    top_trending = (
        TopicDailyMetric.objects.filter(date=today)
        .select_related("topic")
        .order_by("-momentum_score")[:10]
    )

    # Get topics with most engagement
    top_engagement = (
        TopicDailyMetric.objects.filter(date=today)
        .select_related("topic")
        .order_by("-total_engagement")[:10]
    )

    # Calculate overall statistics
    total_posts = Post.objects.filter(published_at__date=today).count()
    total_topics = Topic.objects.filter(is_active=True).count()

    report = {
        "date": today.isoformat(),
        "total_posts": total_posts,
        "total_active_topics": total_topics,
        "top_trending": [
            {
                "topic": m.topic.name,
                "momentum": m.momentum_score,
                "mentions": m.total_mentions,
            }
            for m in top_trending
        ],
        "top_engagement": [
            {
                "topic": m.topic.name,
                "engagement": m.total_engagement,
                "mentions": m.total_mentions,
            }
            for m in top_engagement
        ],
    }

    return report


# Periodic task examples (configure in django-celery-beat)
@shared_task
def hourly_metric_update():
    """
    Run hourly to update today's metrics with latest data.
    """
    return calculate_daily_metrics(date=timezone.now().date())


@shared_task
def daily_maintenance():
    """
    Daily maintenance task - calculate metrics, check alerts, cleanup.
    """
    yesterday = (timezone.now() - timedelta(days=1)).date()

    # Calculate yesterday's metrics
    metrics = calculate_daily_metrics(date=yesterday)

    # Check for watchlist alerts
    alerts = check_watchlist_alerts()

    # Cleanup old data (older than 90 days)
    cleanup = cleanup_old_data(days=90)

    return {
        "metrics_calculated": metrics,
        "alerts_sent": alerts,
        "cleanup": cleanup,
    }


# =============================================================================
# DATA FETCHING TASKS (From proof of concept script)
# =============================================================================


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 300},
)
def fetch_all_platforms(
    self, include_hn=True, include_reddit=True, include_github=True
):
    """
    Fetch trending posts from all platforms (HN, Reddit, GitHub)
    This is the main scheduled task that runs daily.

    Args:
        include_hn: Whether to fetch from Hacker News
        include_reddit: Whether to fetch from Reddit
        include_github: Whether to fetch from GitHub

    Returns:
        Dictionary with stats about fetched posts
    """
    try:
        aggregator = TrendifyAggregator()

        # Fetch all posts
        result = aggregator.fetch_all(
            include_hn=include_hn,
            include_reddit=include_reddit,
            include_github=include_github,
        )

        all_posts = result["all_posts"]
        stats = result["stats"]

        # Save posts to database
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

                # Queue sentiment analysis for this post
                calculate_sentiment_score.delay(post.id)

                saved_count += 1

            except Exception as e:
                print(f"Error saving post: {e}")
                continue

        # Extract and create topics from all posts
        keywords = aggregator.extract_keywords(all_posts)
        topics_created = 0

        for keyword, count in keywords:
            # Create topic if it doesn't exist
            topic, created = Topic.objects.get_or_create(
                name=keyword,
                defaults={"category": "OTHER", "is_active": True},
            )

            if created:
                topics_created += 1

        # Queue topic extraction for all new posts
        new_post_ids = Post.objects.filter(
            external_id__in=[
                f"{p['source']}:{p['external_id']}" for p in all_posts
            ]
        ).values_list("id", flat=True)

        for post_id in new_post_ids:
            extract_topics_from_post.delay(post_id)

        return {
            "total_fetched": stats["total_posts"],
            "saved": saved_count,
            "skipped": skipped_count,
            "topics_created": topics_created,
            "sources": stats["sources"],
            "errors": stats["errors"],
        }

    except Exception as exc:
        print(f"Error in fetch_all_platforms: {exc}")
        raise


@shared_task
def fetch_hacker_news():
    """
    Fetch only from Hacker News
    Useful for more frequent updates or manual testing
    """
    return fetch_all_platforms(include_hn=True, include_reddit=False, include_github=False)


@shared_task
def fetch_reddit():
    """
    Fetch only from Reddit
    Useful for more frequent updates or manual testing
    """
    return fetch_all_platforms(include_hn=False, include_reddit=True, include_github=False)


@shared_task
def fetch_github():
    """
    Fetch only from GitHub
    Useful for more frequent updates or manual testing
    """
    return fetch_all_platforms(include_hn=False, include_reddit=False, include_github=True)


@shared_task
def full_pipeline_daily():
    """
    Complete daily pipeline:
    1. Fetch data from all platforms
    2. Calculate sentiment scores
    3. Extract topics
    4. Calculate daily metrics
    5. Check watchlist alerts
    6. Generate daily report

    This is the main scheduled task that should run once per day.
    Configure in django-celery-beat to run at a specific time (e.g., 2 AM).
    """
    from celery import chain

    # Build the pipeline as a chain of tasks
    pipeline = chain(
        # Step 1: Fetch all data
        fetch_all_platforms.si(),
        # Step 2: Calculate yesterday's metrics (after data is fetched)
        calculate_daily_metrics.si(date=(timezone.now() - timedelta(days=1)).date()),
        # Step 3: Check watchlist alerts
        check_watchlist_alerts.si(),
        # Step 4: Generate daily report
        generate_daily_report.si(),
    )

    # Execute the pipeline
    result = pipeline.apply_async()

    return {
        "pipeline_started": True,
        "chain_id": result.id,
        "timestamp": timezone.now().isoformat(),
    }


# =============================================================================
# ENTITY RESOLUTION TASKS (Dynamic entity resolution with vectors and LLM)
# =============================================================================


@shared_task
def generate_embeddings_for_topics(topic_ids=None):
    """
    Generate vector embeddings for topics for semantic similarity matching.

    Args:
        topic_ids: Optional list of topic IDs. If None, processes all topics without embeddings.

    Returns:
        Number of topics processed
    """
    try:
        embedding_service = get_embedding_service()

        # Get topics to process
        if topic_ids:
            topics = Topic.objects.filter(id__in=topic_ids)
        else:
            # Process topics that don't have entity links yet
            topics = Topic.objects.filter(is_active=True, entity_links__isnull=True)

        if not topics.exists():
            return 0

        processed = 0

        for topic in topics:
            try:
                # Generate embedding for topic name
                embedding = embedding_service.generate_embedding(topic.name)

                # Store embedding in a related entity or use for immediate matching
                # For now, we'll use it to find similar entities
                processed += 1

            except Exception as e:
                print(f"Error generating embedding for topic {topic.id}: {e}")
                continue

        return processed

    except Exception as e:
        print(f"Error in generate_embeddings_for_topics: {e}")
        return 0


@shared_task
def resolve_topics_to_entities(topic_ids=None, similarity_threshold=0.85):
    """
    Use vector clustering to resolve topics to entities.
    This finds topics that are semantically similar and groups them.

    Args:
        topic_ids: Optional list of topic IDs. If None, processes unlinked topics.
        similarity_threshold: Minimum cosine similarity to consider synonyms (0.0-1.0)

    Returns:
        Dictionary with resolution statistics
    """
    try:
        hybrid_resolver = get_hybrid_resolver()

        # Get topics to process
        if topic_ids:
            topics = Topic.objects.filter(id__in=topic_ids, is_active=True)
        else:
            # Get topics that don't have entity links yet
            topics = Topic.objects.filter(is_active=True, entity_links__isnull=True)

        if not topics.exists():
            return {"topics_processed": 0, "entities_created": 0, "links_created": 0}

        # Extract topic names
        topic_names = [t.name for t in topics]
        topic_map = {t.name: t for t in topics}

        # Resolve using hybrid approach (vector + optional LLM)
        resolution_result = hybrid_resolver.resolve_topics_to_entities(
            topic_names, similarity_threshold=similarity_threshold
        )

        entities_created = 0
        links_created = 0

        # Create entities and links based on resolution
        for canonical_name, synonyms in resolution_result.items():
            # Create or get the canonical entity
            entity, created = Entity.objects.get_or_create(
                canonical_name=canonical_name,
                defaults={
                    "entity_type": "OTHER",
                    "is_active": True,
                },
            )

            if created:
                entities_created += 1

                # Generate and store embedding for the entity
                embedding_service = get_embedding_service()
                entity_embedding = embedding_service.generate_embedding(canonical_name)
                entity.embedding = entity_embedding
                entity.save(update_fields=["embedding"])

            # Create entity nodes for each synonym
            for synonym in synonyms:
                if synonym in topic_map:
                    topic = topic_map[synonym]

                    # Create entity node if this is not the canonical name
                    if synonym != canonical_name:
                        node, _ = EntityNode.objects.get_or_create(
                            parent=entity,
                            label=synonym,
                            defaults={
                                "resolution_method": "VECTOR",
                                "confidence_score": 0.9,
                            },
                        )

                        # Link topic to entity node
                        TopicEntityLink.objects.get_or_create(
                            topic=topic,
                            entity_node=node,
                            defaults={
                                "similarity_score": similarity_threshold,
                                "resolution_method": "VECTOR",
                            },
                        )
                    else:
                        # Link directly to entity for canonical name
                        TopicEntityLink.objects.get_or_create(
                            topic=topic,
                            entity=entity,
                            defaults={
                                "similarity_score": 1.0,
                                "resolution_method": "EXACT",
                            },
                        )

                    links_created += 1

        return {
            "topics_processed": len(topics),
            "entities_created": entities_created,
            "links_created": links_created,
            "resolution_groups": len(resolution_result),
        }

    except Exception as e:
        print(f"Error in resolve_topics_to_entities: {e}")
        return {"topics_processed": 0, "entities_created": 0, "links_created": 0}


@shared_task
def llm_entity_cleanup(max_keywords=50):
    """
    Background LLM task to clean up ambiguous entity mappings.
    Runs periodically (e.g., every 6 hours) to identify synonyms the vector math missed.

    Args:
        max_keywords: Maximum number of keywords to send to LLM

    Returns:
        Number of entities resolved
    """
    try:
        # Check if LLM is available
        llm_resolver = get_llm_resolver()
        if not llm_resolver.is_available():
            print("LLM not available. Install Ollama and run: ollama pull llama3.2:3b")
            return 0

        # Get top unlinked topics by momentum
        today = timezone.now().date()
        unlinked_topics = (
            Topic.objects.filter(is_active=True, entity_links__isnull=True)
            .order_by("-created_at")[:max_keywords]
        )

        if not unlinked_topics.exists():
            return 0

        keyword_names = [t.name for t in unlinked_topics]

        # Use LLM to resolve entities
        resolution_result = llm_resolver.resolve_entities(keyword_names)

        if not resolution_result:
            return 0

        entities_resolved = 0

        # Create entities based on LLM resolution
        for canonical_name, synonyms in resolution_result.items():
            entity, created = Entity.objects.get_or_create(
                canonical_name=canonical_name,
                defaults={
                    "entity_type": "OTHER",
                    "is_active": True,
                },
            )

            # Generate embedding for entity
            if created:
                embedding_service = get_embedding_service()
                entity_embedding = embedding_service.generate_embedding(canonical_name)
                entity.embedding = entity_embedding
                entity.save(update_fields=["embedding"])

            # Link synonyms to entity
            for synonym in synonyms:
                matching_topics = unlinked_topics.filter(name=synonym)

                for topic in matching_topics:
                    # Create entity node
                    if synonym != canonical_name:
                        node, _ = EntityNode.objects.get_or_create(
                            parent=entity,
                            label=synonym,
                            defaults={
                                "resolution_method": "LLM",
                                "confidence_score": 0.95,
                                "aliases": [canonical_name],
                            },
                        )

                        # Link topic to entity node
                        TopicEntityLink.objects.get_or_create(
                            topic=topic,
                            entity_node=node,
                            defaults={
                                "similarity_score": 0.95,
                                "resolution_method": "LLM",
                            },
                        )
                    else:
                        # Link directly to entity
                        TopicEntityLink.objects.get_or_create(
                            topic=topic,
                            entity=entity,
                            defaults={
                                "similarity_score": 1.0,
                                "resolution_method": "LLM",
                            },
                        )

                    entities_resolved += 1

        return entities_resolved

    except Exception as e:
        print(f"Error in llm_entity_cleanup: {e}")
        return 0


@shared_task
def find_similar_entities(entity_id, similarity_threshold=0.90):
    """
    Find entities similar to a given entity using vector similarity.

    Args:
        entity_id: ID of the entity to find similar entities for
        similarity_threshold: Minimum similarity score

    Returns:
        List of similar entity IDs
    """
    try:
        entity = Entity.objects.get(id=entity_id)

        if not entity.embedding:
            return []

        embedding_service = get_embedding_service()

        # Get all other entities with embeddings
        other_entities = Entity.objects.filter(is_active=True).exclude(id=entity_id)

        similar_entities = []

        for other in other_entities:
            if not other.embedding:
                continue

            similarity = embedding_service.cosine_similarity(
                entity.embedding, other.embedding
            )

            if similarity >= similarity_threshold:
                similar_entities.append(
                    {"entity_id": other.id, "name": other.canonical_name, "similarity": similarity}
                )

        # Sort by similarity
        similar_entities.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_entities

    except Entity.DoesNotExist:
        return []
    except Exception as e:
        print(f"Error in find_similar_entities: {e}")
        return []


@shared_task
def calculate_entity_momentum(entity_id, date=None):
    """
    Calculate momentum for an entity by aggregating all its related topics.
    This gives you the "Global Vibe" regardless of naming variations.

    Args:
        entity_id: ID of the entity
        date: Date to calculate for (defaults to yesterday)

    Returns:
        Momentum metrics for the entity
    """
    try:
        if date is None:
            date = (timezone.now() - timedelta(days=1)).date()
        elif isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d").date()

        entity = Entity.objects.get(id=entity_id)

        # Get all topics linked to this entity (directly or via nodes)
        direct_topics = Topic.objects.filter(
            entity_links__entity=entity, is_active=True
        ).distinct()

        node_topics = Topic.objects.filter(
            entity_links__entity_node__parent=entity, is_active=True
        ).distinct()

        all_topics = (direct_topics | node_topics).distinct()

        if not all_topics.exists():
            return {
                "entity": entity.canonical_name,
                "total_mentions": 0,
                "momentum": 0.0,
            }

        # Aggregate metrics across all related topics
        total_mentions = 0
        total_engagement = 0
        momentum_sum = 0

        for topic in all_topics:
            metric = TopicDailyMetric.objects.filter(topic=topic, date=date).first()
            if metric:
                total_mentions += metric.total_mentions
                total_engagement += metric.total_engagement
                momentum_sum += metric.momentum_score

        # Calculate average momentum
        avg_momentum = momentum_sum / len(all_topics) if all_topics else 0

        return {
            "entity": entity.canonical_name,
            "date": date.isoformat(),
            "total_mentions": total_mentions,
            "total_engagement": total_engagement,
            "avg_momentum": avg_momentum,
            "topics_count": len(all_topics),
        }

    except Entity.DoesNotExist:
        return {}
    except Exception as e:
        print(f"Error in calculate_entity_momentum: {e}")
        return {}