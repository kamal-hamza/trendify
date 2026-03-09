from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticatedOrReadOnly, IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count, Sum, Avg, Max, Min, Q, F
from django.utils import timezone
from datetime import timedelta, date
import logging

from .models import (
    Topic,
    Entity,
    EntityNode,
    TopicEntityLink,
    Post,
    TopicMention,
    TopicDailyMetric,
    Watchlist,
)
from .serializers import (
    TopicSerializer,
    TopicDetailSerializer,
    EntitySerializer,
    EntityNodeSerializer,
    TopicEntityLinkSerializer,
    PostSerializer,
    PostListSerializer,
    TopicMentionSerializer,
    TopicDailyMetricSerializer,
    WatchlistSerializer,
    TrendingTopicSerializer,
    FeedStatsSerializer,
)
from .tasks import fetch_all_platforms, calculate_daily_metrics

logger = logging.getLogger(__name__)


class TopicViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Topic model
    Provides CRUD operations and trending analysis
    """

    queryset = Topic.objects.filter(is_active=True)
    serializer_class = TopicSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["category", "is_active"]
    search_fields = ["name", "description"]
    ordering_fields = ["created_at", "updated_at", "name"]
    ordering = ["-created_at"]

    def get_serializer_class(self):
        """Use detailed serializer for retrieve action"""
        if self.action == "retrieve":
            return TopicDetailSerializer
        return TopicSerializer

    def get_queryset(self):
        """Annotate queryset with counts"""
        queryset = super().get_queryset()
        queryset = queryset.annotate(
            mention_count=Count("mentions"),
            recent_posts_count=Count(
                "posts",
                filter=Q(posts__published_at__gte=timezone.now() - timedelta(days=7)),
            ),
        )
        return queryset

    @action(detail=False, methods=["get"])
    def trending(self, request):
        """
        Get trending topics based on momentum scores
        Query params:
        - days: Number of days to analyze (default: 7)
        - limit: Number of results (default: 20)
        - category: Filter by category
        - min_momentum: Minimum momentum score (default: 0)
        """
        try:
            days = int(request.query_params.get("days", 7))
            limit = int(request.query_params.get("limit", 20))
            category = request.query_params.get("category")
            min_momentum = float(request.query_params.get("min_momentum", 0))

            # Get date range
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Build query
            metrics_query = TopicDailyMetric.objects.filter(
                date__gte=start_date, date__lte=end_date, topic__is_active=True
            )

            if category:
                metrics_query = metrics_query.filter(topic__category=category)

            # Aggregate metrics by topic
            trending_data = (
                metrics_query.values(
                    "topic__id",
                    "topic__name",
                    "topic__category",
                )
                .annotate(
                    total_mentions=Sum("total_mentions"),
                    total_engagement=Sum("total_engagement"),
                    avg_sentiment=Avg("avg_sentiment"),
                    momentum_score=Avg("momentum_score"),
                    engagement_momentum=Avg("engagement_momentum"),
                    recent_posts_count=Count("topic__posts"),
                    peak_date=Max("date"),
                )
                .filter(momentum_score__gte=min_momentum)
                .order_by("-momentum_score")[:limit]
            )

            # Enrich with source breakdown
            results = []
            for item in trending_data:
                topic_id = item["topic__id"]

                # Get source breakdown
                source_metrics = (
                    TopicDailyMetric.objects.filter(
                        topic_id=topic_id, date__gte=start_date, date__lte=end_date
                    )
                    .values("source_breakdown")
                    .first()
                )

                results.append(
                    {
                        "id": item["topic__id"],
                        "name": item["topic__name"],
                        "category": item["topic__category"],
                        "category_display": dict(Topic.CATEGORY_CHOICES).get(
                            item["topic__category"], item["topic__category"]
                        ),
                        "total_mentions": item["total_mentions"] or 0,
                        "total_engagement": item["total_engagement"] or 0,
                        "avg_sentiment": round(item["avg_sentiment"] or 0, 2),
                        "momentum_score": round(item["momentum_score"] or 0, 2),
                        "engagement_momentum": round(item["engagement_momentum"] or 0, 2),
                        "recent_posts_count": item["recent_posts_count"],
                        "sources": source_metrics["source_breakdown"]
                        if source_metrics
                        else {},
                        "peak_date": item["peak_date"],
                    }
                )

            serializer = TrendingTopicSerializer(results, many=True)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return Response(
                {"error": "Failed to retrieve trending topics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def emerging(self, request):
        """
        Get emerging/fast-growing topics
        Filters for:
        - Recently discovered topics (created within last 7 days)
        - High growth rate (>50% growth)
        - Topics gaining momentum
        - Excludes generic/common topics
        
        Query params:
        - days: Number of days to analyze (default: 7)
        - limit: Number of results (default: 20)
        - category: Filter by category
        - min_growth: Minimum growth rate (default: 0.5 = 50%)
        - max_age_days: Maximum age of topic (default: 30)
        - min_mentions: Minimum mentions to avoid noise (default: 3)
        """
        try:
            days = int(request.query_params.get("days", 7))
            limit = int(request.query_params.get("limit", 20))
            category = request.query_params.get("category")
            min_growth = float(request.query_params.get("min_growth", 0.5))
            max_age_days = int(request.query_params.get("max_age_days", 30))
            min_mentions = int(request.query_params.get("min_mentions", 3))
            
            # Get date range
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            age_cutoff = timezone.now() - timedelta(days=max_age_days)
            
            # Filter out generic/common topic names
            generic_topics = [
                'ai', 'python', 'javascript', 'c', 'discussion', 'webdev',
                'programming', 'artificial', 'code', 'project', 'software',
                'open', 'technology', 'data', 'other', 'tutorial', 'guide',
                'help', 'question', 'announcement', 'news', 'update',
                'and the', 'the', 'a', 'an', 'for', 'with', 'to', 'of',
                'artificial intelligence', 'machine learning', 'deep learning',
                'web development', 'software engineering', 'computer science',
            ]
            
            # Build query for recently created topics
            topics_query = Topic.objects.filter(
                is_active=True,
                created_at__gte=age_cutoff,
            ).exclude(
                name__iregex=r'^(show hn|launch|ban|cli|open-source|productivity)$'
            )
            
            # Exclude very generic single-word topics
            for generic in generic_topics:
                topics_query = topics_query.exclude(name__iexact=generic)
            
            if category:
                topics_query = topics_query.filter(category=category)
            
            # Get metrics for these topics
            metrics_query = TopicDailyMetric.objects.filter(
                date__gte=start_date,
                date__lte=end_date,
                topic__in=topics_query,
            )
            
            # Aggregate metrics by topic with growth rate
            emerging_data = (
                metrics_query.values(
                    "topic__id",
                    "topic__name",
                    "topic__category",
                    "topic__created_at",
                )
                .annotate(
                    total_mentions=Sum("total_mentions"),
                    total_engagement=Sum("total_engagement"),
                    avg_sentiment=Avg("avg_sentiment"),
                    momentum_score=Avg("momentum_score"),
                    engagement_momentum=Avg("engagement_momentum"),
                    avg_growth_rate=Avg("growth_rate"),
                    max_growth_rate=Max("growth_rate"),
                    recent_posts_count=Count("topic__posts"),
                    peak_date=Max("date"),
                    first_seen=Min("date"),
                )
                .filter(
                    avg_growth_rate__gte=min_growth,
                    total_mentions__gte=min_mentions,
                )
                .order_by("-total_mentions", "-avg_growth_rate", "-momentum_score")[:limit * 5]
            )
            
            # Enrich with source breakdown and calculate emergence score
            results = []
            for item in emerging_data:
                topic_id = item["topic__id"]
                topic_name = item["topic__name"]
                
                # Skip if topic name is too short or too generic
                if len(topic_name) <= 2 or topic_name.lower() in ['and', 'the', 'or', 'but']:
                    continue
                
                # Get source breakdown
                source_metrics = (
                    TopicDailyMetric.objects.filter(
                        topic_id=topic_id,
                        date__gte=start_date,
                        date__lte=end_date,
                    )
                    .values("source_breakdown")
                    .first()
                )
                
                # Calculate age in days
                created_at = item["topic__created_at"]
                age_days = (timezone.now() - created_at).days
                
                # Calculate emergence score (custom scoring)
                # Factors: mentions, growth rate, sentiment, source diversity, source quality
                sources = source_metrics.get("source_breakdown", {}) if source_metrics else {}
                source_count = len(sources)
                
                # Boost score for emerging-focused sources
                source_quality_boost = 0
                if 'PRODUCT_HUNT' in sources:
                    source_quality_boost += 50  # Product Hunt = new products
                if 'HN' in sources and 'Show HN' in topic_name:
                    source_quality_boost += 40  # Show HN = launches
                if 'DEVTO' in sources:
                    source_quality_boost += 20  # Dev.to = developer content
                if 'GITHUB' in sources and item["total_mentions"] <= 10:
                    source_quality_boost += 30  # New GitHub repos
                
                # Penalize if topic name is very generic or short
                name_penalty = 0
                if len(topic_name) <= 4:
                    name_penalty = 20
                
                emergence_score = (
                    item["total_mentions"] * 0.25 +
                    item["avg_growth_rate"] * 100 * 0.20 +
                    (item["avg_sentiment"] + 1) * 50 * 0.15 +
                    source_count * 20 * 0.15 +
                    source_quality_boost * 0.25 -
                    name_penalty
                )
                
                results.append(
                    {
                        "id": item["topic__id"],
                        "name": item["topic__name"],
                        "category": item["topic__category"],
                        "category_display": dict(Topic.CATEGORY_CHOICES).get(
                            item["topic__category"], item["topic__category"]
                        ),
                        "total_mentions": item["total_mentions"] or 0,
                        "total_engagement": item["total_engagement"] or 0,
                        "avg_sentiment": round(item["avg_sentiment"] or 0, 2),
                        "momentum_score": round(item["momentum_score"] or 0, 2),
                        "engagement_momentum": round(item["engagement_momentum"] or 0, 2),
                        "avg_growth_rate": round(item["avg_growth_rate"] or 0, 2),
                        "max_growth_rate": round(item["max_growth_rate"] or 0, 2),
                        "recent_posts_count": item["recent_posts_count"],
                        "sources": source_metrics["source_breakdown"]
                        if source_metrics
                        else {},
                        "peak_date": item["peak_date"],
                        "first_seen": item["first_seen"],
                        "age_days": age_days,
                        "is_new": age_days <= 7,
                        "emergence_score": round(emergence_score, 2),
                    }
                )
            
            # Sort by emergence score and limit
            results = sorted(results, key=lambda x: x["emergence_score"], reverse=True)[:limit]
            
            serializer = TrendingTopicSerializer(results, many=True)
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"Error getting emerging topics: {e}")
            return Response(
                {"error": "Failed to retrieve emerging topics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=True, methods=["get"])
    def timeline(self, request, pk=None):
        """
        Get timeline of metrics for a specific topic
        Query params:
        - days: Number of days (default: 30)
        """
        try:
            topic = self.get_object()
            days = int(request.query_params.get("days", 30))

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            metrics = topic.daily_metrics.filter(
                date__gte=start_date, date__lte=end_date
            ).order_by("date")

            serializer = TopicDailyMetricSerializer(metrics, many=True)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error getting topic timeline: {e}")
            return Response(
                {"error": "Failed to retrieve timeline"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def categories(self, request):
        """Get list of categories with counts"""
        try:
            categories = (
                Topic.objects.filter(is_active=True)
                .values("category")
                .annotate(count=Count("id"))
                .order_by("-count")
            )

            result = [
                {
                    "category": item["category"],
                    "display": dict(Topic.CATEGORY_CHOICES).get(
                        item["category"], item["category"]
                    ),
                    "count": item["count"],
                }
                for item in categories
            ]

            return Response(result)

        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return Response(
                {"error": "Failed to retrieve categories"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class EntityViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Entity model
    Manages canonical entities in the knowledge graph
    """

    queryset = Entity.objects.filter(is_active=True)
    serializer_class = EntitySerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["entity_type", "is_active"]
    search_fields = ["canonical_name", "description"]
    ordering_fields = ["created_at", "canonical_name"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Annotate with node count"""
        queryset = super().get_queryset()
        queryset = queryset.annotate(node_count=Count("nodes"))
        return queryset


class EntityNodeViewSet(viewsets.ModelViewSet):
    """
    ViewSet for EntityNode model
    Manages entity variations and versions
    """

    queryset = EntityNode.objects.all()
    serializer_class = EntityNodeSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["parent", "resolution_method"]
    search_fields = ["label", "aliases"]
    ordering_fields = ["created_at", "confidence_score"]
    ordering = ["-confidence_score"]


class PostViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for Post model (read-only)
    Provides access to aggregated posts from various sources
    """

    queryset = Post.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["source", "published_at"]
    search_fields = ["title", "author", "content"]
    ordering_fields = ["published_at", "engagement_score", "comment_count", "sentiment_score"]
    ordering = ["-published_at"]

    def get_serializer_class(self):
        """Use lightweight serializer for list, detailed for retrieve"""
        if self.action == "list":
            return PostListSerializer
        return PostSerializer

    def get_queryset(self):
        """
        Filter posts based on query parameters
        - days: Last N days (default: 7)
        - source: Filter by source
        - topic: Filter by topic ID
        - min_engagement: Minimum engagement score
        - sentiment: Filter by sentiment (positive/neutral/negative)
        """
        queryset = super().get_queryset()

        # Filter by days
        days = self.request.query_params.get("days")
        if days:
            try:
                since = timezone.now() - timedelta(days=int(days))
                queryset = queryset.filter(published_at__gte=since)
            except ValueError:
                pass

        # Filter by topic
        topic_id = self.request.query_params.get("topic")
        if topic_id:
            queryset = queryset.filter(topics__id=topic_id)

        # Filter by minimum engagement
        min_engagement = self.request.query_params.get("min_engagement")
        if min_engagement:
            try:
                queryset = queryset.filter(engagement_score__gte=int(min_engagement))
            except ValueError:
                pass

        # Filter by sentiment
        sentiment = self.request.query_params.get("sentiment")
        if sentiment == "positive":
            queryset = queryset.filter(sentiment_score__gt=0.05)
        elif sentiment == "negative":
            queryset = queryset.filter(sentiment_score__lt=-0.05)
        elif sentiment == "neutral":
            queryset = queryset.filter(sentiment_score__gte=-0.05, sentiment_score__lte=0.05)

        return queryset.distinct()

    @action(detail=False, methods=["get"])
    def top(self, request):
        """
        Get top posts by engagement
        Query params:
        - limit: Number of results (default: 20)
        - days: Last N days (default: 7)
        - source: Filter by source
        """
        try:
            limit = int(request.query_params.get("limit", 20))
            days = int(request.query_params.get("days", 7))
            source = request.query_params.get("source")

            since = timezone.now() - timedelta(days=days)
            queryset = Post.objects.filter(published_at__gte=since)

            if source:
                queryset = queryset.filter(source=source)

            top_posts = queryset.order_by("-engagement_score")[:limit]
            serializer = PostListSerializer(top_posts, many=True)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error getting top posts: {e}")
            return Response(
                {"error": "Failed to retrieve top posts"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def feed(self, request):
        """
        Get personalized feed combining trending topics and top posts
        Query params:
        - limit: Number of results (default: 50)
        - days: Last N days (default: 7)
        """
        try:
            limit = int(request.query_params.get("limit", 50))
            days = int(request.query_params.get("days", 7))

            since = timezone.now() - timedelta(days=days)

            # Get posts ordered by a combination of engagement and recency
            posts = (
                Post.objects.filter(published_at__gte=since)
                .annotate(
                    # Score that balances engagement and recency
                    feed_score=F("engagement_score")
                    + F("comment_count") * 2
                    + F("sentiment_score") * 10
                )
                .order_by("-feed_score", "-published_at")[:limit]
            )

            serializer = PostListSerializer(posts, many=True)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error generating feed: {e}")
            return Response(
                {"error": "Failed to generate feed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class TopicDailyMetricViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for TopicDailyMetric model (read-only)
    Provides historical metrics for topics
    """

    queryset = TopicDailyMetric.objects.all()
    serializer_class = TopicDailyMetricSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["topic", "date"]
    ordering_fields = ["date", "momentum_score", "total_mentions", "total_engagement"]
    ordering = ["-date"]

    @action(detail=False, methods=["get"])
    def heatmap(self, request):
        """
        Get heatmap data for momentum scores
        Query params:
        - days: Number of days (default: 30)
        - limit: Number of topics (default: 10)
        """
        try:
            days = int(request.query_params.get("days", 30))
            limit = int(request.query_params.get("limit", 10))

            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            # Get top topics by average momentum
            top_topics = (
                TopicDailyMetric.objects.filter(date__gte=start_date, date__lte=end_date)
                .values("topic__id", "topic__name")
                .annotate(avg_momentum=Avg("momentum_score"))
                .order_by("-avg_momentum")[:limit]
            )

            # Get daily data for these topics
            topic_ids = [t["topic__id"] for t in top_topics]
            metrics = TopicDailyMetric.objects.filter(
                topic__id__in=topic_ids, date__gte=start_date, date__lte=end_date
            ).order_by("topic", "date")

            # Format as heatmap data
            result = {}
            for metric in metrics:
                topic_name = metric.topic.name
                if topic_name not in result:
                    result[topic_name] = []
                result[topic_name].append(
                    {
                        "date": metric.date.isoformat(),
                        "momentum": round(metric.momentum_score, 2),
                        "mentions": metric.total_mentions,
                    }
                )

            return Response(result)

        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return Response(
                {"error": "Failed to generate heatmap"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class WatchlistViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Watchlist model
    Manages user watchlists (requires authentication)
    """

    queryset = Watchlist.objects.all()
    serializer_class = WatchlistSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ["topic", "enabled"]
    ordering_fields = ["created_at"]
    ordering = ["-created_at"]

    def get_queryset(self):
        """Filter to current user's watchlists"""
        return Watchlist.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """Set user to current user"""
        serializer.save(user=self.request.user)


class FeedStatsViewSet(viewsets.ViewSet):
    """
    ViewSet for overall feed statistics
    Provides aggregated metrics across all data
    """

    permission_classes = [IsAuthenticatedOrReadOnly]

    @action(detail=False, methods=["get"])
    def overview(self, request):
        """
        Get overview statistics
        Query params:
        - days: Number of days to analyze (default: 7)
        """
        try:
            days = int(request.query_params.get("days", 7))
            since = timezone.now() - timedelta(days=days)

            # Total counts
            total_posts = Post.objects.filter(published_at__gte=since).count()
            total_topics = Topic.objects.filter(is_active=True).count()
            total_active_topics = Topic.objects.filter(
                is_active=True, posts__published_at__gte=since
            ).distinct().count()

            # Sources breakdown
            sources = (
                Post.objects.filter(published_at__gte=since)
                .values("source")
                .annotate(count=Count("id"), total_engagement=Sum("engagement_score"))
                .order_by("-count")
            )

            sources_breakdown = {
                item["source"]: {
                    "count": item["count"],
                    "total_engagement": item["total_engagement"] or 0,
                    "display": dict(Post.SOURCE_CHOICES).get(item["source"], item["source"]),
                }
                for item in sources
            }

            # Date range
            date_range_data = Post.objects.filter(published_at__gte=since).aggregate(
                earliest=Min("published_at"), latest=Max("published_at")
            )

            # Top categories
            top_categories = (
                Topic.objects.filter(is_active=True, posts__published_at__gte=since)
                .values("category")
                .annotate(count=Count("posts"))
                .order_by("-count")[:5]
            )

            categories_data = [
                {
                    "category": item["category"],
                    "display": dict(Topic.CATEGORY_CHOICES).get(
                        item["category"], item["category"]
                    ),
                    "count": item["count"],
                }
                for item in top_categories
            ]

            # Average sentiment
            avg_sentiment = (
                Post.objects.filter(published_at__gte=since).aggregate(
                    avg=Avg("sentiment_score")
                )["avg"]
                or 0
            )

            result = {
                "total_posts": total_posts,
                "total_topics": total_topics,
                "total_active_topics": total_active_topics,
                "sources_breakdown": sources_breakdown,
                "date_range": {
                    "start": date_range_data["earliest"].isoformat()
                    if date_range_data["earliest"]
                    else None,
                    "end": date_range_data["latest"].isoformat()
                    if date_range_data["latest"]
                    else None,
                },
                "top_categories": categories_data,
                "avg_sentiment": round(avg_sentiment, 3),
            }

            serializer = FeedStatsSerializer(result)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error getting feed stats: {e}")
            return Response(
                {"error": "Failed to retrieve statistics"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def sentiment_analysis(self, request):
        """
        Get sentiment analysis breakdown
        Query params:
        - days: Number of days (default: 7)
        - source: Filter by source
        """
        try:
            days = int(request.query_params.get("days", 7))
            source = request.query_params.get("source")
            since = timezone.now() - timedelta(days=days)

            queryset = Post.objects.filter(published_at__gte=since)
            if source:
                queryset = queryset.filter(source=source)

            # Count by sentiment category
            positive_count = queryset.filter(sentiment_score__gt=0.05).count()
            neutral_count = queryset.filter(
                sentiment_score__gte=-0.05, sentiment_score__lte=0.05
            ).count()
            negative_count = queryset.filter(sentiment_score__lt=-0.05).count()

            # Average sentiment
            avg_sentiment = queryset.aggregate(avg=Avg("sentiment_score"))["avg"] or 0

            result = {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
                "average": round(avg_sentiment, 3),
                "total": positive_count + neutral_count + negative_count,
            }

            return Response(result)

        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {e}")
            return Response(
                {"error": "Failed to retrieve sentiment analysis"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def refresh_data(self, request):
        """
        Trigger background data fetching from all sources
        This will queue Celery tasks to fetch data from HN, Reddit, GitHub, etc.
        """
        try:
            # Trigger the fetch task
            task = fetch_all_platforms.delay()
            
            return Response({
                "status": "Task queued",
                "task_id": task.id,
                "message": "Data refresh task has been queued. This may take several minutes."
            })
            
        except Exception as e:
            logger.error(f"Error queuing data refresh: {e}")
            return Response(
                {"error": "Failed to queue data refresh task"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def recalculate_metrics(self, request):
        """
        Trigger recalculation of daily metrics for all topics
        """
        try:
            # Trigger the metrics calculation task
            task = calculate_daily_metrics.delay()
            
            return Response({
                "status": "Task queued",
                "task_id": task.id,
                "message": "Metrics recalculation task has been queued."
            })
            
        except Exception as e:
            logger.error(f"Error queuing metrics recalculation: {e}")
            return Response(
                {"error": "Failed to queue metrics recalculation task"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )