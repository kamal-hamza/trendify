from rest_framework import serializers
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


class TopicSerializer(serializers.ModelSerializer):
    """Serializer for Topic model"""

    mention_count = serializers.IntegerField(read_only=True, required=False)
    recent_posts_count = serializers.IntegerField(read_only=True, required=False)

    class Meta:
        model = Topic
        fields = [
            "id",
            "name",
            "category",
            "description",
            "is_active",
            "created_at",
            "updated_at",
            "mention_count",
            "recent_posts_count",
        ]
        read_only_fields = ["created_at", "updated_at"]


class EntityNodeSerializer(serializers.ModelSerializer):
    """Serializer for EntityNode model"""

    class Meta:
        model = EntityNode
        fields = [
            "id",
            "label",
            "aliases",
            "confidence_score",
            "resolution_method",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]


class EntitySerializer(serializers.ModelSerializer):
    """Serializer for Entity model"""

    nodes = EntityNodeSerializer(many=True, read_only=True)
    node_count = serializers.IntegerField(read_only=True, required=False)

    class Meta:
        model = Entity
        fields = [
            "id",
            "canonical_name",
            "entity_type",
            "description",
            "is_active",
            "created_at",
            "updated_at",
            "nodes",
            "node_count",
        ]
        read_only_fields = ["created_at", "updated_at"]


class TopicEntityLinkSerializer(serializers.ModelSerializer):
    """Serializer for TopicEntityLink model"""

    topic_name = serializers.CharField(source="topic.name", read_only=True)
    entity_name = serializers.CharField(source="entity.canonical_name", read_only=True, allow_null=True)
    entity_node_label = serializers.CharField(source="entity_node.label", read_only=True, allow_null=True)

    class Meta:
        model = TopicEntityLink
        fields = [
            "id",
            "topic",
            "topic_name",
            "entity",
            "entity_name",
            "entity_node",
            "entity_node_label",
            "similarity_score",
            "resolution_method",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]


class TopicMentionSerializer(serializers.ModelSerializer):
    """Serializer for TopicMention model"""

    topic_name = serializers.CharField(source="topic.name", read_only=True)
    topic_category = serializers.CharField(source="topic.get_category_display", read_only=True)

    class Meta:
        model = TopicMention
        fields = [
            "id",
            "topic",
            "topic_name",
            "topic_category",
            "relevance_score",
            "is_primary",
            "created_at",
        ]
        read_only_fields = ["created_at"]


class PostSerializer(serializers.ModelSerializer):
    """Serializer for Post model"""

    mentions = TopicMentionSerializer(many=True, read_only=True)
    source_display = serializers.CharField(source="get_source_display", read_only=True)
    topic_names = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            "id",
            "external_id",
            "source",
            "source_display",
            "title",
            "url",
            "engagement_score",
            "comment_count",
            "sentiment_score",
            "published_at",
            "created_at",
            "updated_at",
            "author",
            "content",
            "mentions",
            "topic_names",
        ]
        read_only_fields = ["created_at", "updated_at"]

    def get_topic_names(self, obj):
        """Get list of topic names mentioned in this post"""
        return list(obj.topics.values_list("name", flat=True))


class PostListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing posts"""

    source_display = serializers.CharField(source="get_source_display", read_only=True)
    primary_topic = serializers.SerializerMethodField()
    topic_names = serializers.SerializerMethodField()
    velocity_metrics = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = [
            "id",
            "external_id",
            "source",
            "source_display",
            "title",
            "url",
            "engagement_score",
            "comment_count",
            "sentiment_score",
            "published_at",
            "author",
            "primary_topic",
            "topic_names",
            "velocity_metrics",
        ]

    def get_primary_topic(self, obj):
        """Get the primary topic name if exists"""
        primary_mention = obj.mentions.filter(is_primary=True).first()
        return primary_mention.topic.name if primary_mention else None

    def get_topic_names(self, obj):
        """Get list of all topic names (native tags) for this post"""
        return list(obj.topics.values_list("name", flat=True))

    def get_velocity_metrics(self, obj):
        """Get velocity metrics from primary topic"""
        # Try to get primary mention first, fall back to any mention
        primary_mention = obj.mentions.filter(is_primary=True).first()
        if not primary_mention:
            primary_mention = obj.mentions.first()
        
        if not primary_mention:
            return None
        
        topic = primary_mention.topic
        
        # Get the most recent metric for this topic
        latest_metric = topic.daily_metrics.order_by('-date').first()
        
        if not latest_metric:
            return None
        
        # Get last 7 days of metrics for average calculations
        recent_metrics = list(topic.daily_metrics.order_by('-date')[:7])
        
        if not recent_metrics:
            return None
        
        # Calculate average growth rate over recent period
        avg_growth = sum(m.growth_rate for m in recent_metrics) / len(recent_metrics)
        max_growth = max(m.growth_rate for m in recent_metrics)
        avg_momentum = sum(m.momentum_score for m in recent_metrics) / len(recent_metrics)
        
        return {
            'topic_id': topic.id,
            'topic_name': topic.name,
            'topic_category': topic.category,
            'momentum_score': float(latest_metric.momentum_score),
            'growth_rate': float(latest_metric.growth_rate),
            'avg_growth_rate': float(avg_growth),
            'max_growth_rate': float(max_growth),
            'avg_momentum': float(avg_momentum),
            'total_mentions': latest_metric.total_mentions,
            'latest_date': latest_metric.date.isoformat(),
        }


class TopicDailyMetricSerializer(serializers.ModelSerializer):
    """Serializer for TopicDailyMetric model"""

    topic_name = serializers.CharField(source="topic.name", read_only=True)
    topic_category = serializers.CharField(source="topic.get_category_display", read_only=True)

    class Meta:
        model = TopicDailyMetric
        fields = [
            "id",
            "topic",
            "topic_name",
            "topic_category",
            "date",
            "total_mentions",
            "total_engagement",
            "avg_sentiment",
            "momentum_score",
            "engagement_momentum",
            "source_breakdown",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]


class WatchlistSerializer(serializers.ModelSerializer):
    """Serializer for Watchlist model"""

    topic_name = serializers.CharField(source="topic.name", read_only=True)
    topic_category = serializers.CharField(source="topic.get_category_display", read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = Watchlist
        fields = [
            "id",
            "user",
            "username",
            "topic",
            "topic_name",
            "topic_category",
            "momentum_threshold",
            "enabled",
            "created_at",
            "last_alerted_at",
        ]
        read_only_fields = ["created_at", "last_alerted_at", "user"]


class TopicDetailSerializer(TopicSerializer):
    """Detailed serializer for Topic with related data"""

    recent_posts = serializers.SerializerMethodField()
    daily_metrics = serializers.SerializerMethodField()
    entity_links = TopicEntityLinkSerializer(many=True, read_only=True)

    class Meta(TopicSerializer.Meta):
        fields = TopicSerializer.Meta.fields + [
            "recent_posts",
            "daily_metrics",
            "entity_links",
        ]

    def get_recent_posts(self, obj):
        """Get recent posts mentioning this topic"""
        recent_posts = Post.objects.filter(topics=obj).order_by("-published_at")[:10]
        return PostListSerializer(recent_posts, many=True).data

    def get_daily_metrics(self, obj):
        """Get last 7 days of metrics"""
        metrics = obj.daily_metrics.all()[:7]
        return TopicDailyMetricSerializer(metrics, many=True).data


class TrendingTopicSerializer(serializers.Serializer):
    """Serializer for trending topics with calculated metrics"""

    id = serializers.IntegerField()
    name = serializers.CharField()
    category = serializers.CharField()
    category_display = serializers.CharField()
    total_mentions = serializers.IntegerField()
    total_engagement = serializers.IntegerField()
    avg_sentiment = serializers.FloatField()
    momentum_score = serializers.FloatField()
    engagement_momentum = serializers.FloatField()
    recent_posts_count = serializers.IntegerField()
    sources = serializers.JSONField()
    peak_date = serializers.DateField(allow_null=True)
    
    # Emerging topic fields (optional - only present for emerging endpoint)
    avg_growth_rate = serializers.FloatField(required=False)
    max_growth_rate = serializers.FloatField(required=False)
    first_seen = serializers.DateField(required=False, allow_null=True)
    age_days = serializers.IntegerField(required=False)
    is_new = serializers.BooleanField(required=False)
    emergence_score = serializers.FloatField(required=False)


class FeedStatsSerializer(serializers.Serializer):
    """Serializer for overall feed statistics"""

    total_posts = serializers.IntegerField()
    total_topics = serializers.IntegerField()
    total_active_topics = serializers.IntegerField()
    sources_breakdown = serializers.JSONField()
    date_range = serializers.JSONField()
    top_categories = serializers.JSONField()
    avg_sentiment = serializers.FloatField()