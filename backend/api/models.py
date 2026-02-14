from django.contrib.auth.models import User
from django.db import models
from django.db.models import Index
from django.utils import timezone


class Topic(models.Model):
    """
    The Trend Registry - stores unique entities/keywords the system has identified.
    E.g., "OpenClaw", "Sonnet 4.6", "Claude 3.5"
    """

    CATEGORY_CHOICES = [
        ("LLM", "Large Language Model"),
        ("FRAMEWORK", "Framework"),
        ("LIBRARY", "Library"),
        ("TOOL", "Tool"),
        ("PHILOSOPHY", "Philosophy"),
        ("PLATFORM", "Platform"),
        ("LANGUAGE", "Programming Language"),
        ("OTHER", "Other"),
    ]

    name = models.CharField(max_length=255, unique=True, db_index=True)
    category = models.CharField(
        max_length=50, choices=CATEGORY_CHOICES, default="OTHER", db_index=True
    )
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional metadata
    description = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(
        default=True, help_text="Whether this topic is actively being tracked"
    )

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Topic"
        verbose_name_plural = "Topics"
        indexes = [
            Index(fields=["name", "category"]),
            Index(fields=["is_active", "created_at"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_category_display()})"


class Entity(models.Model):
    """
    Canonical entities for the knowledge graph.
    Represents the "parent" or "main" concept (e.g., "Claude", "OpenClaw")
    """

    ENTITY_TYPE_CHOICES = [
        ("LLM_FAMILY", "LLM Family"),
        ("FRAMEWORK", "Framework"),
        ("LIBRARY", "Library"),
        ("TOOL", "Tool"),
        ("PLATFORM", "Platform"),
        ("LANGUAGE", "Programming Language"),
        ("CONCEPT", "Concept"),
        ("COMPANY", "Company"),
        ("OTHER", "Other"),
    ]

    canonical_name = models.CharField(
        max_length=255, unique=True, db_index=True, help_text="The canonical/official name"
    )
    entity_type = models.CharField(
        max_length=50, choices=ENTITY_TYPE_CHOICES, default="OTHER", db_index=True
    )
    description = models.TextField(blank=True, null=True)

    # Vector embedding for semantic similarity (stored as JSON for SQLite compatibility)
    embedding = models.JSONField(
        blank=True,
        null=True,
        default=None,
        help_text="Vector embedding for semantic search (list of floats)",
    )

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Entity"
        verbose_name_plural = "Entities"
        indexes = [
            Index(fields=["canonical_name", "entity_type"]),
            Index(fields=["is_active", "created_at"]),
        ]

    def __str__(self):
        return f"{self.canonical_name} ({self.get_entity_type_display()})"


class EntityNode(models.Model):
    """
    Specific versions or variations of an entity.
    E.g., "Claude 4.6", "Sonnet 4.6", "Opus 4.6" all point to parent "Claude"
    """

    parent = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="nodes",
        db_index=True,
        help_text="Parent canonical entity",
    )
    label = models.CharField(
        max_length=255, db_index=True, help_text="Specific version/variation name"
    )

    # Aliases discovered by LLM or clustering (stored as JSON for SQLite compatibility)
    aliases = models.JSONField(
        default=list,
        blank=True,
        help_text="Alternative names/spellings (e.g., ['Opus 4.6', 'Sonnet 4.6'])",
    )

    # Vector embedding for this specific node (stored as JSON for SQLite compatibility)
    embedding = models.JSONField(
        blank=True,
        null=True,
        default=None,
        help_text="Vector embedding for semantic search (list of floats)",
    )

    # Confidence score from LLM resolution
    confidence_score = models.FloatField(
        default=1.0, help_text="Confidence that this node belongs to parent (0.0-1.0)"
    )

    # Track if this was resolved by LLM or clustering
    resolution_method = models.CharField(
        max_length=50,
        choices=[
            ("MANUAL", "Manual"),
            ("VECTOR", "Vector Clustering"),
            ("LLM", "LLM Resolution"),
            ("HYBRID", "Hybrid"),
        ],
        default="MANUAL",
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["parent", "label"]]
        verbose_name = "Entity Node"
        verbose_name_plural = "Entity Nodes"
        indexes = [
            Index(fields=["parent", "label"]),
            Index(fields=["confidence_score"]),
            Index(fields=["resolution_method"]),
        ]

    def __str__(self):
        return f"{self.label} -> {self.parent.canonical_name}"


class TopicEntityLink(models.Model):
    """
    Links Topics to Entities/EntityNodes for dynamic resolution.
    This allows topics to be automatically grouped by semantic similarity.
    """

    topic = models.ForeignKey(
        Topic, on_delete=models.CASCADE, related_name="entity_links", db_index=True
    )

    # Can link to either Entity or EntityNode
    entity = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="topic_links",
        null=True,
        blank=True,
    )
    entity_node = models.ForeignKey(
        EntityNode,
        on_delete=models.CASCADE,
        related_name="topic_links",
        null=True,
        blank=True,
    )

    # Similarity score from vector matching
    similarity_score = models.FloatField(
        default=0.0, help_text="Cosine similarity score (0.0-1.0)"
    )

    # Resolution metadata
    resolution_method = models.CharField(
        max_length=50,
        choices=[
            ("EXACT", "Exact Match"),
            ("VECTOR", "Vector Similarity"),
            ("LLM", "LLM Resolution"),
        ],
        default="EXACT",
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Topic-Entity Link"
        verbose_name_plural = "Topic-Entity Links"
        indexes = [
            Index(fields=["topic", "entity"]),
            Index(fields=["topic", "entity_node"]),
            Index(fields=["similarity_score"]),
        ]

    def __str__(self):
        if self.entity:
            return f"{self.topic.name} -> {self.entity.canonical_name}"
        elif self.entity_node:
            return f"{self.topic.name} -> {self.entity_node.label}"
        return f"{self.topic.name} (unlinked)"


class Post(models.Model):
    """
    The Evidence - stores every item from JSON feeds (Reddit, HN, GitHub, etc.)
    """

    SOURCE_CHOICES = [
        ("HN", "Hacker News"),
        ("REDDIT_LOCALLLAMA", "r/LocalLLaMA"),
        ("REDDIT_MACHINELEARNING", "r/MachineLearning"),
        ("REDDIT_PROGRAMMING", "r/programming"),
        ("GITHUB", "GitHub"),
        ("TWITTER", "Twitter/X"),
        ("OTHER", "Other"),
    ]

    external_id = models.CharField(
        max_length=255, unique=True, db_index=True, help_text="Original ID from source"
    )
    source = models.CharField(max_length=50, choices=SOURCE_CHOICES, db_index=True)
    title = models.TextField()
    url = models.URLField(max_length=500)

    # Engagement metrics
    engagement_score = models.IntegerField(
        default=0, help_text="Score/points from the source platform"
    )
    comment_count = models.IntegerField(default=0)

    # Sentiment analysis
    sentiment_score = models.FloatField(
        default=0.0,
        help_text="Sentiment score from VADER analysis (-1.0 to 1.0)",
    )

    # Timestamps
    published_at = models.DateTimeField(db_index=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    # Optional fields
    author = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField(
        blank=True, null=True, help_text="Full text content if available"
    )

    # Many-to-many relationship with topics through TopicMention
    topics = models.ManyToManyField(Topic, through="TopicMention", related_name="posts")

    class Meta:
        ordering = ["-published_at"]
        verbose_name = "Post"
        verbose_name_plural = "Posts"
        indexes = [
            Index(fields=["source", "published_at"]),
            Index(fields=["engagement_score", "published_at"]),
            Index(fields=["sentiment_score"]),
            Index(fields=["-published_at"]),
        ]

    def __str__(self):
        return f"[{self.source}] {self.title[:50]}"


class TopicMention(models.Model):
    """
    The Relationship - mapping table connecting posts to topics.
    One post can mention multiple topics (e.g., "OpenClaw security vs Claude 4.6")
    """

    topic = models.ForeignKey(
        Topic, on_delete=models.CASCADE, related_name="mentions", db_index=True
    )
    post = models.ForeignKey(
        Post, on_delete=models.CASCADE, related_name="mentions", db_index=True
    )

    # Additional context about the mention
    relevance_score = models.FloatField(
        default=1.0,
        help_text="How relevant is this topic to the post (0.0 to 1.0)",
    )
    is_primary = models.BooleanField(
        default=False, help_text="Is this the primary topic of the post?"
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = [["topic", "post"]]
        verbose_name = "Topic Mention"
        verbose_name_plural = "Topic Mentions"
        indexes = [
            Index(fields=["topic", "post"]),
            Index(fields=["is_primary"]),
            Index(fields=["relevance_score"]),
        ]

    def __str__(self):
        return f"{self.topic.name} in {self.post.title[:30]}"


class TopicDailyMetric(models.Model):
    """
    The Velocity Engine - stores daily snapshots for each topic.
    Crucial for calculating momentum and detecting "exploding" trends.
    """

    topic = models.ForeignKey(
        Topic, on_delete=models.CASCADE, related_name="daily_metrics", db_index=True
    )
    date = models.DateField(db_index=True)

    # Daily aggregated metrics
    total_mentions = models.IntegerField(
        default=0, help_text="Count of posts mentioning this topic on this day"
    )
    total_engagement = models.IntegerField(
        default=0, help_text="Sum of engagement scores for this day"
    )
    avg_sentiment = models.FloatField(
        default=0.0, help_text="Average sentiment score for this day"
    )

    # Velocity/Momentum calculations
    momentum_score = models.FloatField(
        default=0.0,
        help_text="Calculated velocity vs. previous day (change in mentions)",
    )
    engagement_momentum = models.FloatField(
        default=0.0,
        help_text="Change in total engagement vs. previous day",
    )

    # Breakdown by source
    source_breakdown = models.JSONField(
        default=dict,
        blank=True,
        help_text="JSON object with mention counts per source, e.g., {'HN': 5, 'REDDIT_LOCALLLAMA': 3}",
    )

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["topic", "date"]]
        ordering = ["-date", "topic"]
        verbose_name = "Topic Daily Metric"
        verbose_name_plural = "Topic Daily Metrics"
        indexes = [
            Index(fields=["topic", "-date"]),
            Index(fields=["-date", "-momentum_score"]),
            Index(fields=["-momentum_score"]),
            Index(fields=["-engagement_momentum"]),
        ]

    def __str__(self):
        return f"{self.topic.name} - {self.date} (momentum: {self.momentum_score:.2f})"


class Watchlist(models.Model):
    """
    User watchlists for personalized tracking.
    When a topic's momentum hits a threshold, users watching it get alerted.
    """

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="watchlists", db_index=True
    )
    topic = models.ForeignKey(
        Topic, on_delete=models.CASCADE, related_name="watchers", db_index=True
    )

    # Alert preferences
    momentum_threshold = models.FloatField(
        default=2.0, help_text="Alert when momentum score exceeds this value"
    )
    enabled = models.BooleanField(default=True)

    created_at = models.DateTimeField(default=timezone.now)
    last_alerted_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        unique_together = [["user", "topic"]]
        verbose_name = "Watchlist"
        verbose_name_plural = "Watchlists"
        indexes = [
            Index(fields=["user", "enabled"]),
            Index(fields=["topic", "enabled"]),
        ]

    def __str__(self):
        return f"{self.user.username} watching {self.topic.name}"