from django.contrib import admin
from django.db.models import Count, Sum
from django.utils.html import format_html

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


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "category",
        "is_active",
        "created_at",
        "mention_count",
        "latest_momentum",
    ]
    list_filter = ["category", "is_active", "created_at"]
    search_fields = ["name", "description"]
    readonly_fields = ["created_at", "updated_at"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("name", "category", "is_active")}),
        ("Details", {"fields": ("description",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def mention_count(self, obj):
        return obj.posts.count()

    mention_count.short_description = "Total Mentions"

    def latest_momentum(self, obj):
        latest = obj.daily_metrics.order_by("-date").first()
        if latest:
            color = "green" if latest.momentum_score > 0 else "red"
            return format_html(
                '<span style="color: {};">{:.2f}</span>', color, latest.momentum_score
            )
        return "-"

    latest_momentum.short_description = "Latest Momentum"


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = [
        "title_short",
        "source",
        "engagement_score",
        "comment_count",
        "sentiment_indicator",
        "published_at",
    ]
    list_filter = ["source", "published_at", "sentiment_score"]
    search_fields = ["title", "author", "external_id"]
    readonly_fields = ["created_at", "updated_at", "external_id"]
    date_hierarchy = "published_at"
    ordering = ["-published_at"]

    fieldsets = (
        (
            None,
            {
                "fields": (
                    "external_id",
                    "source",
                    "title",
                    "url",
                    "author",
                )
            },
        ),
        (
            "Content",
            {
                "fields": ("content",),
                "classes": ("collapse",),
            },
        ),
        (
            "Metrics",
            {
                "fields": (
                    "engagement_score",
                    "comment_count",
                    "sentiment_score",
                )
            },
        ),
        (
            "Timestamps",
            {
                "fields": (
                    "published_at",
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )

    def title_short(self, obj):
        return obj.title[:75] + "..." if len(obj.title) > 75 else obj.title

    title_short.short_description = "Title"

    def sentiment_indicator(self, obj):
        if obj.sentiment_score > 0.3:
            color = "green"
            icon = "😊"
        elif obj.sentiment_score < -0.3:
            color = "red"
            icon = "😞"
        else:
            color = "gray"
            icon = "😐"
        return format_html(
            '<span style="color: {};">{} {:.2f}</span>',
            color,
            icon,
            obj.sentiment_score,
        )

    sentiment_indicator.short_description = "Sentiment"


@admin.register(TopicMention)
class TopicMentionAdmin(admin.ModelAdmin):
    list_display = [
        "topic",
        "post_title_short",
        "is_primary",
        "relevance_score",
        "created_at",
    ]
    list_filter = ["is_primary", "relevance_score", "created_at"]
    search_fields = ["topic__name", "post__title"]
    readonly_fields = ["created_at"]
    autocomplete_fields = ["topic", "post"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    def post_title_short(self, obj):
        return obj.post.title[:50] + "..." if len(obj.post.title) > 50 else obj.post.title

    post_title_short.short_description = "Post"


@admin.register(TopicDailyMetric)
class TopicDailyMetricAdmin(admin.ModelAdmin):
    list_display = [
        "topic",
        "date",
        "total_mentions",
        "total_engagement",
        "momentum_indicator",
        "engagement_momentum_indicator",
        "avg_sentiment",
    ]
    list_filter = ["date", "topic__category"]
    search_fields = ["topic__name"]
    readonly_fields = ["created_at", "updated_at"]
    autocomplete_fields = ["topic"]
    date_hierarchy = "date"
    ordering = ["-date", "topic"]

    fieldsets = (
        (None, {"fields": ("topic", "date")}),
        (
            "Daily Metrics",
            {
                "fields": (
                    "total_mentions",
                    "total_engagement",
                    "avg_sentiment",
                )
            },
        ),
        (
            "Momentum",
            {
                "fields": (
                    "momentum_score",
                    "engagement_momentum",
                )
            },
        ),
        (
            "Source Breakdown",
            {
                "fields": ("source_breakdown",),
                "classes": ("collapse",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
            },
        ),
    )

    def momentum_indicator(self, obj):
        if obj.momentum_score > 2.0:
            color = "red"
            icon = "🚀"
        elif obj.momentum_score > 0:
            color = "green"
            icon = "📈"
        elif obj.momentum_score < 0:
            color = "orange"
            icon = "📉"
        else:
            color = "gray"
            icon = "➖"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{} {:.2f}</span>',
            color,
            icon,
            obj.momentum_score,
        )

    momentum_indicator.short_description = "Momentum"

    def engagement_momentum_indicator(self, obj):
        if obj.engagement_momentum > 0:
            color = "green"
            icon = "⬆"
        elif obj.engagement_momentum < 0:
            color = "red"
            icon = "⬇"
        else:
            color = "gray"
            icon = "➖"
        return format_html(
            '<span style="color: {};">{} {:.0f}</span>',
            color,
            icon,
            obj.engagement_momentum,
        )

    engagement_momentum_indicator.short_description = "Engagement Δ"


@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = [
        "user",
        "topic",
        "momentum_threshold",
        "enabled",
        "last_alerted_at",
        "created_at",
    ]
    list_filter = ["enabled", "created_at", "last_alerted_at"]
    search_fields = ["user__username", "topic__name"]
    readonly_fields = ["created_at", "last_alerted_at"]
    autocomplete_fields = ["user", "topic"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("user", "topic", "enabled")}),
        (
            "Alert Settings",
            {
                "fields": (
                    "momentum_threshold",
                    "last_alerted_at",
                )
            },
        ),
        ("Timestamps", {"fields": ("created_at",)}),
    )


@admin.register(Entity)
class EntityAdmin(admin.ModelAdmin):
    list_display = [
        "canonical_name",
        "entity_type",
        "is_active",
        "node_count",
        "has_embedding",
        "created_at",
    ]
    list_filter = ["entity_type", "is_active", "created_at"]
    search_fields = ["canonical_name", "description"]
    readonly_fields = ["created_at", "updated_at", "embedding_preview"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("canonical_name", "entity_type", "is_active")}),
        ("Details", {"fields": ("description",)}),
        (
            "Vector Embedding",
            {
                "fields": ("embedding_preview",),
                "classes": ("collapse",),
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def node_count(self, obj):
        return obj.nodes.count()

    node_count.short_description = "Variations"

    def has_embedding(self, obj):
        if obj.embedding:
            return format_html('<span style="color: green;">Yes</span>')
        return format_html('<span style="color: gray;">No</span>')

    has_embedding.short_description = "Embedding"

    def embedding_preview(self, obj):
        if obj.embedding:
            vector_length = len(obj.embedding)
            preview = str(obj.embedding[:5])[:-1] + ", ...]"
            return f"Vector ({vector_length} dimensions): {preview}"
        return "No embedding generated"

    embedding_preview.short_description = "Embedding Preview"


@admin.register(EntityNode)
class EntityNodeAdmin(admin.ModelAdmin):
    list_display = [
        "label",
        "parent",
        "resolution_method",
        "confidence_score",
        "alias_count",
        "has_embedding",
        "created_at",
    ]
    list_filter = ["resolution_method", "confidence_score", "created_at", "parent__entity_type"]
    search_fields = ["label", "parent__canonical_name", "aliases"]
    readonly_fields = ["created_at", "updated_at", "embedding_preview"]
    autocomplete_fields = ["parent"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("parent", "label")}),
        (
            "Resolution",
            {
                "fields": (
                    "resolution_method",
                    "confidence_score",
                    "aliases",
                )
            },
        ),
        (
            "Vector Embedding",
            {
                "fields": ("embedding_preview",),
                "classes": ("collapse",),
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def alias_count(self, obj):
        return len(obj.aliases) if obj.aliases else 0

    alias_count.short_description = "Aliases"

    def has_embedding(self, obj):
        if obj.embedding:
            return format_html('<span style="color: green;">Yes</span>')
        return format_html('<span style="color: gray;">No</span>')

    has_embedding.short_description = "Embedding"

    def embedding_preview(self, obj):
        if obj.embedding:
            vector_length = len(obj.embedding)
            preview = str(obj.embedding[:5])[:-1] + ", ...]"
            return f"Vector ({vector_length} dimensions): {preview}"
        return "No embedding generated"

    embedding_preview.short_description = "Embedding Preview"


@admin.register(TopicEntityLink)
class TopicEntityLinkAdmin(admin.ModelAdmin):
    list_display = [
        "topic",
        "linked_to",
        "resolution_method",
        "similarity_score",
        "created_at",
    ]
    list_filter = ["resolution_method", "similarity_score", "created_at"]
    search_fields = ["topic__name", "entity__canonical_name", "entity_node__label"]
    readonly_fields = ["created_at", "updated_at"]
    autocomplete_fields = ["topic", "entity", "entity_node"]
    date_hierarchy = "created_at"
    ordering = ["-created_at"]

    fieldsets = (
        (None, {"fields": ("topic", "entity", "entity_node")}),
        (
            "Resolution Details",
            {
                "fields": (
                    "resolution_method",
                    "similarity_score",
                )
            },
        ),
        ("Timestamps", {"fields": ("created_at", "updated_at")}),
    )

    def linked_to(self, obj):
        if obj.entity:
            return f"Entity: {obj.entity.canonical_name}"
        elif obj.entity_node:
            return f"Node: {obj.entity_node.label} ({obj.entity_node.parent.canonical_name})"
        return "Unlinked"

    linked_to.short_description = "Linked To"