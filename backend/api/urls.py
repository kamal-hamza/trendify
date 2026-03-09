from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TopicViewSet,
    EntityViewSet,
    EntityNodeViewSet,
    PostViewSet,
    TopicDailyMetricViewSet,
    WatchlistViewSet,
    FeedStatsViewSet,
)

# Create router and register viewsets
router = DefaultRouter()
router.register(r'topics', TopicViewSet, basename='topic')
router.register(r'entities', EntityViewSet, basename='entity')
router.register(r'entity-nodes', EntityNodeViewSet, basename='entitynode')
router.register(r'posts', PostViewSet, basename='post')
router.register(r'metrics', TopicDailyMetricViewSet, basename='metric')
router.register(r'watchlist', WatchlistViewSet, basename='watchlist')
router.register(r'stats', FeedStatsViewSet, basename='stats')

urlpatterns = [
    path('', include(router.urls)),
]