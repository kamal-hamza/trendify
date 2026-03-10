"""
Historical data fetchers with date range support
Wrapper around existing fetchers to add date filtering for backfill operations
"""

from datetime import datetime, timedelta, date as date_type
from typing import Any, Dict, List
from django.utils import timezone

from .fetchers import (
    HackerNewsFetcher,
    RedditFetcher,
    GitHubFetcher,
    ProductHuntFetcher,
    DevToFetcher,
    IndieHackersFetcher,
)


class HistoricalFetcherMixin:
    """Mixin to add date filtering to fetchers"""
    
    @staticmethod
    def filter_posts_by_date(posts: List[Dict], target_date: date_type) -> List[Dict]:
        """Filter posts to only include those from a specific date"""
        filtered = []
        for post in posts:
            published_at = post.get("published_at")
            if published_at and published_at.date() == target_date:
                filtered.append(post)
        return filtered


class HistoricalProductHuntFetcher(ProductHuntFetcher):
    """Product Hunt fetcher with date range support"""
    
    def fetch_for_date(self, target_date: date_type) -> Dict[str, Any]:
        """
        Fetch Product Hunt posts for a specific historical date
        
        Args:
            target_date: Date to fetch posts for
            
        Returns:
            Dictionary with posts and topics from that date
        """
        if not self.api_token:
            return {"posts": [], "topics": []}
        
        posts = []
        discovered_topics = set()
        
        try:
            import requests
            url = "https://api.producthunt.com/v2/api/graphql"
            
            # Create date range for the entire day
            start_time = datetime.combine(target_date, datetime.min.time()).strftime("%Y-%m-%dT00:00:00Z")
            end_time = datetime.combine(target_date, datetime.max.time()).strftime("%Y-%m-%dT23:59:59Z")
            
            query = """
            query {
              posts(order: VOTES, postedAfter: "%s", postedBefore: "%s") {
                edges {
                  node {
                    id
                    name
                    tagline
                    description
                    votesCount
                    commentsCount
                    createdAt
                    website
                    topics {
                      edges {
                        node {
                          name
                        }
                      }
                    }
                  }
                }
              }
            }
            """ % (start_time, end_time)
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            
            response = requests.post(
                url,
                json={"query": query},
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            edges = data.get("data", {}).get("posts", {}).get("edges", [])
            
            for edge in edges:
                node = edge.get("node", {})
                
                created_at = node.get("createdAt")
                if created_at:
                    published_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                else:
                    published_at = timezone.make_aware(
                        datetime.combine(target_date, datetime.min.time())
                    )
                
                # Extract topics
                post_topics = []
                topics_edges = node.get("topics", {}).get("edges", [])
                for topic_edge in topics_edges:
                    topic_name = topic_edge.get("node", {}).get("name")
                    if topic_name:
                        post_topics.append(topic_name)
                        discovered_topics.add(topic_name)
                
                # Add product name as a topic
                product_name = node.get("name", "")
                if product_name:
                    post_topics.append(product_name)
                    discovered_topics.add(product_name)
                
                posts.append({
                    "external_id": f"ph_{node.get('id', '')}",
                    "title": f"{node.get('name', '')} - {node.get('tagline', '')}",
                    "url": node.get("website", ""),
                    "source": "PRODUCT_HUNT",
                    "score": node.get("votesCount", 0),
                    "num_comments": node.get("commentsCount", 0),
                    "author": "",
                    "content": node.get("description", ""),
                    "published_at": published_at,
                    "topics": post_topics,
                })
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching from Product Hunt for {target_date}: {e}")
            return {"posts": [], "topics": []}


class HistoricalHackerNewsFetcher(HackerNewsFetcher):
    """HackerNews fetcher with date range support"""
    
    def fetch_show_hn_for_date(self, target_date: date_type, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch Show HN posts for a specific date"""
        import requests
        posts = []
        
        try:
            # Create timestamp range for the target date
            start_time = int(datetime.combine(target_date, datetime.min.time()).timestamp())
            end_time = int(datetime.combine(target_date, datetime.max.time()).timestamp())
            
            url = "https://hn.algolia.com/api/v1/search"
            params = {
                "tags": "show_hn",
                "numericFilters": f"created_at_i>={start_time},created_at_i<={end_time},points>3",
                "hitsPerPage": limit,
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get("hits", [])
            
            for hit in hits:
                created_at = hit.get("created_at")
                if created_at:
                    published_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                else:
                    published_at = timezone.make_aware(
                        datetime.combine(target_date, datetime.min.time())
                    )
                
                # Extract product name from title if present
                title = hit.get("title", "")
                post_topics = ["Show HN"]
                
                # Try to extract product name from "Show HN: ProductName - Description"
                if "show hn:" in title.lower():
                    # Extract text after "Show HN:"
                    parts = title.split(":", 1)
                    if len(parts) > 1:
                        product_part = parts[1].strip()
                        # Take first part before dash or parenthesis
                        product_name = product_part.split("-")[0].split("(")[0].strip()
                        if product_name and len(product_name) > 2:
                            post_topics.append(product_name)
                
                posts.append({
                    "external_id": f"show_hn_{hit.get('objectID', '')}",
                    "title": title,
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                    "source": "HN",
                    "score": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "author": hit.get("author", ""),
                    "published_at": published_at,
                    "topics": post_topics,
                })
                
            return posts
            
        except Exception as e:
            print(f"Error fetching Show HN for {target_date}: {e}")
            return []


class HistoricalAggregator:
    """Aggregator for fetching historical data day-by-day"""
    
    def __init__(self, product_hunt_token: str = None):
        self.product_hunt_fetcher = HistoricalProductHuntFetcher(product_hunt_token)
        self.hn_fetcher = HistoricalHackerNewsFetcher()
        self.reddit_fetcher = RedditFetcher()
        self.github_fetcher = GitHubFetcher()
        self.devto_fetcher = DevToFetcher()
        self.indiehackers_fetcher = IndieHackersFetcher()
        self.mixin = HistoricalFetcherMixin()
    
    def fetch_for_date(
        self,
        target_date: date_type,
        include_product_hunt: bool = True,
        include_devto: bool = True,
        include_show_hn: bool = True,
        include_reddit_rising: bool = True,
        include_github_new: bool = True,
        include_indiehackers: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch data for a specific historical date
        
        Args:
            target_date: The date to fetch data for
            include_*: Which sources to include
            
        Returns:
            Dictionary with posts, topics, and stats for that date
        """
        import time
        
        all_posts = []
        discovered_topics = set()
        stats = {
            "total_posts": 0,
            "sources": {},
            "errors": [],
            "discovered_topics": 0,
            "date": str(target_date),
        }
        
        # Product Hunt - date-specific fetch
        if include_product_hunt:
            try:
                ph_result = self.product_hunt_fetcher.fetch_for_date(target_date)
                ph_posts = ph_result.get("posts", [])
                ph_topics = ph_result.get("topics", [])
                
                all_posts.extend(ph_posts)
                discovered_topics.update(ph_topics)
                stats["sources"]["ProductHunt"] = len(ph_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"ProductHunt: {str(e)}")
        
        # Show HN - date-specific fetch
        if include_show_hn:
            try:
                show_hn_posts = self.hn_fetcher.fetch_show_hn_for_date(target_date, limit=20)
                
                all_posts.extend(show_hn_posts)
                stats["sources"]["ShowHN"] = len(show_hn_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"ShowHN: {str(e)}")
        
        # For other sources (Reddit, GitHub, Dev.to), fetch recent and filter by date
        # These APIs don't always support precise date filtering, so we fetch and filter
        
        if include_devto:
            try:
                devto_result = self.devto_fetcher.fetch(days_ago=30)
                devto_posts = self.mixin.filter_posts_by_date(
                    devto_result.get("posts", []), target_date
                )
                devto_topics = devto_result.get("topics", [])
                
                all_posts.extend(devto_posts)
                discovered_topics.update(devto_topics)
                stats["sources"]["DevTo"] = len(devto_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"DevTo: {str(e)}")
        
        if include_reddit_rising:
            try:
                reddit_result = self.reddit_fetcher.fetch(sort="new", limit=50)
                reddit_posts = self.mixin.filter_posts_by_date(
                    reddit_result.get("posts", []), target_date
                )
                reddit_topics = reddit_result.get("topics", [])
                
                all_posts.extend(reddit_posts)
                discovered_topics.update(reddit_topics)
                stats["sources"]["Reddit"] = len(reddit_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"Reddit: {str(e)}")
        
        if include_github_new:
            try:
                # GitHub: fetch repos created on target date
                github_result = self.github_fetcher.fetch_recently_created(
                    days_ago=30, limit=50, min_stars=5
                )
                github_posts = self.mixin.filter_posts_by_date(
                    github_result.get("posts", []), target_date
                )
                github_topics = github_result.get("topics", [])
                
                all_posts.extend(github_posts)
                discovered_topics.update(github_topics)
                stats["sources"]["GitHub"] = len(github_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"GitHub: {str(e)}")
        
        if include_indiehackers:
            try:
                # Indie Hackers doesn't have good date support, skip for historical
                # (or fetch current and filter)
                ih_result = self.indiehackers_fetcher.fetch(limit=30)
                ih_posts = self.mixin.filter_posts_by_date(
                    ih_result.get("posts", []), target_date
                )
                ih_topics = ih_result.get("topics", [])
                
                all_posts.extend(ih_posts)
                discovered_topics.update(ih_topics)
                stats["sources"]["IndieHackers"] = len(ih_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"IndieHackers: {str(e)}")
        
        stats["total_posts"] = len(all_posts)
        stats["discovered_topics"] = len(discovered_topics)
        
        return {
            "all_posts": all_posts,
            "discovered_topics": list(discovered_topics),
            "stats": stats,
        }
