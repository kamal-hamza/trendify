"""
Data fetchers for Trendify - adapted from proof of concept script
Fetches trending content from multiple platforms (HN, Reddit, GitHub)
"""

import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Dict, List, Tuple

import requests
from django.utils import timezone


class BaseFetcher:
    """Base class for all platform fetchers"""

    def __init__(self):
        self.user_agent = {"User-Agent": "Trendify/1.0 (Django)"}
        self.timeout = 10

    def fetch(self) -> List[Dict[str, Any]]:
        """Fetch posts from the platform. Must be implemented by subclasses."""
        raise NotImplementedError


class HackerNewsFetcher(BaseFetcher):
    """Fetches trending stories from Hacker News using Algolia API"""

    def fetch_show_hn(self, limit: int = 20, hours_ago: int = 48) -> List[Dict[str, Any]]:
        """
        Fetch "Show HN" posts - great for catching product launches
        
        Args:
            limit: Maximum number of stories to fetch
            hours_ago: Only fetch stories from this many hours ago
            
        Returns:
            List of post dictionaries
        """
        posts = []
        
        try:
            cutoff_time = int(
                (datetime.now() - timedelta(hours=hours_ago)).timestamp()
            )
            
            url = "https://hn.algolia.com/api/v1/search"
            params = {
                "tags": "show_hn",
                "numericFilters": f"created_at_i>{cutoff_time},points>3",
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
                    published_at = timezone.now()
                
                posts.append(
                    {
                        "external_id": f"show_hn_{hit.get('objectID', '')}",
                        "title": hit.get("title", ""),
                        "url": hit.get("url")
                        or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                        "source": "HN",
                        "score": hit.get("points", 0),
                        "num_comments": hit.get("num_comments", 0),
                        "author": hit.get("author", ""),
                        "published_at": published_at,
                        "topics": ["Show HN"],
                    }
                )
                
            return posts
            
        except Exception as e:
            print(f"Error fetching Show HN: {e}")
            return []

    def fetch_launches(self, limit: int = 20, hours_ago: int = 72) -> List[Dict[str, Any]]:
        """
        Fetch product launches by searching for launch-related keywords
        
        Args:
            limit: Maximum number of stories to fetch
            hours_ago: Only fetch stories from this many hours ago
            
        Returns:
            List of post dictionaries
        """
        launch_keywords = ["launch", "released", "announcing", "introducing", "new version"]
        posts = []
        seen_ids = set()
        
        try:
            cutoff_time = int(
                (datetime.now() - timedelta(hours=hours_ago)).timestamp()
            )
            
            url = "https://hn.algolia.com/api/v1/search"
            
            for keyword in launch_keywords:
                params = {
                    "query": keyword,
                    "tags": "story",
                    "numericFilters": f"created_at_i>{cutoff_time},points>5",
                    "hitsPerPage": limit // len(launch_keywords) + 1,
                }
                
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                hits = data.get("hits", [])
                
                for hit in hits:
                    story_id = str(hit.get("objectID", ""))
                    
                    if story_id in seen_ids:
                        continue
                    seen_ids.add(story_id)
                    
                    created_at = hit.get("created_at")
                    if created_at:
                        published_at = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                    else:
                        published_at = timezone.now()
                    
                    posts.append(
                        {
                            "external_id": f"launch_{story_id}",
                            "title": hit.get("title", ""),
                            "url": hit.get("url")
                            or f"https://news.ycombinator.com/item?id={story_id}",
                            "source": "HN",
                            "score": hit.get("points", 0),
                            "num_comments": hit.get("num_comments", 0),
                            "author": hit.get("author", ""),
                            "published_at": published_at,
                            "topics": ["Launch"],
                        }
                    )
                
                time.sleep(0.3)
                
            return posts[:limit]
            
        except Exception as e:
            print(f"Error fetching launches: {e}")
            return []

    def fetch(self, limit: int = 30, hours_ago: int = 24) -> List[Dict[str, Any]]:
        """
        Fetch HOT stories from Hacker News (front page material)

        Args:
            limit: Maximum number of stories to fetch
            hours_ago: Only fetch stories from this many hours ago

        Returns:
            List of post dictionaries
        """
        posts = []

        try:
            # Calculate timestamp for filtering
            cutoff_time = int(
                (datetime.now() - timedelta(hours=hours_ago)).timestamp()
            )

            url = "https://hn.algolia.com/api/v1/search"
            params = {
                "tags": "story",
                "numericFilters": f"created_at_i>{cutoff_time},points>10",
                "hitsPerPage": limit,
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            hits = data.get("hits", [])

            for hit in hits:
                # Parse timestamp
                created_at = hit.get("created_at")
                if created_at:
                    published_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                else:
                    published_at = timezone.now()

                posts.append(
                    {
                        "external_id": str(hit.get("objectID", "")),
                        "title": hit.get("title", ""),
                        "url": hit.get("url")
                        or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                        "source": "HN",
                        "score": hit.get("points", 0),
                        "num_comments": hit.get("num_comments", 0),
                        "author": hit.get("author", ""),
                        "published_at": published_at,
                        "topics": [],  # HN doesn't provide explicit topics
                    }
                )

            return posts

        except Exception as e:
            print(f"Error fetching from Hacker News: {e}")
            return []

    def search_by_topics(self, topics: List[str], limit_per_topic: int = 5, hours_ago: int = 24) -> List[Dict[str, Any]]:
        """
        Search Hacker News for posts related to specific topics
        
        Args:
            topics: List of topics/keywords to search for
            limit_per_topic: Maximum posts per topic
            hours_ago: Only fetch stories from this many hours ago
            
        Returns:
            List of post dictionaries matching the topics
        """
        posts = []
        seen_ids = set()  # Avoid duplicates

        try:
            cutoff_time = int(
                (datetime.now() - timedelta(hours=hours_ago)).timestamp()
            )

            url = "https://hn.algolia.com/api/v1/search"

            for topic in topics:
                try:
                    params = {
                        "query": topic,
                        "tags": "story",
                        "numericFilters": f"created_at_i>{cutoff_time},points>5",
                        "hitsPerPage": limit_per_topic,
                    }

                    response = requests.get(url, params=params, timeout=self.timeout)
                    response.raise_for_status()

                    data = response.json()
                    hits = data.get("hits", [])

                    for hit in hits:
                        story_id = str(hit.get("objectID", ""))
                        
                        # Skip duplicates
                        if story_id in seen_ids:
                            continue
                        seen_ids.add(story_id)

                        # Parse timestamp
                        created_at = hit.get("created_at")
                        if created_at:
                            published_at = datetime.fromisoformat(
                                created_at.replace("Z", "+00:00")
                            )
                        else:
                            published_at = timezone.now()

                        posts.append(
                            {
                                "external_id": story_id,
                                "title": hit.get("title", ""),
                                "url": hit.get("url")
                                or f"https://news.ycombinator.com/item?id={story_id}",
                                "source": "HN",
                                "score": hit.get("points", 0),
                                "num_comments": hit.get("num_comments", 0),
                                "author": hit.get("author", ""),
                                "published_at": published_at,
                                "topics": [topic],  # Tag with search topic
                            }
                        )

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    print(f"Error searching HN for topic '{topic}': {e}")
                    continue

            return posts

        except Exception as e:
            print(f"Error in HN topic search: {e}")
            return []


class RedditFetcher(BaseFetcher):
    """Fetches hot posts from tech-focused subreddits"""

    DEFAULT_SUBREDDITS = [
        "programming",
        "LocalLLaMA",
        "technology",
        "MachineLearning",
        "webdev",
        "Python",
        "javascript",
        "artificial",
        # Bleeding-edge / emerging content subreddits
        "SideProject",           # New side projects and launches
        "AlphaAndBetaUsers",     # Beta testing new products
        "coolgithubprojects",    # Fresh GitHub discoveries
        "startups",              # Startup launches and news
        "Entrepreneur",          # New business ventures
        "EntrepreneurRideAlong", # Startup journey posts
        "indiehackers",          # Indie maker launches (if exists)
        "buildinpublic",         # Building in public
        "SaaS",                  # New SaaS products
        "nocode",                # No-code tool launches
        "IMadeThis",             # Personal project launches
        "golang",                # Go language (emerging projects)
        "rust",                  # Rust language (bleeding edge)
        "webassembly",           # WebAssembly projects
        "kubernetes",            # K8s new tools
        "devops",                # DevOps tool launches
        "selfhosted",            # Self-hosted project launches
        "opensource",            # Open source project announcements
        "aws",                   # AWS new services/tools
        "docker",                # Docker/container innovations
    ]

    def fetch(
        self, subreddits: List[str] = None, limit: int = 20, sort: str = "rising"
    ) -> Dict[str, Any]:
        """
        Fetch posts from tech subreddits (default: rising for emerging content)

        Args:
            subreddits: List of subreddit names (without r/)
            limit: Maximum posts per subreddit
            sort: Sort method - "rising", "hot", "new" (default: rising for emerging trends)

        Returns:
            Dictionary with posts and discovered topics
        """
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS

        posts = []
        discovered_topics = set()  # Track unique topics

        for sub in subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/{sort}.json"
                params = {"limit": limit}

                response = requests.get(
                    url, headers=self.user_agent, timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                children = data.get("data", {}).get("children", [])

                for child in children:
                    post_data = child.get("data", {})

                    # Filter out stickied posts and low-score posts
                    if post_data.get("stickied") or post_data.get("score", 0) < 10:
                        continue

                    # Parse timestamp
                    created_utc = post_data.get("created_utc", 0)
                    if created_utc:
                        published_at = datetime.fromtimestamp(
                            created_utc, tz=dt_timezone.utc
                        )
                    else:
                        published_at = timezone.now()

                    # Extract topics from subreddit and flair
                    post_topics = []
                    
                    # Add subreddit as topic
                    subreddit_name = post_data.get("subreddit", sub)
                    post_topics.append(subreddit_name)
                    discovered_topics.add(subreddit_name)
                    
                    # Add flair if available
                    flair = post_data.get("link_flair_text")
                    if flair:
                        post_topics.append(flair)
                        discovered_topics.add(flair)

                    posts.append(
                        {
                            "external_id": post_data.get("id", ""),
                            "title": post_data.get("title", ""),
                            "url": post_data.get("url", ""),
                            "source": f"REDDIT_{sub.upper()}",
                            "score": post_data.get("score", 0),
                            "num_comments": post_data.get("num_comments", 0),
                            "author": post_data.get("author", ""),
                            "content": post_data.get("selftext", ""),
                            "published_at": published_at,
                            "topics": post_topics,  # Add topics metadata
                        }
                    )

                # Be nice to Reddit's rate limits
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching from r/{sub}: {e}")
                continue

        return {
            "posts": posts,
            "topics": list(discovered_topics),
        }


class GitHubFetcher(BaseFetcher):
    """Fetches trending repositories from GitHub"""

    def fetch_recently_created(self, days_ago: int = 14, limit: int = 20, min_stars: int = 10) -> Dict[str, Any]:
        """
        Fetch recently created repositories - better for emerging trends
        
        Args:
            days_ago: Only fetch repos created in last N days
            limit: Maximum number of repos to fetch
            min_stars: Minimum stars to filter noise
            
        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_ago)).strftime(
                "%Y-%m-%d"
            )
            
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"created:>{cutoff_date} stars:>={min_stars}",
                "sort": "stars",
                "order": "desc",
                "per_page": limit,
            }
            headers = {
                "Accept": "application/vnd.github+json",
                **self.user_agent,
            }
            
            response = requests.get(
                url, params=params, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            for repo in items:
                name = repo.get("name", "")
                description = repo.get("description", "No description")
                title = f"{name} - {description}"
                
                created_at = repo.get("created_at")
                if created_at:
                    published_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                else:
                    published_at = timezone.now()
                
                post_topics = []
                
                repo_topics = repo.get("topics", [])
                for topic in repo_topics:
                    post_topics.append(topic)
                    discovered_topics.add(topic)
                
                language = repo.get("language")
                if language:
                    post_topics.append(language)
                    discovered_topics.add(language)
                
                posts.append(
                    {
                        "external_id": f"github_new_{repo.get('id', '')}",
                        "title": title,
                        "url": repo.get("html_url", ""),
                        "source": "GITHUB",
                        "score": repo.get("stargazers_count", 0),
                        "num_comments": repo.get("open_issues_count", 0),
                        "author": repo.get("owner", {}).get("login", ""),
                        "content": description,
                        "published_at": published_at,
                        "topics": post_topics,
                    }
                )
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching recently created GitHub repos: {e}")
            return {"posts": [], "topics": []}

    def fetch(self, days_ago: int = 7, limit: int = 15) -> Dict[str, Any]:
        """
        Fetch trending repositories from GitHub

        Args:
            days_ago: Only fetch repos created in last N days
            limit: Maximum number of repos to fetch

        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()  # Track unique topics

        try:
            # Search for repos created recently with high stars
            cutoff_date = (datetime.now() - timedelta(days=days_ago)).strftime(
                "%Y-%m-%d"
            )

            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"created:>{cutoff_date}",
                "sort": "stars",
                "order": "desc",
                "per_page": limit,
            }
            headers = {
                "Accept": "application/vnd.github+json",
                **self.user_agent,
            }

            response = requests.get(
                url, params=params, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            for repo in items:
                # Use repo name + description as "title"
                name = repo.get("name", "")
                description = repo.get("description", "No description")
                title = f"{name} - {description}"

                # Parse timestamp
                created_at = repo.get("created_at")
                if created_at:
                    published_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                else:
                    published_at = timezone.now()

                # Extract topics from GitHub repo
                post_topics = []
                
                # Add GitHub topics (hashtags)
                repo_topics = repo.get("topics", [])
                for topic in repo_topics:
                    post_topics.append(topic)
                    discovered_topics.add(topic)
                
                # Add programming language
                language = repo.get("language")
                if language:
                    post_topics.append(language)
                    discovered_topics.add(language)

                posts.append(
                    {
                        "external_id": str(repo.get("id", "")),
                        "title": title,
                        "url": repo.get("html_url", ""),
                        "source": "GITHUB",
                        "score": repo.get("stargazers_count", 0),
                        "num_comments": repo.get("open_issues_count", 0),
                        "author": repo.get("owner", {}).get("login", ""),
                        "content": description,
                        "published_at": published_at,
                        "topics": post_topics,  # Add topics metadata
                    }
                )

            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }

        except Exception as e:
            print(f"Error fetching from GitHub: {e}")
            return {"posts": [], "topics": []}


class ProductHuntFetcher(BaseFetcher):
    """Fetches new product launches from Product Hunt"""
    
    def __init__(self, api_token: str = None):
        super().__init__()
        self.api_token = api_token
    
    def fetch(self, days_ago: int = 1) -> Dict[str, Any]:
        """
        Fetch recent product launches from Product Hunt
        
        Note: This requires a Product Hunt API token.
        Set PRODUCT_HUNT_API_TOKEN in environment variables.
        
        Args:
            days_ago: Number of days back to fetch
            
        Returns:
            Dictionary with posts and discovered topics
        """
        if not self.api_token:
            print("Product Hunt API token not configured. Skipping Product Hunt fetch.")
            return {"posts": [], "topics": []}
        
        posts = []
        discovered_topics = set()
        
        try:
            # Product Hunt GraphQL API
            url = "https://api.producthunt.com/v2/api/graphql"
            
            # Query for posts from the last N days
            query = """
            query {
              posts(order: VOTES, postedAfter: "%s") {
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
            """ % (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
            
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
                    published_at = timezone.now()
                
                # Extract topics
                post_topics = []
                topics_edges = node.get("topics", {}).get("edges", [])
                for topic_edge in topics_edges:
                    topic_name = topic_edge.get("node", {}).get("name")
                    if topic_name:
                        post_topics.append(topic_name)
                        discovered_topics.add(topic_name)
                
                posts.append(
                    {
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
                    }
                )
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching from Product Hunt: {e}")
            return {"posts": [], "topics": []}


class IndieHackersFetcher(BaseFetcher):
    """Fetches trending posts from Indie Hackers via web scraping"""
    
    def fetch(self, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch trending posts from Indie Hackers
        
        Note: This uses web scraping since Indie Hackers doesn't have a public API
        
        Args:
            limit: Maximum number of posts to fetch
            
        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()
        
        try:
            url = "https://www.indiehackers.com/posts"
            
            response = requests.get(url, headers=self.user_agent, timeout=self.timeout)
            response.raise_for_status()
            
            # Simple extraction - look for trending posts in the feed
            # This is a basic implementation that may need adjustment
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Indie Hackers structure may vary, this is a basic approach
            articles = soup.find_all('div', class_='feed-item', limit=limit)
            
            for article in articles:
                try:
                    title_elem = article.find(['h2', 'h3', 'a'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://www.indiehackers.com{link}"
                    
                    # Try to get votes/score
                    score_elem = article.find(['span', 'div'], class_=['votes', 'score'])
                    score = 0
                    if score_elem:
                        score_text = score_elem.get_text(strip=True)
                        try:
                            score = int(''.join(filter(str.isdigit, score_text)))
                        except:
                            score = 10  # Default
                    
                    posts.append(
                        {
                            "external_id": f"ih_{hash(link)}",
                            "title": title,
                            "url": link or "https://www.indiehackers.com/posts",
                            "source": "INDIEHACKERS",
                            "score": score,
                            "num_comments": 0,
                            "author": "",
                            "content": "",
                            "published_at": timezone.now(),
                            "topics": ["indie", "startup"],
                        }
                    )
                    
                    discovered_topics.add("indie")
                    discovered_topics.add("startup")
                except Exception as e:
                    continue
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching from Indie Hackers: {e}")
            return {"posts": [], "topics": []}


class DevToFetcher(BaseFetcher):
    """Fetches trending posts from Dev.to, especially #showdev"""
    
    def fetch(self, days_ago: int = 7, per_page: int = 30, tag: str = "showdev") -> Dict[str, Any]:
        """
        Fetch trending articles from Dev.to, filtered by tag
        
        Args:
            days_ago: Only fetch articles from last N days
            per_page: Number of articles per page
            tag: Tag to filter by (default: "showdev" for Show Dev posts)
            
        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()
        
        try:
            url = "https://dev.to/api/articles"
            params = {
                "per_page": per_page,
                "top": days_ago,  # Top articles from last N days
                "tag": tag,  # Filter by showdev tag
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            articles = response.json()
            
            for article in articles:
                published_at_str = article.get("published_at")
                if published_at_str:
                    published_at = datetime.fromisoformat(
                        published_at_str.replace("Z", "+00:00")
                    )
                else:
                    published_at = timezone.now()
                
                # Extract tags as topics
                post_topics = []
                tags = article.get("tag_list", [])
                for tag_name in tags:
                    post_topics.append(tag_name)
                    discovered_topics.add(tag_name)
                
                posts.append(
                    {
                        "external_id": f"devto_{article.get('id', '')}",
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "source": "DEVTO",
                        "score": article.get("positive_reactions_count", 0),
                        "num_comments": article.get("comments_count", 0),
                        "author": article.get("user", {}).get("username", ""),
                        "content": article.get("description", ""),
                        "published_at": published_at,
                        "topics": post_topics,
                    }
                )
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching from Dev.to: {e}")
            return {"posts": [], "topics": []}


class GitHubTrendingFetcher(BaseFetcher):
    """Fetches daily trending repositories from GitHub"""
    
    def fetch(self, language: str = "", days_ago: int = 1) -> Dict[str, Any]:
        """
        Fetch trending repositories from GitHub
        
        Args:
            language: Filter by programming language (empty for all languages)
            days_ago: Fetch repos created in the last N days (default: 1 for daily trending)
            
        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()
        
        try:
            # Use GitHub Search API to find recently created repos sorted by stars
            url = "https://api.github.com/search/repositories"
            
            # Calculate date for filtering
            from datetime import datetime, timedelta
            since_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Build query
            query = f"created:>{since_date}"
            if language:
                query += f" language:{language}"
            
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": 30,
            }
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
            }
            
            # Add GitHub token if available
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            repos = data.get("items", [])
            
            for repo in repos:
                # Extract topics from GitHub repo
                post_topics = repo.get("topics", [])
                for topic in post_topics:
                    discovered_topics.add(topic)
                
                # Also add language as a topic
                if repo.get("language"):
                    post_topics.append(repo["language"])
                    discovered_topics.add(repo["language"])
                
                posts.append(
                    {
                        "external_id": f"github_trending_{repo.get('id', '')}",
                        "title": repo.get("full_name", ""),
                        "url": repo.get("html_url", ""),
                        "source": "GITHUB_TRENDING",
                        "score": repo.get("stargazers_count", 0),
                        "num_comments": repo.get("open_issues_count", 0),
                        "author": repo.get("owner", {}).get("login", ""),
                        "content": repo.get("description", ""),
                        "published_at": datetime.fromisoformat(
                            repo.get("created_at", "").replace("Z", "+00:00")
                        ),
                        "topics": post_topics,
                    }
                )
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching GitHub trending: {e}")
            return {"posts": [], "topics": []}


class LobstersFetcher(BaseFetcher):
    """Fetches posts from Lobste.rs with the 'show' tag"""
    
    def fetch(self, tag: str = "show") -> Dict[str, Any]:
        """
        Fetch posts from Lobste.rs filtered by tag
        
        Args:
            tag: Tag to filter by (default: "show" for Show posts)
            
        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()
        
        try:
            # Lobste.rs has a simple JSON API - just append .json to any URL
            url = f"https://lobste.rs/t/{tag}.json"
            
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            items = response.json()
            
            for item in items:
                # Extract tags as topics
                post_topics = item.get("tags", [])
                for topic in post_topics:
                    discovered_topics.add(topic)
                
                # Get submitter username correctly
                submitter = ""
                if isinstance(item.get("submitter_user"), dict):
                    submitter = item.get("submitter_user", {}).get("username", "")
                elif isinstance(item.get("submitter_user"), str):
                    submitter = item.get("submitter_user", "")
                
                posts.append(
                    {
                        "external_id": f"lobsters_{item.get('short_id', '')}",
                        "title": item.get("title", ""),
                        "url": item.get("url", "") or item.get("comments_url", ""),
                        "source": "LOBSTERS",
                        "score": item.get("score", 0),
                        "num_comments": item.get("comment_count", 0),
                        "author": submitter,
                        "content": item.get("description", ""),
                        "published_at": datetime.fromisoformat(
                            item.get("created_at", "").replace("Z", "+00:00")
                        ),
                        "topics": post_topics,
                    }
                )
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching from Lobste.rs: {e}")
            return {"posts": [], "topics": []}


class TAAFTFetcher(BaseFetcher):
    """Fetches newest AI tools from There's An API For That"""
    
    def fetch(self, limit: int = 30) -> Dict[str, Any]:
        """
        Fetch newest AI tools from TAAFT
        
        Args:
            limit: Number of tools to fetch
            
        Returns:
            Dictionary with posts and discovered topics
        """
        posts = []
        discovered_topics = set()
        
        try:
            # Note: TAAFT API may require authentication or may block automated requests
            # Using a simple scraping approach as fallback if API is not accessible
            # For now, we'll return empty results if API fails (403 errors are common)
            print("Note: TAAFT fetcher is experimental and may not work without API access")
            return {"posts": [], "topics": []}
            
            # TAAFT has an API for their newest tools (keeping code for reference)
            url = "https://www.theresanaiforthat.com/api/tools"
            params = {
                "sort": "newest",
                "limit": limit,
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            tools = data.get("tools", [])
            
            for tool in tools:
                # Extract categories/tags as topics
                post_topics = []
                categories = tool.get("categories", [])
                for category in categories:
                    if isinstance(category, dict):
                        cat_name = category.get("name", "")
                    else:
                        cat_name = str(category)
                    
                    if cat_name:
                        post_topics.append(cat_name)
                        discovered_topics.add(cat_name)
                
                # Also add AI-related tags
                tags = tool.get("tags", [])
                for tag in tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get("name", "")
                    else:
                        tag_name = str(tag)
                    
                    if tag_name:
                        post_topics.append(tag_name)
                        discovered_topics.add(tag_name)
                
                posts.append(
                    {
                        "external_id": f"taaft_{tool.get('id', '')}",
                        "title": tool.get("name", ""),
                        "url": tool.get("url", "") or tool.get("website", ""),
                        "source": "TAAFT",
                        "score": tool.get("upvotes", 0) or tool.get("votes", 0),
                        "num_comments": 0,  # TAAFT doesn't have comments
                        "author": tool.get("submitter", "") or "",
                        "content": tool.get("description", ""),
                        "published_at": datetime.fromisoformat(
                            tool.get("created_at", "").replace("Z", "+00:00")
                        ) if tool.get("created_at") else timezone.now(),
                        "topics": post_topics,
                    }
                )
            
            return {
                "posts": posts,
                "topics": list(discovered_topics),
            }
            
        except Exception as e:
            print(f"Error fetching from TAAFT: {e}")
            # If API fails, try scraping as fallback
            return {"posts": [], "topics": []}


class KeywordExtractor:
    """
    Extract trending keywords from post titles using word frequency analysis
    
    NOTE: This class is currently DISABLED in the main pipeline.
    We are focusing on native tags from platforms (Product Hunt tags, GitHub topics, etc.)
    instead of extracting keywords from text.
    """

    def __init__(self):
        # Common words to filter out (stop words + tech noise)
        self.stop_words = {
            "the",
            "and",
            "for",
            "with",
            "this",
            "that",
            "from",
            "show",
            "hn:",
            "new",
            "released",
            "how",
            "why",
            "what",
            "when",
            "where",
            "who",
            "your",
            "you",
            "are",
            "can",
            "have",
            "has",
            "had",
            "was",
            "were",
            "been",
            "being",
            "not",
            "but",
            "all",
            "any",
            "some",
            "use",
            "using",
            "make",
            "made",
            "get",
            "got",
            "like",
            "just",
            "now",
            "than",
            "more",
            "out",
            "into",
            "about",
            "them",
            "these",
            "those",
            "then",
        }

    def extract(
        self, posts: List[Dict[str, Any]], min_length: int = 3, top_n: int = 20
    ) -> List[Tuple[str, int]]:
        """
        Extract trending keywords from posts

        Args:
            posts: List of post dictionaries
            min_length: Minimum keyword length
            top_n: Return top N keywords

        Returns:
            List of (keyword, count) tuples
        """
        # Combine all titles
        all_titles = " ".join([post.get("title", "") for post in posts])

        keywords = []

        # 1. Find proper nouns and capitalized terms (product names, companies)
        proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\.[A-Z0-9]+)*\b", all_titles)
        keywords.extend([w.lower() for w in proper_nouns if len(w) >= min_length])

        # 2. Find version numbers (e.g., "4.6", "v2.0")
        versions = re.findall(r"\bv?\d+\.\d+(?:\.\d+)?\b", all_titles, re.IGNORECASE)
        keywords.extend(versions)

        # 3. Find CamelCase terms (e.g., "OpenAI", "MachineLearning")
        camel_case = re.findall(r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b", all_titles)
        keywords.extend([w.lower() for w in camel_case])

        # 4. Find simple bigrams (two-word phrases)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", all_titles.lower())
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        keywords.extend(bigrams)

        # Filter out stop words and short terms
        filtered = [
            kw
            for kw in keywords
            if kw.lower() not in self.stop_words
            and len(kw) >= min_length
            and not kw.lower().startswith("http")
        ]

        # Count frequency
        keyword_counts = Counter(filtered)

        return keyword_counts.most_common(top_n)


class TrendifyAggregator:
    """
    Main aggregator that coordinates all fetchers
    """

    def __init__(self, product_hunt_token: str = None):
        self.hn_fetcher = HackerNewsFetcher()
        self.reddit_fetcher = RedditFetcher()
        self.github_fetcher = GitHubFetcher()
        self.product_hunt_fetcher = ProductHuntFetcher(product_hunt_token)
        self.devto_fetcher = DevToFetcher()
        self.indiehackers_fetcher = IndieHackersFetcher()
        self.github_trending_fetcher = GitHubTrendingFetcher()
        self.lobsters_fetcher = LobstersFetcher()
        self.taaft_fetcher = TAAFTFetcher()
        self.keyword_extractor = KeywordExtractor()

    def fetch_all(
        self,
        include_hn: bool = True,
        include_reddit: bool = True,
        include_github: bool = True,
        use_topic_search: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch from all platforms with intelligent topic discovery
        
        First fetches from Reddit/GitHub to discover trending topics,
        then uses those topics to search Hacker News for related content.

        Args:
            include_hn: Whether to fetch from Hacker News
            include_reddit: Whether to fetch from Reddit
            include_github: Whether to fetch from GitHub
            use_topic_search: Whether to use discovered topics to search HN

        Returns:
            Dictionary with all_posts, discovered_topics, and stats
        """
        all_posts = []
        discovered_topics = set()
        stats = {
            "total_posts": 0,
            "sources": {},
            "errors": [],
            "discovered_topics": 0,
        }

        # Phase 1: Fetch from Reddit and GitHub to discover topics
        if include_reddit:
            try:
                reddit_result = self.reddit_fetcher.fetch()
                reddit_posts = reddit_result.get("posts", [])
                reddit_topics = reddit_result.get("topics", [])
                
                all_posts.extend(reddit_posts)
                discovered_topics.update(reddit_topics)
                stats["sources"]["Reddit"] = len(reddit_posts)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                stats["errors"].append(f"Reddit: {str(e)}")

        if include_github:
            try:
                github_result = self.github_fetcher.fetch()
                github_posts = github_result.get("posts", [])
                github_topics = github_result.get("topics", [])
                
                all_posts.extend(github_posts)
                discovered_topics.update(github_topics)
                stats["sources"]["GitHub"] = len(github_posts)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                stats["errors"].append(f"GitHub: {str(e)}")

        # Phase 2: Fetch from Hacker News
        if include_hn:
            try:
                # Get front page posts
                hn_posts = self.hn_fetcher.fetch()
                all_posts.extend(hn_posts)
                hn_count = len(hn_posts)
                
                # Phase 3: Search HN using discovered topics (if enabled)
                if use_topic_search and discovered_topics:
                    # Filter and prioritize topics for search
                    # Focus on tech-related topics, limit to top 15
                    priority_topics = self._prioritize_topics(list(discovered_topics))[:15]
                    
                    print(f"Searching HN for {len(priority_topics)} discovered topics...")
                    topic_posts = self.hn_fetcher.search_by_topics(priority_topics, limit_per_topic=3)
                    all_posts.extend(topic_posts)
                    hn_count += len(topic_posts)
                
                stats["sources"]["HN"] = hn_count
                time.sleep(1)  # Rate limiting
            except Exception as e:
                stats["errors"].append(f"HN: {str(e)}")

        stats["total_posts"] = len(all_posts)
        stats["discovered_topics"] = len(discovered_topics)

        return {
            "all_posts": all_posts,
            "discovered_topics": list(discovered_topics),
            "stats": stats,
        }
    
    def _prioritize_topics(self, topics: List[str]) -> List[str]:
        """
        Prioritize and filter topics for HN search
        
        Args:
            topics: List of discovered topics
            
        Returns:
            Filtered and prioritized list of topics
        """
        # Filter out very generic or non-tech topics
        skip_keywords = {
            "question", "help", "discussion", "tutorial", "guide",
            "meta", "news", "announcement", "other", "general",
        }
        
        # Prioritize tech-focused topics
        priority_keywords = {
            "ai", "ml", "python", "javascript", "rust", "go", "java",
            "react", "vue", "angular", "docker", "kubernetes", "aws",
            "machine-learning", "deep-learning", "llm", "gpt", "claude",
            "openai", "anthropic", "security", "blockchain", "crypto",
        }
        
        prioritized = []
        regular = []
        
        for topic in topics:
            topic_lower = topic.lower()
            
            # Skip generic topics
            if topic_lower in skip_keywords:
                continue
            
            # Skip very short topics
            if len(topic) < 2:
                continue
                
            # Prioritize tech keywords
            if topic_lower in priority_keywords or any(kw in topic_lower for kw in priority_keywords):
                prioritized.append(topic)
            else:
                regular.append(topic)
        
        # Return prioritized first, then regular
        return prioritized + regular

    def fetch_emerging_only(
        self,
        include_product_hunt: bool = True,
        include_devto: bool = True,
        include_show_hn: bool = True,
        include_reddit_rising: bool = True,
        include_github_new: bool = True,
        include_indiehackers: bool = True,
        include_github_trending: bool = True,
        include_lobsters: bool = True,
        include_taaft: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch only from emerging/early-stage sources
        Optimized for discovering new products, launches, and trending topics
        
        Args:
            include_product_hunt: Fetch from Product Hunt
            include_devto: Fetch from Dev.to (#showdev tag)
            include_show_hn: Fetch Show HN posts
            include_reddit_rising: Fetch rising posts from Reddit
            include_github_new: Fetch recently created GitHub repos
            include_indiehackers: Fetch from Indie Hackers
            include_github_trending: Fetch from GitHub Trending (daily)
            include_lobsters: Fetch from Lobste.rs (show tag)
            include_taaft: Fetch from There's An API For That (newest AI tools)
            
        Returns:
            Dictionary with all_posts, discovered_topics, and stats
        """
        all_posts = []
        discovered_topics = set()
        stats = {
            "total_posts": 0,
            "sources": {},
            "errors": [],
            "discovered_topics": 0,
        }
        
        # Product Hunt - best for new product launches
        if include_product_hunt:
            try:
                ph_result = self.product_hunt_fetcher.fetch(days_ago=2)
                ph_posts = ph_result.get("posts", [])
                ph_topics = ph_result.get("topics", [])
                
                all_posts.extend(ph_posts)
                discovered_topics.update(ph_topics)
                stats["sources"]["ProductHunt"] = len(ph_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"ProductHunt: {str(e)}")
        
        # Dev.to #showdev - developers showing off their projects
        if include_devto:
            try:
                devto_result = self.devto_fetcher.fetch(days_ago=7, tag="showdev")
                devto_posts = devto_result.get("posts", [])
                devto_topics = devto_result.get("topics", [])
                
                all_posts.extend(devto_posts)
                discovered_topics.update(devto_topics)
                stats["sources"]["DevTo_ShowDev"] = len(devto_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"DevTo: {str(e)}")
        
        # Show HN - product launches on Hacker News
        if include_show_hn:
            try:
                show_hn_posts = self.hn_fetcher.fetch_show_hn(limit=20, hours_ago=48)
                launch_posts = self.hn_fetcher.fetch_launches(limit=15, hours_ago=72)
                
                all_posts.extend(show_hn_posts)
                all_posts.extend(launch_posts)
                stats["sources"]["ShowHN"] = len(show_hn_posts)
                stats["sources"]["HN_Launches"] = len(launch_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"ShowHN: {str(e)}")
        
        # Reddit Rising - early momentum posts
        if include_reddit_rising:
            try:
                # Use rising feed instead of hot
                reddit_result = self.reddit_fetcher.fetch(sort="rising", limit=15)
                reddit_posts = reddit_result.get("posts", [])
                reddit_topics = reddit_result.get("topics", [])
                
                all_posts.extend(reddit_posts)
                discovered_topics.update(reddit_topics)
                stats["sources"]["Reddit_Rising"] = len(reddit_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"Reddit: {str(e)}")
        
        # GitHub Recently Created - new projects
        if include_github_new:
            try:
                github_result = self.github_fetcher.fetch_recently_created(
                    days_ago=14, limit=20, min_stars=10
                )
                github_posts = github_result.get("posts", [])
                github_topics = github_result.get("topics", [])
                
                all_posts.extend(github_posts)
                discovered_topics.update(github_topics)
                stats["sources"]["GitHub_New"] = len(github_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"GitHub: {str(e)}")
        
        # Indie Hackers - indie maker launches
        if include_indiehackers:
            try:
                ih_result = self.indiehackers_fetcher.fetch(limit=15)
                ih_posts = ih_result.get("posts", [])
                ih_topics = ih_result.get("topics", [])
                
                all_posts.extend(ih_posts)
                discovered_topics.update(ih_topics)
                stats["sources"]["IndieHackers"] = len(ih_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"IndieHackers: {str(e)}")
        
        # GitHub Trending - daily trending repos
        if include_github_trending:
            try:
                trending_result = self.github_trending_fetcher.fetch(days_ago=1)
                trending_posts = trending_result.get("posts", [])
                trending_topics = trending_result.get("topics", [])
                
                all_posts.extend(trending_posts)
                discovered_topics.update(trending_topics)
                stats["sources"]["GitHub_Trending"] = len(trending_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"GitHub_Trending: {str(e)}")
        
        # Lobste.rs - high-quality technical content with 'show' tag
        if include_lobsters:
            try:
                lobsters_result = self.lobsters_fetcher.fetch(tag="show")
                lobsters_posts = lobsters_result.get("posts", [])
                lobsters_topics = lobsters_result.get("topics", [])
                
                all_posts.extend(lobsters_posts)
                discovered_topics.update(lobsters_topics)
                stats["sources"]["Lobsters"] = len(lobsters_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"Lobsters: {str(e)}")
        
        # TAAFT - newest AI tools and APIs
        if include_taaft:
            try:
                taaft_result = self.taaft_fetcher.fetch(limit=30)
                taaft_posts = taaft_result.get("posts", [])
                taaft_topics = taaft_result.get("topics", [])
                
                all_posts.extend(taaft_posts)
                discovered_topics.update(taaft_topics)
                stats["sources"]["TAAFT"] = len(taaft_posts)
                time.sleep(1)
            except Exception as e:
                stats["errors"].append(f"TAAFT: {str(e)}")
        
        stats["total_posts"] = len(all_posts)
        stats["discovered_topics"] = len(discovered_topics)
        
        return {
            "all_posts": all_posts,
            "discovered_topics": list(discovered_topics),
            "stats": stats,
        }

    def extract_keywords(self, posts: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Extract trending keywords from posts
        
        NOTE: This method is DISABLED in the main pipeline.
        We are focusing on native tags from platforms instead.

        Args:
            posts: List of post dictionaries

        Returns:
            List of (keyword, count) tuples
        """
        return self.keyword_extractor.extract(posts)
