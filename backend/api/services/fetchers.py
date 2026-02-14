"""
Data fetchers for Trendify - adapted from proof of concept script
Fetches trending content from multiple platforms (HN, Reddit, GitHub)
"""

import re
import time
from collections import Counter
from datetime import datetime, timedelta
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
                    }
                )

            return posts

        except Exception as e:
            print(f"Error fetching from Hacker News: {e}")
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
    ]

    def fetch(
        self, subreddits: List[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch HOT posts from tech subreddits

        Args:
            subreddits: List of subreddit names (without r/)
            limit: Maximum posts per subreddit

        Returns:
            List of post dictionaries
        """
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS

        posts = []

        for sub in subreddits:
            try:
                url = f"https://www.reddit.com/r/{sub}/hot.json"
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
                            created_utc, tz=timezone.utc
                        )
                    else:
                        published_at = timezone.now()

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
                        }
                    )

                # Be nice to Reddit's rate limits
                time.sleep(1)

            except Exception as e:
                print(f"Error fetching from r/{sub}: {e}")
                continue

        return posts


class GitHubFetcher(BaseFetcher):
    """Fetches trending repositories from GitHub"""

    def fetch(self, days_ago: int = 7, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Fetch trending repositories from GitHub

        Args:
            days_ago: Only fetch repos created in last N days
            limit: Maximum number of repos to fetch

        Returns:
            List of post dictionaries (repos as posts)
        """
        posts = []

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
                    }
                )

            return posts

        except Exception as e:
            print(f"Error fetching from GitHub: {e}")
            return []


class KeywordExtractor:
    """
    Extract trending keywords from post titles using word frequency analysis
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

    def __init__(self):
        self.hn_fetcher = HackerNewsFetcher()
        self.reddit_fetcher = RedditFetcher()
        self.github_fetcher = GitHubFetcher()
        self.keyword_extractor = KeywordExtractor()

    def fetch_all(
        self,
        include_hn: bool = True,
        include_reddit: bool = True,
        include_github: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch from all platforms

        Args:
            include_hn: Whether to fetch from Hacker News
            include_reddit: Whether to fetch from Reddit
            include_github: Whether to fetch from GitHub

        Returns:
            Dictionary with all_posts and stats
        """
        all_posts = []
        stats = {
            "total_posts": 0,
            "sources": {},
            "errors": [],
        }

        # Fetch from each platform
        if include_hn:
            try:
                hn_posts = self.hn_fetcher.fetch()
                all_posts.extend(hn_posts)
                stats["sources"]["HN"] = len(hn_posts)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                stats["errors"].append(f"HN: {str(e)}")

        if include_reddit:
            try:
                reddit_posts = self.reddit_fetcher.fetch()
                all_posts.extend(reddit_posts)
                stats["sources"]["Reddit"] = len(reddit_posts)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                stats["errors"].append(f"Reddit: {str(e)}")

        if include_github:
            try:
                github_posts = self.github_fetcher.fetch()
                all_posts.extend(github_posts)
                stats["sources"]["GitHub"] = len(github_posts)
            except Exception as e:
                stats["errors"].append(f"GitHub: {str(e)}")

        stats["total_posts"] = len(all_posts)

        return {
            "all_posts": all_posts,
            "stats": stats,
        }

    def extract_keywords(self, posts: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Extract trending keywords from posts

        Args:
            posts: List of post dictionaries

        Returns:
            List of (keyword, count) tuples
        """
        return self.keyword_extractor.extract(posts)