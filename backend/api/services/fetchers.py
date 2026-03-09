"""
Data fetchers for Trendify - adapted from proof of concept script
Fetches trending content from multiple platforms (HN, Reddit, GitHub)
"""

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
    ]

    def fetch(
        self, subreddits: List[str] = None, limit: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch HOT posts from tech subreddits

        Args:
            subreddits: List of subreddit names (without r/)
            limit: Maximum posts per subreddit

        Returns:
            Dictionary with posts and discovered topics
        """
        if subreddits is None:
            subreddits = self.DEFAULT_SUBREDDITS

        posts = []
        discovered_topics = set()  # Track unique topics

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

    def extract_keywords(self, posts: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """
        Extract trending keywords from posts

        Args:
            posts: List of post dictionaries

        Returns:
            List of (keyword, count) tuples
        """
        return self.keyword_extractor.extract(posts)
