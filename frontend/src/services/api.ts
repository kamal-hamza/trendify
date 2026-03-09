import axios from 'axios';

// API base URL - defaults to /api which will be proxied by Vite
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens (if needed in future)
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('token');
      // Could redirect to login page here
    }
    return Promise.reject(error);
  }
);

// Types
export interface Topic {
  id: number;
  name: string;
  category: string;
  description?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  mention_count?: number;
  recent_posts_count?: number;
}

export interface TrendingTopic {
  id: number;
  name: string;
  category: string;
  category_display: string;
  total_mentions: number;
  total_engagement: number;
  avg_sentiment: number;
  momentum_score: number;
  engagement_momentum: number;
  recent_posts_count: number;
  sources: Record<string, number>;
  peak_date: string | null;
}

export interface Post {
  id: number;
  external_id: string;
  source: string;
  source_display: string;
  title: string;
  url: string;
  engagement_score: number;
  comment_count: number;
  sentiment_score: number;
  published_at: string;
  created_at: string;
  updated_at: string;
  author?: string;
  content?: string;
  mentions?: TopicMention[];
  topic_names?: string[];
}

export interface TopicMention {
  id: number;
  topic: number;
  topic_name: string;
  topic_category: string;
  relevance_score: number;
  is_primary: boolean;
  created_at: string;
}

export interface Entity {
  id: number;
  canonical_name: string;
  entity_type: string;
  description?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  nodes?: EntityNode[];
  node_count?: number;
}

export interface EntityNode {
  id: number;
  label: string;
  aliases: string[];
  confidence_score: number;
  resolution_method: string;
  created_at: string;
  updated_at: string;
}

export interface TopicDailyMetric {
  id: number;
  topic: number;
  topic_name: string;
  topic_category: string;
  date: string;
  total_mentions: number;
  total_engagement: number;
  avg_sentiment: number;
  momentum_score: number;
  engagement_momentum: number;
  source_breakdown: Record<string, number>;
  created_at: string;
  updated_at: string;
}

export interface FeedStats {
  total_posts: number;
  total_topics: number;
  total_active_topics: number;
  sources_breakdown: Record<string, number>;
  date_range: {
    earliest: string;
    latest: string;
  };
  top_categories: Array<{
    category: string;
    count: number;
  }>;
  avg_sentiment: number;
}

export interface SentimentAnalysis {
  overall_sentiment: number;
  sentiment_by_source: Record<string, number>;
  sentiment_by_category: Record<string, number>;
  sentiment_trend: Array<{
    date: string;
    avg_sentiment: number;
  }>;
}

export interface PaginatedResponse<T> {
  count: number;
  next: string | null;
  previous: string | null;
  results: T[];
}

export interface FilterParams {
  source?: string;
  category?: string;
  days?: number;
  limit?: number;
  ordering?: string;
  search?: string;
  page?: number;
}

// API Functions

// Topics
export const fetchTopics = async (params?: FilterParams): Promise<PaginatedResponse<Topic>> => {
  const { data } = await api.get('/topics/', { params });
  return data;
};

export const fetchTopic = async (id: number): Promise<Topic> => {
  const { data } = await api.get(`/topics/${id}/`);
  return data;
};

export const fetchTrendingTopics = async (params?: FilterParams): Promise<TrendingTopic[]> => {
  const { data } = await api.get('/topics/trending/', { params });
  return data;
};

export const fetchTopicTimeline = async (
  id: number,
  params?: { days?: number }
): Promise<TopicDailyMetric[]> => {
  const { data } = await api.get(`/topics/${id}/timeline/`, { params });
  return data;
};

export const fetchTopicCategories = async (): Promise<
  Array<{ category: string; display_name: string; count: number }>
> => {
  const { data } = await api.get('/topics/categories/');
  return data;
};

// Posts
export const fetchPosts = async (params?: FilterParams): Promise<PaginatedResponse<Post>> => {
  const { data } = await api.get('/posts/', { params });
  return data;
};

export const fetchPost = async (id: number): Promise<Post> => {
  const { data } = await api.get(`/posts/${id}/`);
  return data;
};

export const fetchTopPosts = async (params?: FilterParams): Promise<Post[]> => {
  const { data } = await api.get('/posts/top/', { params });
  return data;
};

export const fetchPostFeed = async (params?: FilterParams): Promise<Post[]> => {
  const { data } = await api.get('/posts/feed/', { params });
  return data;
};

// Entities
export const fetchEntities = async (params?: FilterParams): Promise<PaginatedResponse<Entity>> => {
  const { data } = await api.get('/entities/', { params });
  return data;
};

export const fetchEntity = async (id: number): Promise<Entity> => {
  const { data } = await api.get(`/entities/${id}/`);
  return data;
};

export const fetchEntityNodes = async (
  params?: FilterParams
): Promise<PaginatedResponse<EntityNode>> => {
  const { data } = await api.get('/entity-nodes/', { params });
  return data;
};

// Metrics
export const fetchMetrics = async (
  params?: FilterParams
): Promise<PaginatedResponse<TopicDailyMetric>> => {
  const { data } = await api.get('/metrics/', { params });
  return data;
};

export const fetchMetricsHeatmap = async (params?: {
  days?: number;
  topic_id?: number;
}): Promise<
  Array<{
    topic_name: string;
    date: string;
    momentum_score: number;
  }>
> => {
  const { data } = await api.get('/metrics/heatmap/', { params });
  return data;
};

// Stats
export const fetchStats = async (): Promise<FeedStats> => {
  const { data } = await api.get('/stats/overview/');
  return data;
};

export const fetchSentimentAnalysis = async (params?: {
  days?: number;
}): Promise<SentimentAnalysis> => {
  const { data } = await api.get('/stats/sentiment_analysis/', { params });
  return data;
};

export const refreshData = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/stats/refresh_data/');
  return data;
};

export const recalculateMetrics = async (): Promise<{ status: string; message: string }> => {
  const { data } = await api.post('/stats/recalculate_metrics/');
  return data;
};

export default api;