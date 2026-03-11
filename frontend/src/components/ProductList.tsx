import {
  List,
  ListItem,
  Typography,
  Box,
  Link as MuiLink,
  Skeleton,
  Stack,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import GitHubIcon from '@mui/icons-material/GitHub';
import RedditIcon from '@mui/icons-material/Reddit';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { formatDistanceToNow } from 'date-fns';
import { ArrowUp, MessageSquare, Clock } from 'lucide-react';
import TopicTags from './TopicTags';
import SourceIndicator from './SourceIndicator';
import type { Post } from '../services/api';

interface ProductListProps {
  products: Post[];
  isLoading: boolean;
}

const sourceConfig: Record<string, { icon: any; color: string; label: string }> = {
  HN: { icon: TrendingUpIcon, color: '#ff6600', label: 'Hacker News' },
  GITHUB: { icon: GitHubIcon, color: '#24292e', label: 'GitHub' },
  GITHUB_TRENDING: { icon: GitHubIcon, color: '#24292e', label: 'GitHub Trending' },
  REDDIT_LOCALLLAMA: { icon: RedditIcon, color: '#ff4500', label: 'r/LocalLLaMA' },
  REDDIT_MACHINELEARNING: { icon: RedditIcon, color: '#ff4500', label: 'r/MachineLearning' },
  REDDIT_PROGRAMMING: { icon: RedditIcon, color: '#ff4500', label: 'r/programming' },
  PRODUCT_HUNT: { icon: TrendingUpIcon, color: '#da552f', label: 'Product Hunt' },
  DEVTO: { icon: TrendingUpIcon, color: '#0a0a0a', label: 'Dev.to' },
  LOBSTERS: { icon: TrendingUpIcon, color: '#990000', label: 'Lobste.rs' },
  TAAFT: { icon: TrendingUpIcon, color: '#6366f1', label: 'TAAFT' },
};

const ProductList = ({ products, isLoading }: ProductListProps) => {
  if (isLoading) {
    return (
      <Box sx={{ p: 2.5 }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Box key={i} sx={{ mb: 2, p: 2.5, borderRadius: 5, bgcolor: 'background.default' }}>
            <Skeleton variant="text" width="80%" height={32} />
            <Skeleton variant="text" width="60%" height={24} />
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Skeleton variant="rectangular" width={80} height={24} />
              <Skeleton variant="rectangular" width={80} height={24} />
              <Skeleton variant="rectangular" width={80} height={24} />
            </Box>
          </Box>
        ))}
      </Box>
    );
  }

  if (products.length === 0) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No trending products found. Try adjusting your filters.
        </Typography>
      </Box>
    );
  }

  const getEngagementColor = (score: number): string => {
    if (score > 500) return '#f44336'; // Very high engagement - red
    if (score > 200) return '#ff9800'; // High engagement - orange
    if (score > 50) return '#4caf50'; // Good engagement - green
    return '#2196f3'; // Normal engagement - blue
  };

  return (
    <List disablePadding sx={{ p: 1.5 }}>
      {products.map((product, index) => {
        const config = sourceConfig[product.source];

        return (
          <ListItem
            key={product.id}
            sx={{
              flexDirection: 'column',
              alignItems: 'flex-start',
              border: '1px solid',
              borderColor: 'transparent',
              borderRadius: 3,
              py: 3,
              px: 3,
              mb: 2,
              bgcolor: '#FFFFFF',
              boxShadow: '0 1px 3px rgba(0,0,0,0.02)',
              transition: 'transform 0.2s, box-shadow 0.2s',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
              }
            }}
          >
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ width: '100%' }}>
              <Box
                sx={{
                  minWidth: { xs: 'auto', md: 72 },
                  color: 'text.secondary',
                  fontWeight: 700,
                  fontSize: '0.95rem',
                }}
              >
                #{index + 1}
              </Box>

              <Box sx={{ width: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', width: '100%', mb: 1, gap: 1, flexWrap: 'wrap' }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <MuiLink
                      href={product.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      underline="hover"
                      sx={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 0.5,
                        fontSize: '1.08rem',
                        fontWeight: 600,
                        color: 'text.primary',
                      }}
                    >
                      {product.title}
                      <OpenInNewIcon sx={{ fontSize: 16, opacity: 0.7 }} />
                    </MuiLink>

                    {product.author && (
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
                        by {product.author}
                      </Typography>
                    )}
                  </Box>

                  <SourceIndicator
                    source={product.source}
                    sourceDisplay={product.source_display}
                    config={config}
                  />
                </Box>

                {product.content && (
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      mb: 1,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      display: '-webkit-box',
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: 'vertical',
                    }}
                  >
                    {product.content}
                  </Typography>
                )}

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2.5, mb: 1.5, alignItems: 'center' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, color: getEngagementColor(product.engagement_score) }}>
                    <ArrowUp size={16} strokeWidth={2.5} />
                    <Typography variant="body2" sx={{ fontWeight: 600, fontSize: '0.875rem' }}>
                      {product.engagement_score.toLocaleString()}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, color: 'text.secondary', opacity: 0.85 }}>
                    <MessageSquare size={16} strokeWidth={2} />
                    <Typography variant="body2" sx={{ fontWeight: 500, fontSize: '0.875rem' }}>
                      {product.comment_count}
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, color: 'text.secondary', opacity: 0.75 }}>
                    <Clock size={14} strokeWidth={2} />
                    <Typography variant="body2" sx={{ fontSize: '0.85rem' }}>
                      {formatDistanceToNow(new Date(product.published_at), { addSuffix: true })}
                    </Typography>
                  </Box>
                </Box>

                {product.topic_names && product.topic_names.length > 0 && (
                  <TopicTags topics={product.topic_names} />
                )}
              </Box>
            </Stack>
          </ListItem>
        );
      })}
    </List>
  );
};

export default ProductList;