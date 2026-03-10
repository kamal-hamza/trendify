import {
  List,
  ListItem,
  Paper,
  Typography,
  Box,
  Chip,
  Link as MuiLink,
  Skeleton,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import GitHubIcon from '@mui/icons-material/GitHub';
import RedditIcon from '@mui/icons-material/Reddit';
import CommentIcon from '@mui/icons-material/Comment';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { formatDistanceToNow } from 'date-fns';
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
      <Paper sx={{ p: 2 }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Box key={i} sx={{ mb: 2 }}>
            <Skeleton variant="text" width="80%" height={32} />
            <Skeleton variant="text" width="60%" height={24} />
            <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
              <Skeleton variant="rectangular" width={80} height={24} />
              <Skeleton variant="rectangular" width={80} height={24} />
              <Skeleton variant="rectangular" width={80} height={24} />
            </Box>
          </Box>
        ))}
      </Paper>
    );
  }

  if (products.length === 0) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No trending products found. Try adjusting your filters.
        </Typography>
      </Paper>
    );
  }

  const getEngagementColor = (score: number): string => {
    if (score > 500) return '#f44336'; // Very high engagement - red
    if (score > 200) return '#ff9800'; // High engagement - orange
    if (score > 50) return '#4caf50'; // Good engagement - green
    return '#2196f3'; // Normal engagement - blue
  };

  return (
    <Paper sx={{ p: 2 }}>
      <List disablePadding>
        {products.map((product, index) => {
          const config = sourceConfig[product.source];
          const SourceIcon = config?.icon || TrendingUpIcon;

          return (
            <ListItem
              key={product.id}
              sx={{
                flexDirection: 'column',
                alignItems: 'flex-start',
                borderBottom: index < products.length - 1 ? '1px solid' : 'none',
                borderColor: 'divider',
                py: 2.5,
                px: 0,
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              {/* Header Row - Title with Link */}
              <Box sx={{ display: 'flex', alignItems: 'flex-start', width: '100%', mb: 1 }}>
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
                      fontSize: '1.1rem',
                      fontWeight: 600,
                      color: 'primary.main',
                      '&:hover': {
                        color: 'primary.dark',
                      },
                    }}
                  >
                    {product.title}
                    <OpenInNewIcon sx={{ fontSize: 16, opacity: 0.7 }} />
                  </MuiLink>
                  
                  {/* Author */}
                  {product.author && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                      by {product.author}
                    </Typography>
                  )}
                </Box>

                {/* Source Badge */}
                <Chip
                  icon={<SourceIcon />}
                  label={config?.label || product.source_display}
                  size="small"
                  variant="outlined"
                  sx={{
                    borderColor: config?.color,
                    color: config?.color,
                    '& .MuiChip-icon': { color: config?.color },
                  }}
                />
              </Box>

              {/* Description/Content */}
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

              {/* Metrics Row */}
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                <Chip
                  icon={<ThumbUpIcon />}
                  label={`${product.engagement_score.toLocaleString()} pts`}
                  size="small"
                  sx={{
                    backgroundColor: getEngagementColor(product.engagement_score),
                    color: 'white',
                    fontWeight: 600,
                    '& .MuiChip-icon': { color: 'white' },
                  }}
                />
                
                <Chip
                  icon={<CommentIcon />}
                  label={`${product.comment_count} comments`}
                  size="small"
                  variant="outlined"
                />

                <Chip
                  label={formatDistanceToNow(new Date(product.published_at), { addSuffix: true })}
                  size="small"
                  variant="outlined"
                  color="default"
                />
              </Box>

              {/* Tags/Topics Row */}
              {product.topic_names && product.topic_names.length > 0 && (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {product.topic_names.slice(0, 8).map((tag) => (
                    <Chip
                      key={tag}
                      label={`#${tag}`}
                      size="small"
                      variant="outlined"
                      sx={{
                        height: 22,
                        fontSize: '0.7rem',
                        borderRadius: 1,
                        backgroundColor: 'action.hover',
                        '&:hover': {
                          backgroundColor: 'action.selected',
                        },
                      }}
                    />
                  ))}
                  {product.topic_names.length > 8 && (
                    <Chip
                      label={`+${product.topic_names.length - 8} more`}
                      size="small"
                      variant="outlined"
                      sx={{
                        height: 22,
                        fontSize: '0.7rem',
                        borderRadius: 1,
                      }}
                    />
                  )}
                </Box>
              )}
            </ListItem>
          );
        })}
      </List>
    </Paper>
  );
};

export default ProductList;