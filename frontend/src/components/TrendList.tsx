import {
  List,
  ListItem,
  Paper,
  Typography,
  Box,
  Chip,
  Tooltip,
  Skeleton,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import GitHubIcon from '@mui/icons-material/GitHub';
import RedditIcon from '@mui/icons-material/Reddit';
import { formatDistanceToNow } from 'date-fns';
import type { TrendingTopic } from '../services/api';

interface TrendListProps {
  trends: TrendingTopic[];
  isLoading: boolean;
}

const sourceConfig: Record<string, { icon: any; color: string; label: string }> = {
  HN: { icon: TrendingUpIcon, color: '#ff6600', label: 'HN' },
  GITHUB: { icon: GitHubIcon, color: '#24292e', label: 'GitHub' },
  REDDIT_LOCALLLAMA: { icon: RedditIcon, color: '#ff4500', label: 'r/LocalLLaMA' },
  REDDIT_MACHINELEARNING: { icon: RedditIcon, color: '#ff4500', label: 'r/ML' },
  REDDIT_PROGRAMMING: { icon: RedditIcon, color: '#ff4500', label: 'r/programming' },
};

const TrendList = ({ trends, isLoading }: TrendListProps) => {
  if (isLoading) {
    return (
      <Paper sx={{ p: 2 }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Box key={i} sx={{ mb: 2 }}>
            <Skeleton variant="text" width="60%" height={32} />
            <Skeleton variant="text" width="40%" height={24} />
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

  if (trends.length === 0) {
    return (
      <Paper sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No trending topics found. Try adjusting your filters.
        </Typography>
      </Paper>
    );
  }

  const getMomentumColor = (score: number): string => {
    if (score > 5) return '#f44336'; // High momentum - red
    if (score > 2) return '#ff9800'; // Medium momentum - orange
    if (score > 0) return '#4caf50'; // Positive momentum - green
    return '#9e9e9e'; // No momentum - grey
  };

  const getSentimentColor = (sentiment: number): string => {
    if (sentiment > 0.3) return '#4caf50';
    if (sentiment < -0.3) return '#f44336';
    return '#757575';
  };

  return (
    <Paper sx={{ p: 2 }}>
      <List disablePadding>
        {trends.map((trend, index) => (
          <ListItem
            key={trend.id}
            sx={{
              flexDirection: 'column',
              alignItems: 'flex-start',
              borderBottom: index < trends.length - 1 ? '1px solid' : 'none',
              borderColor: 'divider',
              py: 2,
              px: 0,
            }}
          >
            {/* Header Row */}
            <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 1 }}>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                {trend.name}
              </Typography>
              <Chip
                label={trend.category_display}
                size="small"
                variant="outlined"
                sx={{ ml: 1 }}
              />
            </Box>

            {/* Metrics Row */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
              <Chip
                icon={<TrendingUpIcon />}
                label={`Momentum: ${trend.momentum_score.toFixed(1)}`}
                size="small"
                sx={{
                  backgroundColor: getMomentumColor(trend.momentum_score),
                  color: 'white',
                  '& .MuiChip-icon': { color: 'white' },
                }}
              />
              <Chip
                label={`${trend.total_mentions} mentions`}
                size="small"
                variant="outlined"
              />
              <Chip
                label={`${trend.total_engagement.toLocaleString()} engagement`}
                size="small"
                variant="outlined"
              />
              <Chip
                label={`Sentiment: ${trend.avg_sentiment.toFixed(2)}`}
                size="small"
                sx={{
                  borderColor: getSentimentColor(trend.avg_sentiment),
                  color: getSentimentColor(trend.avg_sentiment),
                }}
                variant="outlined"
              />
            </Box>

            {/* Sources Row */}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
                Sources:
              </Typography>
              {Object.entries(trend.sources).map(([source, count]) => {
                const config = sourceConfig[source];
                if (!config) return null;
                
                const SourceIcon = config.icon;
                return (
                  <Tooltip key={source} title={`${config.label}: ${count} posts`}>
                    <Chip
                      icon={<SourceIcon />}
                      label={count}
                      size="small"
                      variant="outlined"
                      sx={{ height: 20, fontSize: '0.7rem' }}
                    />
                  </Tooltip>
                );
              })}
            </Box>

            {/* Peak Date */}
            {trend.peak_date && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                Peak: {formatDistanceToNow(new Date(trend.peak_date), { addSuffix: true })}
              </Typography>
            )}
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default TrendList;
