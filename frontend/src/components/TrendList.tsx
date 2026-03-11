import {
  List,
  ListItem,
  Typography,
  Box,
  Chip,
  Tooltip,
  Skeleton,
  Stack,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import GitHubIcon from '@mui/icons-material/GitHub';
import RedditIcon from '@mui/icons-material/Reddit';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import { formatDistanceToNow } from 'date-fns';
import type { TrendingTopic, EmergingTopic } from '../services/api';

interface TrendListProps {
  trends: (TrendingTopic | EmergingTopic)[];
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
      <Box sx={{ p: 2.5 }}>
        {[1, 2, 3, 4, 5].map((i) => (
          <Box key={i} sx={{ mb: 2, p: 2.5, borderRadius: 5, bgcolor: 'background.default' }}>
            <Skeleton variant="text" width="60%" height={32} />
            <Skeleton variant="text" width="40%" height={24} />
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

  if (trends.length === 0) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No trending topics found. Try adjusting your filters.
        </Typography>
      </Box>
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

  const getGrowthColor = (rate: number): string => {
    if (rate > 2) return '#f44336'; // >200% growth - red
    if (rate > 1) return '#ff9800'; // >100% growth - orange
    if (rate > 0.5) return '#4caf50'; // >50% growth - green
    return '#757575';
  };

  const isEmergingTopic = (topic: TrendingTopic | EmergingTopic): topic is EmergingTopic => {
    return 'avg_growth_rate' in topic;
  };

  return (
      <List disablePadding sx={{ p: 1.5 }}>
        {trends.map((trend, index) => (
          <ListItem
            key={trend.id}
            sx={{
              flexDirection: 'column',
              alignItems: 'flex-start',
              border: '1px solid',
              borderColor: index < trends.length - 1 ? 'divider' : 'rgba(28, 27, 31, 0.12)',
              borderRadius: 3,
              py: 2.5,
              px: 2.5,
              mb: 1.5,
              bgcolor: '#fffbff',
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
                <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 1, gap: 1, flexWrap: 'wrap' }}>
                  <Typography variant="h6" component="div" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                    {trend.name}
                    {isEmergingTopic(trend) && trend.is_new && (
                      <Chip
                        icon={<NewReleasesIcon />}
                        label="NEW"
                        size="small"
                        sx={{ height: 22, fontSize: '0.72rem', fontWeight: 700, bgcolor: '#d6f8d3', color: '#146c2e' }}
                      />
                    )}
                  </Typography>
                  <Chip label={trend.category_display} size="small" variant="outlined" />
                </Box>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1.5 }}>
                  {isEmergingTopic(trend) ? (
                    <>
                      <Chip
                        icon={<RocketLaunchIcon />}
                        label={`Growth ${(trend.avg_growth_rate * 100).toFixed(0)}%`}
                        size="small"
                        sx={{
                          backgroundColor: getGrowthColor(trend.avg_growth_rate),
                          color: 'white',
                          '& .MuiChip-icon': { color: 'white' },
                        }}
                      />
                      <Chip label={`${trend.age_days} days old`} size="small" variant="outlined" />
                    </>
                  ) : (
                    <Chip
                      icon={<TrendingUpIcon />}
                      label={`Momentum ${trend.momentum_score.toFixed(1)}`}
                      size="small"
                      sx={{
                        backgroundColor: getMomentumColor(trend.momentum_score),
                        color: 'white',
                        '& .MuiChip-icon': { color: 'white' },
                      }}
                    />
                  )}
                  <Chip label={`${trend.total_mentions} mentions`} size="small" variant="outlined" />
                  <Chip label={`${trend.total_engagement.toLocaleString()} engagement`} size="small" variant="outlined" />
                  <Chip
                    label={`Sentiment ${trend.avg_sentiment.toFixed(2)}`}
                    size="small"
                    sx={{
                      borderColor: getSentimentColor(trend.avg_sentiment),
                      color: getSentimentColor(trend.avg_sentiment),
                    }}
                    variant="outlined"
                  />
                </Box>

                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {Object.entries(trend.sources).map(([source, count]) => {
                    const config = sourceConfig[source];
                    if (!config) return null;

                    const SourceIcon = config.icon;
                    return (
                      <Tooltip key={source} title={`${config.label}: ${count} posts`}>
                        <Chip
                          icon={<SourceIcon />}
                          label={`${config.label} · ${count}`}
                          size="small"
                          variant="outlined"
                          sx={{ height: 24, fontSize: '0.74rem' }}
                        />
                      </Tooltip>
                    );
                  })}
                </Box>

            {isEmergingTopic(trend) && trend.first_seen ? (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    First seen {formatDistanceToNow(new Date(trend.first_seen), { addSuffix: true })}
                  </Typography>
            ) : trend.peak_date ? (
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Peak {formatDistanceToNow(new Date(trend.peak_date), { addSuffix: true })}
                  </Typography>
            ) : null}
              </Box>
            </Stack>
          </ListItem>
        ))}
      </List>
  );
};

export default TrendList;
