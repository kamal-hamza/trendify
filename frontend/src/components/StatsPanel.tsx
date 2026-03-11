import { Paper, Typography, Box, Skeleton } from '@mui/material';
import type { FeedStats } from '../services/api';

interface StatsPanelProps {
  stats?: FeedStats;
  isLoading: boolean;
}

const StatsPanel = ({ stats, isLoading }: StatsPanelProps) => {
  if (isLoading || !stats) {
    return (
      <Paper sx={{ p: 3, height: '100%' }}>
        <Skeleton variant="text" width="60%" height={32} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={200} />
        <Box sx={{ mt: 2 }}>
          <Skeleton variant="text" height={24} />
          <Skeleton variant="text" height={24} />
          <Skeleton variant="text" height={24} />
        </Box>
      </Paper>
    );
  }

  const sentimentColor = 
    stats.avg_sentiment > 0.3 ? '#4caf50' : 
    stats.avg_sentiment < -0.3 ? '#f44336' : 
    '#757575';

  const sentimentLabel =
    stats.avg_sentiment > 0.3 ? 'Positive' :
    stats.avg_sentiment < -0.3 ? 'Negative' :
    'Neutral';

  return (
    <Paper sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Overview
      </Typography>
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1.5, mb: 3 }}>
        {[
          { label: 'Total posts', value: stats.total_posts.toLocaleString(), tone: 'text.primary' },
          { label: 'Total topics', value: stats.total_topics.toLocaleString(), tone: 'text.primary' },
          { label: 'Active topics', value: stats.total_active_topics.toLocaleString(), tone: 'primary.main' },
          { label: 'Average sentiment', value: sentimentLabel, tone: sentimentColor, caption: `(${stats.avg_sentiment.toFixed(2)})` },
        ].map((item) => (
          <Box key={item.label} sx={{ p: 2, borderRadius: 3, bgcolor: '#f8f2ff' }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {item.label}
            </Typography>
            <Typography variant="h5" sx={{ color: item.tone, fontWeight: 700 }}>
              {item.value}
            </Typography>
            {'caption' in item && item.caption ? (
              <Typography variant="caption" color="text.secondary">
                {item.caption}
              </Typography>
            ) : null}
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default StatsPanel;
