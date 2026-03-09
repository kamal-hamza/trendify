import { Paper, Typography, Box, Skeleton, Chip } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import type { FeedStats } from '../services/api';

const COLORS: Record<string, string> = {
  HN: '#ff6600',
  REDDIT_LOCALLLAMA: '#ff4500',
  REDDIT_MACHINELEARNING: '#ff4500',
  REDDIT_PROGRAMMING: '#ff4500',
  GITHUB: '#24292e',
  TWITTER: '#1da1f2',
};

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

  // Prepare chart data from sources breakdown
  const chartData = Object.entries(stats.sources_breakdown).map(([source, count]) => ({
    name: source,
    value: count,
    color: COLORS[source] || '#999999',
  }));

  const sentimentColor = 
    stats.avg_sentiment > 0.3 ? '#4caf50' : 
    stats.avg_sentiment < -0.3 ? '#f44336' : 
    '#757575';

  const sentimentLabel =
    stats.avg_sentiment > 0.3 ? 'Positive' :
    stats.avg_sentiment < -0.3 ? 'Negative' :
    'Neutral';

  return (
    <Paper 
      elevation={2}
      sx={{ 
        p: 3, 
        height: '100%',
        borderRadius: 2,
      }}
    >
      <Typography variant="h6" gutterBottom fontWeight="bold">
        Overview
      </Typography>

      {/* Pie Chart */}
      {chartData.length > 0 && (
        <Box sx={{ height: 250, my: 3 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${((percent || 0) * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Stats Grid */}
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, mb: 3 }}>
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Total Posts
          </Typography>
          <Typography variant="h5" fontWeight="bold">
            {stats.total_posts.toLocaleString()}
          </Typography>
        </Box>

        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Total Topics
          </Typography>
          <Typography variant="h5" fontWeight="bold">
            {stats.total_topics.toLocaleString()}
          </Typography>
        </Box>

        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Active Topics
          </Typography>
          <Typography variant="h5" fontWeight="bold" color="primary.main">
            {stats.total_active_topics.toLocaleString()}
          </Typography>
        </Box>

        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Avg Sentiment
          </Typography>
          <Typography variant="h5" fontWeight="bold" color={sentimentColor}>
            {sentimentLabel}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            ({stats.avg_sentiment.toFixed(2)})
          </Typography>
        </Box>
      </Box>

      {/* Top Categories */}
      {stats.top_categories && stats.top_categories.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Top Categories
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            {stats.top_categories.slice(0, 5).map((cat) => (
              <Chip
                key={cat.category}
                label={`${cat.category}: ${cat.count}`}
                size="small"
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Date Range */}
      {stats.date_range && (
        <Box sx={{ pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Data Range
          </Typography>
          <Typography variant="body2">
            {new Date(stats.date_range.earliest).toLocaleDateString()} 
            {' - '}
            {new Date(stats.date_range.latest).toLocaleDateString()}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default StatsPanel;
