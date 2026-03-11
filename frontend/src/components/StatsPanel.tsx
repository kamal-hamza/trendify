import { Paper, Typography, Box, Skeleton } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import type { FeedStats } from '../services/api';

interface StatsPanelProps {
  stats?: FeedStats;
  isLoading: boolean;
}

const StatsPanel = ({ stats, isLoading }: StatsPanelProps) => {
  if (isLoading || !stats) {
    return (
      <Paper sx={{ p: 2, height: '100%' }}>
        <Skeleton variant="text" width="60%" height={24} sx={{ mb: 1 }} />
        <Skeleton variant="rectangular" height={100} sx={{ mb: 2, borderRadius: 2 }} />
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
          <Skeleton variant="rectangular" height={60} sx={{ borderRadius: 2 }} />
          <Skeleton variant="rectangular" height={60} sx={{ borderRadius: 2 }} />
        </Box>
      </Paper>
    );
  }

  const sentimentColor =
    stats.avg_sentiment > 0.05 ? '#10B981' :
      stats.avg_sentiment < -0.05 ? '#EF4444' :
        '#64748B';

  const sentimentLabel =
    stats.avg_sentiment > 0.05 ? 'Positive' :
      stats.avg_sentiment < -0.05 ? 'Negative' :
        'Neutral';

  // Create data for a simple gauge/pie chart to visualize sentiment (-1 to 1 mapped to 0-100)
  const normalizedSentiment = ((stats.avg_sentiment + 1) / 2) * 100;
  const sentimentData = [
    { name: 'Sentiment', value: normalizedSentiment },
    { name: 'Remaining', value: 100 - normalizedSentiment }
  ];

  return (
    <Paper sx={{ p: 2.5, height: '100%' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>
        Overview
      </Typography>

      {/* Sentiment Chart Section */}
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        p: 2,
        mb: 2,
        borderRadius: 2,
        bgcolor: '#F8FAFC',
        border: '1px solid rgba(0,0,0,0.04)'
      }}>
        <Box sx={{ width: 80, height: 80, position: 'relative', mr: 2 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={sentimentData}
                cx="50%"
                cy="50%"
                innerRadius={25}
                outerRadius={35}
                startAngle={180}
                endAngle={0}
                dataKey="value"
                stroke="none"
              >
                <Cell fill={sentimentColor} />
                <Cell fill="#E2E8F0" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <Box
            sx={{
              position: 'absolute',
              top: '55%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              textAlign: 'center'
            }}
          >
            <Typography variant="caption" sx={{ fontWeight: 700, color: sentimentColor }}>
              {stats.avg_sentiment > 0 ? '+' : ''}{stats.avg_sentiment.toFixed(2)}
            </Typography>
          </Box>
        </Box>
        <Box>
          <Typography variant="body2" color="text.secondary">
            Average Sentiment
          </Typography>
          <Typography variant="h6" sx={{ color: sentimentColor, fontWeight: 700, lineHeight: 1.2 }}>
            {sentimentLabel}
          </Typography>
        </Box>
      </Box>

      {/* Metrics Section */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {[
          { label: 'Total posts', value: stats.total_posts.toLocaleString(), tone: 'text.primary' },
          { label: 'Total topics', value: stats.total_topics.toLocaleString(), tone: 'text.primary' },
          { label: 'Active topics', value: stats.total_active_topics.toLocaleString(), tone: 'primary.main' },
        ].map((item, idx) => (
          <Box
            key={item.label}
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              py: 1,
              borderBottom: idx < 2 ? '1px solid rgba(0,0,0,0.04)' : 'none'
            }}
          >
            <Typography variant="body2" color="text.secondary">
              {item.label}
            </Typography>
            <Typography variant="body2" sx={{ color: item.tone, fontWeight: 600 }}>
              {item.value}
            </Typography>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default StatsPanel;
