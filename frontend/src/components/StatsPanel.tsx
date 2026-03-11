import { Paper, Typography, Box, Skeleton, Chip, Stack } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import type { FeedStats } from '../services/api';

const COLORS: Record<string, string> = {
  HN: '#005ac2',
  REDDIT_LOCALLLAMA: '#b3261e',
  REDDIT_MACHINELEARNING: '#8e24aa',
  REDDIT_PROGRAMMING: '#00897b',
  GITHUB: '#4f378b',
  PRODUCT_HUNT: '#c24e00',
  DEVTO: '#006875',
  LOBSTERS: '#7a2e0b',
  TAAFT: '#146c2e',
  TWITTER: '#00639b',
  OTHER: '#7d5260',
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

  const sortedSources = Object.entries(stats.sources_breakdown).sort(([, countA], [, countB]) => countB - countA);
  const mainSources = sortedSources.slice(0, 4);
  const otherCount = sortedSources.slice(4).reduce((sum, [, count]) => sum + count, 0);
  const chartData = [
    ...mainSources.map(([source, count]) => ({
      name: source,
      value: count,
      color: COLORS[source] || COLORS.OTHER,
    })),
    ...(otherCount > 0 ? [{ name: 'OTHER', value: otherCount, color: COLORS.OTHER }] : []),
  ];
  const totalSources = chartData.reduce((sum, item) => sum + item.value, 0);

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
      <Typography variant="overline" color="text.secondary">
        Overview
      </Typography>
      <Typography variant="h5" sx={{ mt: 0.5 }}>
        Dataset summary
      </Typography>

      {chartData.length > 0 && (
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ my: 3 }}>
          <Box sx={{ height: 210, minWidth: { sm: 190 }, flex: { sm: '0 0 190px' }, p: 1.5, borderRadius: 3, bgcolor: '#f8f2ff' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={42}
                  outerRadius={72}
                  paddingAngle={2}
                  stroke="none"
                  dataKey="value"
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => Number(value ?? 0).toLocaleString()} />
              </PieChart>
            </ResponsiveContainer>
          </Box>

          <Stack spacing={1.25} sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Sources are grouped to keep the summary readable.
            </Typography>
            {chartData.map((item) => {
              const percent = totalSources > 0 ? Math.round((item.value / totalSources) * 100) : 0;
              const label = item.name === 'OTHER' ? 'Other sources' : item.name.replaceAll('_', ' ');

              return (
                <Box key={item.name}>
                  <Stack direction="row" justifyContent="space-between" spacing={2} sx={{ mb: 0.5 }}>
                    <Typography variant="body2" fontWeight={600}>
                      {label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {percent}%
                    </Typography>
                  </Stack>
                  <Box sx={{ height: 8, borderRadius: 999, bgcolor: '#e7e0ec', overflow: 'hidden' }}>
                    <Box sx={{ width: `${percent}%`, height: '100%', bgcolor: item.color }} />
                  </Box>
                </Box>
              );
            })}
          </Stack>
        </Stack>
      )}

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

      {stats.top_categories && stats.top_categories.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Top categories
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

      {stats.date_range && (
        <Stack spacing={0.5} sx={{ pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
          <Typography variant="caption" color="text.secondary" display="block">
            Data range
          </Typography>
          <Typography variant="body2">
            {new Date(stats.date_range.earliest).toLocaleDateString()} 
            {' - '}
            {new Date(stats.date_range.latest).toLocaleDateString()}
          </Typography>
        </Stack>
      )}
    </Paper>
  );
};

export default StatsPanel;
