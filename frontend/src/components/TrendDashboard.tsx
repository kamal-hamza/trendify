import { useState } from 'react';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { fetchTrendingTopics, fetchEmergingTopics, fetchStats, refreshData } from '../services/api';
import { useFilterStore } from '../store/filterStore';
import FilterBar from './FilterBar';
import TrendList from './TrendList';
import StatsPanel from './StatsPanel';

const TrendDashboard = () => {
  const queryClient = useQueryClient();
  const { source, category, days, mode } = useFilterStore();
  const [refreshing, setRefreshing] = useState(false);

  // Build query params
  const buildParams = () => {
    const params: any = { days };
    if (source !== 'all') params.source = source;
    if (category !== 'all') params.category = category;
    return params;
  };

  // Fetch trending or emerging topics based on mode
  const {
    data: trends = [],
    isLoading: trendsLoading,
    error: trendsError,
  } = useQuery({
    queryKey: ['topics', mode, source, category, days],
    queryFn: () => mode === 'emerging' ? fetchEmergingTopics(buildParams()) : fetchTrendingTopics(buildParams()),
    refetchInterval: 60000, // Refetch every minute
  });

  // Fetch stats
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
  } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 60000, // Refetch every minute
  });

  // Refresh mutation
  const refreshMutation = useMutation({
    mutationFn: refreshData,
    onSuccess: () => {
      // Invalidate all queries to refetch data
      queryClient.invalidateQueries({ queryKey: ['topics'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
      setRefreshing(false);
    },
    onError: (error) => {
      console.error('Refresh failed:', error);
      setRefreshing(false);
    },
  });

  const handleRefresh = () => {
    setRefreshing(true);
    refreshMutation.mutate();
  };

  return (
    <Stack spacing={3}>
      <Paper
        sx={{
          p: { xs: 3, md: 4 },
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Box sx={{ position: 'absolute', inset: 0, borderTop: '4px solid', borderColor: 'primary.main', pointerEvents: 'none' }} />
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          spacing={2}
          alignItems={{ xs: 'flex-start', md: 'center' }}
          justifyContent="space-between"
        >
          <Box>
            <Typography variant="overline" color="text.secondary">
              Topic overview
            </Typography>
            <Stack direction="row" spacing={1.25} alignItems="center" sx={{ mt: 0.75 }}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 2,
                  display: 'grid',
                  placeItems: 'center',
                  bgcolor: 'primary.main',
                  color: 'primary.contrastText',
                }}
              >
                <TrendingUpIcon />
              </Box>
              <Typography variant="h3" component="h1">
                {mode === 'emerging' ? 'Emerging topics' : 'Trending topics'}
              </Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Tracks aggregate topic activity and momentum so category filters directly affect the ranked results below.
            </Typography>
          </Box>

          <Button
            variant="contained"
            startIcon={refreshing ? <CircularProgress size={20} color="inherit" /> : <RefreshIcon />}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? 'Refreshing' : 'Refresh data'}
          </Button>
        </Stack>
      </Paper>

      <StatsPanel stats={stats} isLoading={statsLoading} />

      <FilterBar />

      {trendsError && (
        <Alert severity="error">
          Failed to load trending topics: {(trendsError as Error).message}
        </Alert>
      )}
      {statsError && (
        <Alert severity="error">
          Failed to load statistics: {(statsError as Error).message}
        </Alert>
      )}

      <Paper sx={{ overflow: 'hidden' }}>
        <Box
          sx={{
            px: { xs: 2.5, md: 3.5 },
            py: 3,
            borderBottom: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            alignItems: { xs: 'flex-start', md: 'center' },
            justifyContent: 'space-between',
            gap: 2,
            flexWrap: 'wrap',
          }}
        >
          <Box>
            <Typography variant="h5">
              {mode === 'emerging' ? 'Emerging topic list' : 'Topic ranking'}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>
              {trends.length} results · {days} day window{source !== 'all' ? ` · ${source}` : ''}{category !== 'all' ? ` · ${category}` : ''}
            </Typography>
          </Box>

        </Box>
        <TrendList trends={trends} isLoading={trendsLoading} />
      </Paper>
    </Stack>
  );
};

export default TrendDashboard;
