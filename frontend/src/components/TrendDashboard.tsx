import { useState } from 'react';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Paper,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { fetchTrendingTopics, fetchStats, refreshData } from '../services/api';
import { useFilterStore } from '../store/filterStore';
import FilterBar from './FilterBar';
import TrendList from './TrendList';
import StatsPanel from './StatsPanel';

const TrendDashboard = () => {
  const queryClient = useQueryClient();
  const { source, category, days } = useFilterStore();
  const [refreshing, setRefreshing] = useState(false);

  // Build query params
  const buildParams = () => {
    const params: any = { days };
    if (source !== 'all') params.source = source;
    if (category !== 'all') params.category = category;
    return params;
  };

  // Fetch trending topics
  const {
    data: trends = [],
    isLoading: trendsLoading,
    error: trendsError,
  } = useQuery({
    queryKey: ['trending-topics', source, category, days],
    queryFn: () => fetchTrendingTopics(buildParams()),
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
      queryClient.invalidateQueries({ queryKey: ['trending-topics'] });
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
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 3,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <TrendingUpIcon sx={{ fontSize: 40, color: 'primary.main' }} />
          <Typography variant="h3" component="h1" fontWeight="bold">
            Trendify
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={refreshing ? <CircularProgress size={20} /> : <RefreshIcon />}
          onClick={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? 'Refreshing...' : 'Refresh Data'}
        </Button>
      </Box>

      {/* Subtitle */}
      <Typography variant="subtitle1" color="text.secondary" sx={{ mb: 3 }}>
        Real-time tech trend aggregation from GitHub, Hacker News, and Reddit
      </Typography>

      {/* Filters */}
      <FilterBar />

      {/* Error Alerts */}
      {trendsError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load trending topics: {(trendsError as Error).message}
        </Alert>
      )}
      {statsError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load statistics: {(statsError as Error).message}
        </Alert>
      )}

      {/* Main Content Grid */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 2fr' }, gap: 3 }}>
        {/* Stats Panel - Left Side */}
        <Box>
          <StatsPanel stats={stats} isLoading={statsLoading} />
        </Box>

        {/* Trending Topics List - Right Side */}
        <Box>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom fontWeight="bold">
              Trending Topics
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Sorted by momentum score (velocity of mentions over time)
            </Typography>
            <TrendList trends={trends} isLoading={trendsLoading} />
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default TrendDashboard;
