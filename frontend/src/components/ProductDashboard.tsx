import { useState } from 'react';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Paper,
  Stack,
  IconButton,
  Typography,
  Chip,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import { fetchTopPosts, fetchStats, refreshData } from '../services/api';
import ProductList from './ProductList';
import StatsPanel from './StatsPanel';
import { format, subDays, addDays, parseISO } from 'date-fns';

const ProductDashboard = () => {
  const queryClient = useQueryClient();
  const [refreshing, setRefreshing] = useState(false);
  const [sortBy, setSortBy] = useState<'engagement' | 'recency' | 'momentum' | 'growth'>('engagement');

  // Use specific date instead of days range (default to March 9, 2026 - the latest date with data)
  const [selectedDate, setSelectedDate] = useState('2026-03-09');
  const [localSource, setLocalSource] = useState('all');

  // Build query params - fetch posts from selected date only
  const buildParams = (): { limit: number; date: string; source?: string } => {
    const params: { limit: number; date: string; source?: string } = {
      limit: 100,
      date: selectedDate,
    };
    if (localSource !== 'all') params.source = localSource;
    return params;
  };

  // Date navigation helpers
  const goToPreviousDay = () => {
    const currentDate = parseISO(selectedDate);
    const prevDay = subDays(currentDate, 1);
    setSelectedDate(format(prevDay, 'yyyy-MM-dd'));
  };

  const goToNextDay = () => {
    const currentDate = parseISO(selectedDate);
    const nextDay = addDays(currentDate, 1);
    const today = new Date();
    // Don't go beyond today
    if (nextDay <= today) {
      setSelectedDate(format(nextDay, 'yyyy-MM-dd'));
    }
  };

  // Fetch top products/posts
  const {
    data: products = [],
    isLoading: productsLoading,
    error: productsError,
  } = useQuery({
    queryKey: ['posts', 'top', localSource, selectedDate, sortBy],
    queryFn: () => fetchTopPosts(buildParams()),
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
    refetchInterval: 60000,
  });

  // Refresh mutation
  const refreshMutation = useMutation({
    mutationFn: refreshData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['posts'] });
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

  // Sort products based on selected criteria
  const sortedProducts = [...products].sort((a, b) => {
    if (sortBy === 'engagement') {
      return b.engagement_score - a.engagement_score;
    } else if (sortBy === 'recency') {
      return new Date(b.published_at).getTime() - new Date(a.published_at).getTime();
    } else if (sortBy === 'momentum') {
      const aMomentum = a.velocity_metrics?.momentum_score ?? -Infinity;
      const bMomentum = b.velocity_metrics?.momentum_score ?? -Infinity;
      return bMomentum - aMomentum;
    } else if (sortBy === 'growth') {
      const aGrowth = a.velocity_metrics?.growth_rate ?? -Infinity;
      const bGrowth = b.velocity_metrics?.growth_rate ?? -Infinity;
      return bGrowth - aGrowth;
    }
    return 0;
  });

  return (
    <Stack spacing={3}>
      <Paper
        sx={{
          p: 3,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          spacing={2}
          alignItems={{ xs: 'flex-start', md: 'center' }}
          justifyContent="space-between"
        >
          <Box>
            <Typography variant="overline" color="text.secondary">
              Product discovery
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
                <RocketLaunchIcon />
              </Box>
              <Typography variant="h3" component="h1">
                Product dashboard
              </Typography>
            </Stack>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
              {format(parseISO(selectedDate), 'MMMM d, yyyy')} · {sortedProducts.length} items · {
                sortBy === 'engagement' ? 'sorted by engagement' :
                sortBy === 'recency' ? 'sorted by recency' :
                sortBy === 'momentum' ? 'sorted by momentum' :
                'sorted by growth rate'
              }{localSource !== 'all' ? ` · ${localSource}` : ''}
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

      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'grid', gap: 2, gridTemplateColumns: { xs: '1fr', lg: 'repeat(3, minmax(0, 1fr))' } }}>
          {/* Source Filter */}
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="body2" color="text.secondary" sx={{ display: 'block', mb: 1.25 }}>
              Source
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {[
                { value: 'all', label: 'All' },
                { value: 'PRODUCT_HUNT', label: 'Product Hunt' },
                { value: 'HN', label: 'Hacker News' },
                { value: 'DEVTO', label: 'Dev.to' },
                { value: 'GITHUB_TRENDING', label: 'GitHub' },
                { value: 'LOBSTERS', label: 'Lobsters' },
                { value: 'TAAFT', label: 'AI Tools' },
              ].map(s => (
                <Chip
                  key={s.value}
                  label={s.label}
                  onClick={() => setLocalSource(s.value)}
                  clickable
                  sx={{
                    borderRadius: 1.5,
                    bgcolor: localSource === s.value ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                    color: localSource === s.value ? 'secondary.main' : 'text.secondary',
                    border: '1px solid',
                    borderColor: localSource === s.value ? 'rgba(99, 102, 241, 0.3)' : 'divider',
                    fontWeight: localSource === s.value ? 600 : 500,
                    transition: 'all 0.2s',
                    '&:hover': {
                      bgcolor: localSource === s.value ? 'rgba(99, 102, 241, 0.15)' : 'action.hover',
                    }
                  }}
                />
              ))}
            </Box>
          </Box>

          {/* Date Navigation */}
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="body2" color="text.secondary" sx={{ display: 'block', mb: 1.25 }}>
              Date
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0, flexWrap: 'wrap' }}>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0,
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 2.5,
                  overflow: 'hidden',
                  bgcolor: 'background.paper',
                }}
              >
                <IconButton
                  size="small"
                  onClick={goToPreviousDay}
                  title="Previous day"
                  sx={{
                    borderRadius: 0,
                    borderRight: '1px solid',
                    borderColor: 'divider',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                >
                  <NavigateBeforeIcon fontSize="small" />
                </IconButton>
                <Typography
                  variant="body2"
                  sx={{
                    minWidth: 156,
                    textAlign: 'center',
                    fontWeight: 500,
                    userSelect: 'none',
                    px: 2,
                    py: 0.5
                  }}
                >
                  {format(parseISO(selectedDate), 'MMMM d, yyyy')}
                </Typography>
                <IconButton
                  size="small"
                  onClick={goToNextDay}
                  disabled={selectedDate >= '2026-03-09'}
                  title="Next day"
                  sx={{
                    borderRadius: 0,
                    borderLeft: '1px solid',
                    borderColor: 'divider',
                    '&:hover': { backgroundColor: 'action.hover' }
                  }}
                >
                  <NavigateNextIcon fontSize="small" />
                </IconButton>
              </Box>
              <Button
                size="small"
                variant="text"
                onClick={() => setSelectedDate('2026-03-09')}
                disabled={selectedDate === '2026-03-09'}
                sx={{ minWidth: 'auto', px: 1.5 }}
              >
                Latest
              </Button>
            </Box>
          </Box>

          {/* Sort By */}
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="body2" color="text.secondary" sx={{ display: 'block', mb: 1.25 }}>
              Sort By
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip
                icon={<TrendingUpIcon sx={{ fontSize: '16px !important' }} />}
                label="Trending"
                onClick={() => setSortBy('engagement')}
                clickable
                sx={{
                  borderRadius: 1.5,
                  bgcolor: sortBy === 'engagement' ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                  color: sortBy === 'engagement' ? 'secondary.main' : 'text.secondary',
                  border: '1px solid',
                  borderColor: sortBy === 'engagement' ? 'rgba(99, 102, 241, 0.3)' : 'divider',
                  fontWeight: sortBy === 'engagement' ? 600 : 500,
                  '& .MuiChip-icon': {
                    color: sortBy === 'engagement' ? 'secondary.main' : 'text.secondary',
                  },
                  transition: 'all 0.2s',
                }}
              />
              <Chip
                icon={<NewReleasesIcon sx={{ fontSize: '16px !important' }} />}
                label="Latest"
                onClick={() => setSortBy('recency')}
                clickable
                sx={{
                  borderRadius: 1.5,
                  bgcolor: sortBy === 'recency' ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                  color: sortBy === 'recency' ? 'secondary.main' : 'text.secondary',
                  border: '1px solid',
                  borderColor: sortBy === 'recency' ? 'rgba(99, 102, 241, 0.3)' : 'divider',
                  fontWeight: sortBy === 'recency' ? 600 : 500,
                  '& .MuiChip-icon': {
                    color: sortBy === 'recency' ? 'secondary.main' : 'text.secondary',
                  },
                  transition: 'all 0.2s',
                }}
              />
              <Chip
                icon={<RocketLaunchIcon sx={{ fontSize: '16px !important' }} />}
                label="Momentum"
                onClick={() => setSortBy('momentum')}
                clickable
                sx={{
                  borderRadius: 1.5,
                  bgcolor: sortBy === 'momentum' ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                  color: sortBy === 'momentum' ? 'secondary.main' : 'text.secondary',
                  border: '1px solid',
                  borderColor: sortBy === 'momentum' ? 'rgba(99, 102, 241, 0.3)' : 'divider',
                  fontWeight: sortBy === 'momentum' ? 600 : 500,
                  '& .MuiChip-icon': {
                    color: sortBy === 'momentum' ? 'secondary.main' : 'text.secondary',
                  },
                  transition: 'all 0.2s',
                }}
              />
              <Chip
                icon={<TrendingUpIcon sx={{ fontSize: '16px !important' }} />}
                label="Growth"
                onClick={() => setSortBy('growth')}
                clickable
                sx={{
                  borderRadius: 1.5,
                  bgcolor: sortBy === 'growth' ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                  color: sortBy === 'growth' ? 'secondary.main' : 'text.secondary',
                  border: '1px solid',
                  borderColor: sortBy === 'growth' ? 'rgba(99, 102, 241, 0.3)' : 'divider',
                  fontWeight: sortBy === 'growth' ? 600 : 500,
                  '& .MuiChip-icon': {
                    color: sortBy === 'growth' ? 'secondary.main' : 'text.secondary',
                  },
                  transition: 'all 0.2s',
                }}
              />
            </Box>
          </Box>
        </Box>
      </Paper>

      {productsError && (
        <Alert severity="error">
          Failed to load products: {(productsError as Error).message}
        </Alert>
      )}
      {statsError && (
        <Alert severity="error">
          Failed to load statistics: {(statsError as Error).message}
        </Alert>
      )}

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '320px minmax(0, 1fr)' }, gap: 3, alignItems: 'start' }}>
        <Box sx={{ position: { xl: 'sticky' }, top: { xl: 104 } }}>
          <StatsPanel stats={stats} isLoading={statsLoading} />
        </Box>

        <Paper sx={{ overflow: 'hidden' }}>
          <Box sx={{ p: 3, borderBottom: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2, flexWrap: 'wrap' }}>
              <Box>
                <Typography variant="h5" gutterBottom>
                  Products from {format(parseISO(selectedDate), 'MMMM d, yyyy')}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {sortBy === 'engagement'
                    ? 'Sorted by engagement score (upvotes + comments)'
                    : sortBy === 'recency'
                    ? 'Sorted by publish date (newest first)'
                    : sortBy === 'momentum'
                    ? 'Sorted by momentum score (velocity of mentions)'
                    : 'Sorted by growth rate (acceleration percentage)'}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {sortedProducts.length} {sortedProducts.length === 1 ? 'product' : 'products'}
              </Typography>
            </Box>
          </Box>
          <ProductList products={sortedProducts} isLoading={productsLoading} />
        </Paper>
      </Box>
    </Stack>
  );
};

export default ProductDashboard;