import { useState } from 'react';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Button,
  Alert,
  CircularProgress,
  Paper,
  ToggleButtonGroup,
  ToggleButton,
  TextField,
  IconButton,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import { fetchTopPosts, fetchStats, refreshData } from '../services/api';
import { useFilterStore } from '../store/filterStore';
import ProductList from './ProductList';
import StatsPanel from './StatsPanel';
import { format, subDays, addDays, parseISO } from 'date-fns';

const ProductDashboard = () => {
  const queryClient = useQueryClient();
  const { source, days } = useFilterStore();
  const [refreshing, setRefreshing] = useState(false);
  const [sortBy, setSortBy] = useState<'engagement' | 'recency'>('engagement');
  
  // Use specific date instead of days range (default to March 9, 2026 - the latest date with data)
  const [selectedDate, setSelectedDate] = useState('2026-03-09');
  const [localSource, setLocalSource] = useState('all');

  // Build query params - fetch posts from selected date only
  const buildParams = () => {
    const params: any = { 
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
    } else {
      return new Date(b.published_at).getTime() - new Date(a.published_at).getTime();
    }
  });

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
          <RocketLaunchIcon sx={{ fontSize: 40, color: 'primary.main' }} />
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
        Browse day-by-day trending products from Product Hunt, Hacker News, Dev.to, GitHub, Lobsters, and TAAFT
      </Typography>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
          {/* Source Filter */}
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Source
            </Typography>
            <ToggleButtonGroup
              value={localSource}
              exclusive
              onChange={(_, newSource) => {
                if (newSource !== null) setLocalSource(newSource);
              }}
              size="small"
            >
              <ToggleButton value="all">All</ToggleButton>
              <ToggleButton value="PRODUCT_HUNT">Product Hunt</ToggleButton>
              <ToggleButton value="HN">Hacker News</ToggleButton>
              <ToggleButton value="DEVTO">Dev.to</ToggleButton>
              <ToggleButton value="GITHUB_TRENDING">GitHub</ToggleButton>
              <ToggleButton value="LOBSTERS">Lobsters</ToggleButton>
              <ToggleButton value="TAAFT">AI Tools</ToggleButton>
            </ToggleButtonGroup>
          </Box>

          {/* Date Navigation */}
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Date
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0, 
                border: '1px solid', 
                borderColor: 'divider', 
                borderRadius: 1, 
                overflow: 'hidden'
              }}>
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
                    minWidth: 160, 
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
                variant="outlined"
                onClick={() => setSelectedDate('2026-03-09')}
                disabled={selectedDate === '2026-03-09'}
                sx={{ minWidth: 'auto', px: 1.5 }}
              >
                Latest
              </Button>
            </Box>
          </Box>

          {/* Sort By */}
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Sort By
            </Typography>
            <ToggleButtonGroup
              value={sortBy}
              exclusive
              onChange={(_, newSort) => {
                if (newSort) setSortBy(newSort);
              }}
              size="small"
            >
              <ToggleButton value="engagement">
                <TrendingUpIcon sx={{ fontSize: 16, mr: 0.5 }} />
                Trending
              </ToggleButton>
              <ToggleButton value="recency">
                <NewReleasesIcon sx={{ fontSize: 16, mr: 0.5 }} />
                Latest
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
        </Box>
      </Paper>

      {/* Error Alerts */}
      {productsError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load products: {(productsError as Error).message}
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
          
          {/* Info Card */}
          <Paper sx={{ p: 2, mt: 2 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              About
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Trendify tracks trending products and projects from across the web.
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Products are ranked by engagement score, which combines upvotes, comments, and social signals.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Tags shown are native to each platform (Product Hunt tags, GitHub topics, subreddit tags).
            </Typography>
          </Paper>
        </Box>

        {/* Products List - Right Side */}
        <Box>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Box>
                <Typography variant="h5" gutterBottom fontWeight="bold">
                  Products from {format(parseISO(selectedDate), 'MMMM d, yyyy')}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {sortBy === 'engagement' 
                    ? 'Sorted by engagement score (upvotes + comments)'
                    : 'Sorted by publish date (newest first)'}
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                {sortedProducts.length} {sortedProducts.length === 1 ? 'product' : 'products'}
              </Typography>
            </Box>
            <ProductList products={sortedProducts} isLoading={productsLoading} />
          </Paper>
        </Box>
      </Box>
    </Box>
  );
};

export default ProductDashboard;