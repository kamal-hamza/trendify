import { Box, FormControl, InputLabel, Select, MenuItem, Paper, ToggleButton, ToggleButtonGroup, Chip, Typography } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import { useFilterStore } from '../store/filterStore';

const FilterBar = () => {
  const { source, category, days, mode, setSource, setCategory, setDays, setMode } = useFilterStore();

  const sources = [
    { value: 'all', label: 'All Sources' },
    { value: 'HN', label: 'Hacker News' },
    { value: 'REDDIT_LOCALLLAMA', label: 'r/LocalLLaMA' },
    { value: 'REDDIT_MACHINELEARNING', label: 'r/MachineLearning' },
    { value: 'REDDIT_PROGRAMMING', label: 'r/programming' },
    { value: 'GITHUB', label: 'GitHub' },
  ];

  const categories = [
    { value: 'all', label: 'All Categories' },
    { value: 'LLM', label: 'LLM' },
    { value: 'FRAMEWORK', label: 'Framework' },
    { value: 'LIBRARY', label: 'Library' },
    { value: 'TOOL', label: 'Tool' },
    { value: 'PLATFORM', label: 'Platform' },
    { value: 'LANGUAGE', label: 'Programming Language' },
    { value: 'PHILOSOPHY', label: 'Philosophy' },
    { value: 'OTHER', label: 'Other' },
  ];

  const timeRanges = [
    { value: 1, label: 'Last 24 hours' },
    { value: 3, label: 'Last 3 days' },
    { value: 7, label: 'Last 7 days' },
    { value: 14, label: 'Last 14 days' },
    { value: 30, label: 'Last 30 days' },
  ];

  const handleSourceChange = (event: SelectChangeEvent) => {
    setSource(event.target.value);
  };

  const handleCategoryChange = (event: SelectChangeEvent) => {
    setCategory(event.target.value);
  };

  const handleDaysChange = (event: SelectChangeEvent) => {
    setDays(Number(event.target.value));
  };

  const handleModeChange = (_event: React.MouseEvent<HTMLElement>, newMode: 'trending' | 'emerging' | null) => {
    if (newMode !== null) {
      setMode(newMode);
    }
  };

  return (
    <Paper
      sx={{
        p: { xs: 2, md: 2.5 },
        mb: 3,
      }}
    >
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1.1fr) repeat(3, minmax(0, 1fr))' },
          gap: 2,
          alignItems: 'end',
        }}
      >
        <Box
          sx={{
            minWidth: 0,
          }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
            Mode
          </Typography>
          <ToggleButtonGroup
            value={mode}
            exclusive
            onChange={handleModeChange}
            size="small"
            fullWidth
            sx={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: 1,
              '& .MuiToggleButtonGroup-grouped': { mr: '0 !important', borderLeft: '1px solid rgba(28, 27, 31, 0.12) !important' },
            }}
          >
            <ToggleButton value="trending" aria-label="trending">
              <TrendingUpIcon sx={{ mr: 0.75, fontSize: 18 }} />
              Trending
            </ToggleButton>
            <ToggleButton value="emerging" aria-label="emerging">
              <RocketLaunchIcon sx={{ mr: 0.75, fontSize: 18 }} />
              Emerging
              <Chip
                label="New"
                size="small"
                sx={{ ml: 0.75, height: 20, bgcolor: '#d6f8d3', color: '#146c2e' }}
              />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Box sx={{ minWidth: 0 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
            Source
          </Typography>
          <FormControl size="small" fullWidth>
            <InputLabel id="source-filter-label">Source</InputLabel>
            <Select
              labelId="source-filter-label"
              id="source-filter"
              value={source}
              label="Source"
              onChange={handleSourceChange}
            >
              {sources.map((s) => (
                <MenuItem key={s.value} value={s.value}>
                  {s.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        <Box sx={{ minWidth: 0 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
            Category
          </Typography>
          <FormControl size="small" fullWidth>
            <InputLabel id="category-filter-label">Category</InputLabel>
            <Select
              labelId="category-filter-label"
              id="category-filter"
              value={category}
              label="Category"
              onChange={handleCategoryChange}
            >
              {categories.map((c) => (
                <MenuItem key={c.value} value={c.value}>
                  {c.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        <Box sx={{ minWidth: 0 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
            Time range
          </Typography>
          <FormControl size="small" fullWidth>
            <InputLabel id="days-filter-label">Time Range</InputLabel>
            <Select
              labelId="days-filter-label"
              id="days-filter"
              value={days.toString()}
              label="Time Range"
              onChange={handleDaysChange}
            >
              {timeRanges.map((t) => (
                <MenuItem key={t.value} value={t.value}>
                  {t.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Box>
    </Paper>
  );
};

export default FilterBar;