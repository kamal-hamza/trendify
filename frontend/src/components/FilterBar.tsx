import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  InputAdornment,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Chip,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import { useFilterStore } from '../store/filterStore';

interface FilterBarProps {
  showSearch?: boolean;
}

const FilterBar = ({ showSearch = true }: FilterBarProps) => {
  const { source, category, days, search, mode, setSource, setCategory, setDays, setSearch, setMode } =
    useFilterStore();

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

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(event.target.value);
  };

  const handleModeChange = (_event: React.MouseEvent<HTMLElement>, newMode: 'trending' | 'emerging' | null) => {
    if (newMode !== null) {
      setMode(newMode);
    }
  };

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        mb: 3,
        backgroundColor: 'white',
        borderRadius: 2,
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: 2,
          alignItems: 'center',
        }}
      >
        {/* Mode Toggle */}
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={handleModeChange}
          size="small"
          sx={{ flexShrink: 0 }}
        >
          <ToggleButton value="trending" aria-label="trending">
            <TrendingUpIcon sx={{ mr: 0.5, fontSize: 18 }} />
            Trending
          </ToggleButton>
          <ToggleButton value="emerging" aria-label="emerging">
            <RocketLaunchIcon sx={{ mr: 0.5, fontSize: 18 }} />
            Emerging
            <Chip
              label="New"
              size="small"
              color="success"
              sx={{ ml: 0.5, height: 18, fontSize: '0.7rem' }}
            />
          </ToggleButton>
        </ToggleButtonGroup>

        {/* Search Field */}
        {showSearch && (
          <TextField
            size="small"
            placeholder="Search topics..."
            value={search}
            onChange={handleSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            sx={{ flexGrow: 1, minWidth: 200 }}
          />
        )}

        {/* Source Filter */}
        <FormControl size="small" sx={{ minWidth: 180 }}>
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

        {/* Category Filter */}
        <FormControl size="small" sx={{ minWidth: 180 }}>
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

        {/* Time Range Filter */}
        <FormControl size="small" sx={{ minWidth: 160 }}>
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
    </Paper>
  );
};

export default FilterBar;