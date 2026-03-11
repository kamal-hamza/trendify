import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline, responsiveFontSizes } from '@mui/material';
import TrendDashboard from './components/TrendDashboard';
import ProductDashboard from './components/ProductDashboard';
import AppShell from './components/AppShell';

// Create React Query client with default options
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

// Create Material-UI theme
let theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#4f378b',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#00639b',
    },
    success: {
      main: '#146c2e',
    },
    warning: {
      main: '#b26a00',
    },
    error: {
      main: '#b3261e',
    },
    text: {
      primary: '#1c1b1f',
      secondary: '#49454f',
    },
    background: {
      default: '#f3edf7',
      paper: '#fffbff',
    },
    divider: 'rgba(28, 27, 31, 0.14)',
  },
  shape: {
    borderRadius: 16,
  },
  typography: {
    fontFamily: '"Roboto Flex", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '3rem',
      fontWeight: 700,
      letterSpacing: '-0.04em',
    },
    h2: {
      fontSize: '2.4rem',
      fontWeight: 700,
      letterSpacing: '-0.03em',
    },
    h3: {
      fontSize: '1.9rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 700,
    },
    h5: {
      fontSize: '1.2rem',
      fontWeight: 700,
    },
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
    },
    button: {
      fontWeight: 600,
      textTransform: 'none',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: 'linear-gradient(180deg, #f8f2ff 0%, #f3edf7 100%)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 18,
          boxShadow: 'none',
          border: '1px solid rgba(28, 27, 31, 0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          borderRadius: 18,
          border: '1px solid rgba(28, 27, 31, 0.1)',
          boxShadow: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 999,
          paddingInline: 18,
          minHeight: 42,
          boxShadow: 'none',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          fontWeight: 500,
        },
      },
    },
    MuiToggleButtonGroup: {
      styleOverrides: {
        grouped: {
          borderRadius: '10px !important',
          border: '1px solid rgba(28, 27, 31, 0.12) !important',
          margin: 0,
        },
      },
    },
    MuiToggleButton: {
      styleOverrides: {
        root: {
          borderRadius: '10px !important',
          paddingInline: 14,
          textTransform: 'none',
          backgroundColor: '#fffbff',
          '&.Mui-selected': {
            backgroundColor: '#e8def8',
            color: '#21005d',
          },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: '#ffffff',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 18,
        },
      },
    },
  },
});

theme = responsiveFontSizes(theme);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <AppShell>
              <Routes>
                <Route path="/" element={<ProductDashboard />} />
                <Route path="/topics" element={<TrendDashboard />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
          </AppShell>
        </Router>
      </ThemeProvider>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;