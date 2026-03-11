import { Box, ButtonBase, Stack, Typography } from '@mui/material';
import AutoGraphOutlinedIcon from '@mui/icons-material/AutoGraphOutlined';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import InsightsOutlinedIcon from '@mui/icons-material/InsightsOutlined';
import { Link, useLocation } from 'react-router-dom';
import type { ReactNode, ElementType } from 'react';

interface AppShellProps {
  children: ReactNode;
}

interface NavigationItem {
  label: string;
  path: string;
  description: string;
  icon: ElementType;
}

const navigationItems: NavigationItem[] = [
  {
    label: 'Products',
    path: '/',
    description: 'Daily launches and high-engagement posts',
    icon: DashboardOutlinedIcon,
  },
  {
    label: 'Topics',
    path: '/topics',
    description: 'Cross-source trend and emerging topic signals',
    icon: InsightsOutlinedIcon,
  },
];

const AppShell = ({ children }: AppShellProps) => {
  const location = useLocation();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        bgcolor: 'background.default',
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', md: '264px minmax(0, 1fr)' },
      }}
    >
      <Box
        component="aside"
        sx={{
          display: { xs: 'none', md: 'flex' },
          flexDirection: 'column',
          position: 'sticky',
          top: 0,
          height: '100vh',
          px: 2.5,
          py: 3,
          bgcolor: '#f8f2ff',
          borderRight: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ px: 1, mb: 4 }}>
          <Box
            sx={{
              width: 44,
              height: 44,
              borderRadius: 2,
              display: 'grid',
              placeItems: 'center',
              bgcolor: 'primary.main',
              color: 'primary.contrastText',
            }}
          >
            <AutoGraphOutlinedIcon />
          </Box>
          <Box>
            <Typography variant="subtitle1" fontWeight={700}>
              Trendify
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Market and topic signals
            </Typography>
          </Box>
        </Stack>

        <Typography variant="overline" color="text.secondary" sx={{ px: 1.5, mb: 1 }}>
          Navigation
        </Typography>
        <Stack spacing={0.75}>
          {navigationItems.map((item) => {
            const active = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <ButtonBase
                key={item.path}
                component={Link}
                to={item.path}
                sx={{
                  width: '100%',
                  justifyContent: 'flex-start',
                  textAlign: 'left',
                  borderRadius: 3,
                  px: 1.5,
                  py: 1.25,
                  bgcolor: active ? '#e8def8' : 'transparent',
                  color: active ? '#21005d' : 'text.primary',
                  border: '1px solid',
                  borderColor: active ? '#b69df8' : 'transparent',
                }}
              >
                <Stack direction="row" spacing={1.25} alignItems="flex-start">
                  <Box
                    sx={{
                      width: 36,
                      height: 36,
                      borderRadius: 2,
                      display: 'grid',
                      placeItems: 'center',
                      bgcolor: active ? 'primary.main' : '#ede7f6',
                      color: active ? 'primary.contrastText' : 'primary.main',
                      flexShrink: 0,
                    }}
                  >
                    <Icon fontSize="small" />
                  </Box>
                  <Box>
                    <Typography variant="subtitle2" fontWeight={700}>
                      {item.label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {item.description}
                    </Typography>
                  </Box>
                </Stack>
              </ButtonBase>
            );
          })}
        </Stack>

        <Box sx={{ mt: 'auto', px: 1.5, pt: 3 }}>
          <Typography variant="body2" color="text.secondary">
            Documentation-style navigation that stays visible while scrolling.
          </Typography>
        </Box>
      </Box>

      <Box sx={{ minWidth: 0 }}>
        <Box
          component="header"
          sx={{
            position: 'sticky',
            top: 0,
            zIndex: 1200,
            backdropFilter: 'blur(16px)',
            bgcolor: 'rgba(243, 237, 247, 0.88)',
            borderBottom: '1px solid',
            borderColor: 'divider',
            px: { xs: 2, md: 4 },
            py: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 2,
          }}
        >
          <Stack direction="row" spacing={1.5} alignItems="center">
            <Box
              sx={{
                width: 44,
                height: 44,
                borderRadius: 2,
                display: 'grid',
                placeItems: 'center',
                bgcolor: 'primary.main',
                color: 'primary.contrastText',
              }}
            >
              <AutoGraphOutlinedIcon />
            </Box>
            <Box>
              <Typography variant="subtitle1" fontWeight={700}>
                Trendify
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Market and topic signals
              </Typography>
            </Box>
          </Stack>

          <Typography variant="body2" color="text.secondary" sx={{ display: { xs: 'none', md: 'block' } }}>
            Clean dashboard views for products and topics
          </Typography>
        </Box>

        <Box
          component="main"
          sx={{
          px: { xs: 2, md: 4 },
          py: { xs: 2, md: 4 },
          maxWidth: 1320,
        }}
      >
          <Stack direction="row" spacing={1} sx={{ display: { xs: 'flex', md: 'none' }, mb: 2 }}>
            {navigationItems.map((item) => {
              const active = location.pathname === item.path;
              return (
                <ButtonBase
                  key={item.path}
                  component={Link}
                  to={item.path}
                  sx={{
                    flex: 1,
                    px: 2,
                    py: 1.25,
                    borderRadius: 2.5,
                    bgcolor: active ? '#e8def8' : 'background.paper',
                    color: active ? '#21005d' : 'text.primary',
                    border: '1px solid',
                    borderColor: active ? '#b69df8' : 'divider',
                    fontWeight: 600,
                  }}
                >
                  {item.label}
                </ButtonBase>
              );
            })}
          </Stack>
          {children}
        </Box>
      </Box>
    </Box>
  );
};

export default AppShell;
