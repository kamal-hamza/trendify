import { Box, Typography } from '@mui/material';

interface SourceIndicatorProps {
    source: string;
    sourceDisplay: string;
    config?: { icon: any; color: string; label: string };
}

const SourceIndicator = ({ sourceDisplay, config }: SourceIndicatorProps) => {
    if (!config) {
        return (
            <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                {sourceDisplay}
            </Typography>
        );
    }

    const SourceIcon = config.icon;

    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <Box
                sx={{
                    display: 'grid',
                    placeItems: 'center',
                    width: 24,
                    height: 24,
                    borderRadius: 1,
                    bgcolor: 'rgba(0,0,0,0.04)',
                    color: config.color,
                }}
            >
                <SourceIcon sx={{ fontSize: 14 }} />
            </Box>
            <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                {config.label || sourceDisplay}
            </Typography>
        </Box>
    );
};

export default SourceIndicator;
