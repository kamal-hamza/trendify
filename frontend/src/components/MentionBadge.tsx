import { Typography, ButtonBase, Tooltip, keyframes, Box } from '@mui/material';
import { MessageCircle, Sparkles } from 'lucide-react';

interface MentionBadgeProps {
    count: number;
    isHot?: boolean;
    label?: string;
}

const pulse = keyframes`
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.1); opacity: 0.8; }
  100% { transform: scale(1); opacity: 1; }
`;

const MentionBadge = ({ count, isHot = false, label }: MentionBadgeProps) => {
    const Icon = isHot ? Sparkles : MessageCircle;

    return (
        <Tooltip title={label || `${count} mentions across sources`}>
            <ButtonBase
                sx={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 0.75,
                    px: 1,
                    py: 0.5,
                    borderRadius: 1.5,
                    transition: 'all 0.2s ease-in-out',
                    color: isHot ? '#F59E0B' : 'text.secondary',
                    bgcolor: isHot ? 'rgba(245, 158, 11, 0.1)' : 'transparent',
                    '&:hover': {
                        bgcolor: isHot ? 'rgba(245, 158, 11, 0.15)' : 'rgba(0,0,0,0.04)',
                        color: isHot ? '#D97706' : 'text.primary',
                    },
                }}
            >
                <Box
                    component="span"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        animation: isHot ? `${pulse} 2s infinite ease-in-out` : 'none',
                    }}
                >
                    <Icon size={16} strokeWidth={isHot ? 2.5 : 2} />
                </Box>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {count}
                </Typography>
            </ButtonBase>
        </Tooltip>
    );
};

export default MentionBadge;
