import { Box } from '@mui/material';

interface TopicTagsProps {
    topics: string[];
    maxDisplay?: number;
}

const TopicTags = ({ topics, maxDisplay = 8 }: TopicTagsProps) => {
    if (!topics || topics.length === 0) return null;

    return (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
            {topics.slice(0, maxDisplay).map((tag) => (
                <Box
                    key={tag}
                    sx={{
                        px: 1,
                        py: 0.25,
                        fontSize: '0.75rem',
                        fontWeight: 500,
                        color: 'secondary.main',
                        bgcolor: 'rgba(99, 102, 241, 0.08)',
                        borderRadius: 1,
                    }}
                >
                    {tag}
                </Box>
            ))}
            {topics.length > maxDisplay && (
                <Box
                    sx={{
                        px: 1,
                        py: 0.25,
                        fontSize: '0.75rem',
                        fontWeight: 500,
                        color: 'text.secondary',
                        bgcolor: 'action.hover',
                        borderRadius: 1,
                    }}
                >
                    +{topics.length - maxDisplay} more
                </Box>
            )}
        </Box>
    );
};

export default TopicTags;
