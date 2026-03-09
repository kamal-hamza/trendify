/**
 * Utility functions for formatting data
 */

/**
 * Format large numbers with commas
 */
export const formatNumber = (num: number): string => {
  return num.toLocaleString();
};

/**
 * Format large numbers with K, M, B suffixes
 */
export const formatCompactNumber = (num: number): string => {
  if (num >= 1000000000) {
    return (num / 1000000000).toFixed(1) + 'B';
  }
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toString();
};

/**
 * Get color based on momentum score
 */
export const getMomentumColor = (score: number): string => {
  if (score > 5) return '#f44336'; // High momentum - red
  if (score > 2) return '#ff9800'; // Medium momentum - orange
  if (score > 0) return '#4caf50'; // Positive momentum - green
  return '#9e9e9e'; // No momentum - grey
};

/**
 * Get color based on sentiment score
 */
export const getSentimentColor = (sentiment: number): string => {
  if (sentiment > 0.3) return '#4caf50'; // Positive
  if (sentiment < -0.3) return '#f44336'; // Negative
  return '#757575'; // Neutral
};

/**
 * Get sentiment label
 */
export const getSentimentLabel = (sentiment: number): string => {
  if (sentiment > 0.3) return 'Positive';
  if (sentiment < -0.3) return 'Negative';
  return 'Neutral';
};
