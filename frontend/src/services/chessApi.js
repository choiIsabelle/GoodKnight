/**
 * Chess API service for communicating with the AWS Lambda chess engine
 */

// TODO: Replace this with your actual Lambda API endpoint once deployed
const API_ENDPOINT = import.meta.env.VITE_CHESS_API_ENDPOINT || 'YOUR_LAMBDA_ENDPOINT_HERE';

/**
 * Get the best move from the chess engine for a given position
 * @param {string} fen - The FEN string representing the board position
 * @returns {Promise<{move: string, thinkingTime?: number}>}
 */
export async function getBestMove(fen) {
  const startTime = performance.now();

  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ fen }),
    });

    if (!response.ok) {
      throw new Error(`API returned ${response.status}: ${response.statusText}`);
    }

    let data = await response.json();

    // Handle Lambda response format (has statusCode and body)
    if (data.statusCode) {
      // Parse the body string into JSON
      data = JSON.parse(data.body);
    }

    const endTime = performance.now();
    const thinkingTime = endTime - startTime;

    if (!data.move) {
      throw new Error(data.exception || data.error || 'No move returned from engine');
    }

    return {
      move: data.move,
      thinkingTime,
    };
  } catch (error) {
    console.error('Error calling chess API:', error);
    throw error;
  }
}

/**
 * Check if the API endpoint is configured
 * @returns {boolean}
 */
export function isApiConfigured() {
  return API_ENDPOINT && API_ENDPOINT !== 'YOUR_LAMBDA_ENDPOINT_HERE';
}
