import json
import time
import chess
import io
from src.eval import find_best_move

# Model is loaded at module level (cold start only)
# This ensures the model stays in memory between warm invocations
print("Lambda handler initialized - model loaded")


def handler(event, context):
    """
    AWS Lambda handler for chess move API.

    Expected event format:
    {
        "body": "{\"fen\": \"...\"}"
    }
    or for direct invocation:
    {
        "fen": "...",
    }
    """

    # Handle API Gateway format (has 'body' key) or direct invocation
    if 'body' in event:
        try:
            body = json.loads(event['body'])
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Invalid JSON in request body'})
            }
    else:
        body = event

    # Validate required fields
    if 'fen' not in body:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': 'Missing fen in request body'})
        }

    fen = body['fen']
    print(f"Processing move for fen:\n{fen}")
    try:
        start_time = time.perf_counter()
        move = find_best_move(chess.Board(fen))
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except Exception as e:
        time_taken = (time.perf_counter() - start_time) * 1000
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'move': None,
                'exception': str(e),
            })
        }

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            'move': move.uci(),
        })
    }


def health_check_handler(event, context):
    """
    Simple health check endpoint.
    """
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'running': True})
    }
