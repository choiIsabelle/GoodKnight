from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import chess
import os
import concurrent.futures
import traceback

from src.utils import chess_manager
from src import main

app = FastAPI()

# Configure CORS using an environment variable so deployments can set the
# allowed origins without editing code. Set ALLOWED_ORIGINS to a
# comma-separated list of origins (example: "https://site.pages.dev,https://app.example.com").
allowed = os.getenv("ALLOWED_ORIGINS")
if allowed:
    allow_list = [o.strip() for o in allowed.split(",") if o.strip()]
else:
    # Default to the common Cloudflare Pages pattern if not provided.
    allow_list = ["https://goodknight.pages.dev"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def root():
    return JSONResponse(content={"running": True})


@app.get("/health")
async def health():
    # Quick health check that also reports whether the model object was loaded
    model_loaded = hasattr(main, "model") and main.model is not None
    return JSONResponse(content={"running": True, "model_loaded": model_loaded})


@app.post("/move")
async def get_move(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    if ("pgn" not in data or "timeleft" not in data):
        return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

    pgn = data["pgn"]
    timeleft = data["timeleft"]  # in milliseconds

    chess_manager.set_context(pgn, timeleft)
    print("pgn", pgn, flush=True)

    # Run model inference in a worker thread with a timeout so a slow/busy
    # model doesn't hang the request forever. This also gives clear logs
    # when inference is slow.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    start_time = time.perf_counter()
    try:
        future = executor.submit(chess_manager.get_model_move)
        # Timeout after 12 seconds — tune as needed for your model/instance
        move, move_probs, logs = future.result(timeout=12)
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
    except concurrent.futures.TimeoutError:
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
        print("get_move timed out", flush=True)
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Inference timeout", "time_taken": time_taken}, status_code=504)
    except Exception as e:
        end_time = time.perf_counter()
        time_taken = (end_time - start_time) * 1000
        tb = traceback.format_exc()
        print("get_move exception:\n", tb, flush=True)
        return JSONResponse(
            content={
                "move": None,
                "move_probs": None,
                "time_taken": time_taken,
                "error": "Bot raised an exception",
                "logs": None,
                "exception": str(e),
                "traceback": tb,
            },
            status_code=500,
        )

    # Confirm type of move_probs
    if not isinstance(move_probs, dict):
        return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    for m, prob in move_probs.items():
        if not isinstance(m, chess.Move) or not isinstance(prob, float):
            return JSONResponse(content={m: None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

    # Translate move_probs to Dict[str, float]
    move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

    return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})


@app.get("/move")
async def move_get_info():
    # Helpful human-readable response for GET requests to /move so that
    # someone browsing the URL or a misconfigured client doesn't just see
    # FastAPI's generic Method Not Allowed. The actual API expects a POST
    # with JSON: { "pgn": "<pgn>", "timeleft": 60000 }
    return JSONResponse(
        content={
            "detail": "This endpoint expects a POST with JSON body { 'pgn': string, 'timeleft': int (ms) }. Use POST /move to request a bot move.",
            "example_curl": "curl -X POST https://<your-host>/move -H 'Content-Type: application/json' -d '{\"pgn\":\"1. e4 e5 2. Nf3 Nc6\",\"timeleft\":60000}'",
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT") or os.getenv("SERVE_PORT", "5058"))
    uvicorn.run(app, host="0.0.0.0", port=port)
