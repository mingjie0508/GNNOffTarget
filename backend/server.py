# backend/server.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import json
from ga import run_ga_stream

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/runs/stream")
async def runs_stream(request: Request):
    params = dict(
        request.query_params
    )  # {"lambda": "0.6", "generations":"50", ...}

    def event_gen():
        for frame in run_ga_stream(params):
            yield {"event": "frame", "data": json.dumps(frame)}

    return EventSourceResponse(event_gen())
