"""
FastAPI streaming relay: receives SSE from OpenAI and relays to the browser.

Demonstrates StreamingResponse with async generator, client disconnect
detection, and proper SSE headers (Cache-Control, X-Accel-Buffering).

Requires: OPENAI_API_KEY environment variable.
Run with: uvicorn fastapi_relay:app --reload

From: https://dadops.dev/blog/streaming-llm-responses/
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import json

app = FastAPI()
client = AsyncOpenAI()


async def relay_stream(prompt: str, request: Request):
    """Relay OpenAI stream to browser as SSE."""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    try:
        async for chunk in stream:
            # If the client disconnected, stop burning tokens
            if await request.is_disconnected():
                await stream.close()
                return
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield f"data: {json.dumps({'text': delta})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception:
        await stream.close()
        raise


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    return StreamingResponse(
        relay_stream(body["prompt"], request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
