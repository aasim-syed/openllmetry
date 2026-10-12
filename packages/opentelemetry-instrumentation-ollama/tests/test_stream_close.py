"""Closing a streamed response early must still end the span and release the connection."""

import json

import httpx
import ollama
import pytest

CHUNKS = [
    {"model": "llama3", "response": "Hello", "done": False},
    {"model": "llama3", "response": " world", "done": False},
    {"model": "llama3", "response": "", "done": True},
]


class _SyncBody(httpx.SyncByteStream):
    def __init__(self):
        self.closed = False

    def __iter__(self):
        for chunk in CHUNKS:
            yield (json.dumps(chunk) + "\n").encode()

    def close(self):
        self.closed = True


class _AsyncBody(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        for chunk in CHUNKS:
            yield (json.dumps(chunk) + "\n").encode()

    async def aclose(self):
        self.closed = True


def _sync_client():
    body = _SyncBody()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    return ollama.Client(transport=transport), body


def _async_client():
    body = _AsyncBody()
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=body))
    return ollama.AsyncClient(transport=transport), body


def test_stream_close_before_read_ends_span(instrument_legacy, span_exporter):
    client, _ = _sync_client()

    stream = client.generate(model="llama3", prompt="hi", stream=True)
    stream.close()

    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["ollama.completion"]


def test_stream_close_mid_read_ends_span_once(instrument_legacy, span_exporter):
    client, body = _sync_client()

    stream = client.generate(model="llama3", prompt="hi", stream=True)
    next(stream)
    stream.close()

    assert len(span_exporter.get_finished_spans()) == 1
    assert body.closed


def test_stream_full_read_still_works(instrument_legacy, span_exporter):
    client, _ = _sync_client()

    text = "".join(chunk["response"] for chunk in client.generate(
        model="llama3", prompt="hi", stream=True
    ))

    assert text == "Hello world"
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_stream_aclose_before_read_ends_span(
    instrument_legacy, span_exporter
):
    client, _ = _async_client()

    stream = await client.generate(model="llama3", prompt="hi", stream=True)
    await stream.aclose()

    spans = span_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["ollama.completion"]


@pytest.mark.asyncio
async def test_async_stream_aclose_mid_read_closes_response(
    instrument_legacy, span_exporter
):
    client, body = _async_client()

    stream = await client.generate(model="llama3", prompt="hi", stream=True)
    await stream.__anext__()
    await stream.aclose()

    assert len(span_exporter.get_finished_spans()) == 1
    assert body.closed


@pytest.mark.asyncio
async def test_async_stream_full_read_still_works(instrument_legacy, span_exporter):
    client, _ = _async_client()

    stream = await client.generate(model="llama3", prompt="hi", stream=True)
    text = "".join([chunk["response"] async for chunk in stream])

    assert text == "Hello world"
    assert len(span_exporter.get_finished_spans()) == 1
