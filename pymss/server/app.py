import asyncio
import base64
import binascii
import json

from .audio import (
    decode_pcm,
    json_response,
    normalize_stems,
    parse_int,
    validate_common_options,
    zip_response,
)
from .config import ServerConfig
from .errors import APIError
from .state import load_state, model_card


try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
    raise RuntimeError("Install server dependencies with `pip install pymss[server]` or `uv sync --extra server`.") from exc


def _error_response(exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


def _check_auth(request, state):
    if not state.config.api_key:
        return
    expected = f"Bearer {state.config.api_key}"
    if request.headers.get("authorization") != expected:
        raise APIError(401, "invalid_api_key", "Invalid or missing API key.")


def _content_type(request):
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


async def _read_body(request, state):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > state.config.max_request_bytes:
                raise APIError(413, "request_too_large", "Request body is too large.")
        except ValueError:
            raise APIError(400, "invalid_request", "Invalid Content-Length header.")

    body = await request.body()
    if len(body) > state.config.max_request_bytes:
        raise APIError(413, "request_too_large", "Request body is too large.")
    return body


def _require_served_model(state, model):
    if not model:
        raise APIError(400, "invalid_model", "The 'model' field is required.", param="model")
    if not state.is_served_model(model):
        raise APIError(404, "model_not_found", f"Model {model!r} is not served by this process.", param="model")
    return model


async def _parse_json_request(request, state):
    body = await _read_body(request, state)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise APIError(400, "invalid_request", "Request body must be valid JSON.")
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_request", "JSON request body must be an object.")

    model = _require_served_model(state, payload.get("model"))
    input_data = payload.get("input")
    if not isinstance(input_data, dict):
        raise APIError(400, "invalid_request", "The 'input' object is required.", param="input")

    audio_format = str(input_data.get("format", "")).lower()
    sample_rate = parse_int(input_data.get("sample_rate"), "input.sample_rate")
    channels = parse_int(input_data.get("channels"), "input.channels")
    encoded = input_data.get("data")
    if not isinstance(encoded, str):
        raise APIError(400, "invalid_request", "input.data must be a base64 string.", param="input.data")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        raise APIError(400, "invalid_base64", "input.data must be valid base64.", param="input.data")

    stems = normalize_stems(payload.get("stems"), state.instruments)
    response_format = str(payload.get("response_format", "json")).lower()
    output_audio_format = str(payload.get("output_audio_format", "pcm_f32le")).lower()
    validate_common_options(response_format, output_audio_format)
    mix, seconds = decode_pcm(
        raw,
        audio_format,
        sample_rate,
        channels,
        state.sample_rate,
        state.config.max_audio_seconds,
    )
    return model, mix, stems, response_format, output_audio_format, seconds


async def _parse_binary_request(request, state):
    params = request.query_params
    model = _require_served_model(state, params.get("model"))
    audio_format = str(params.get("format", "")).lower()
    if not audio_format or params.get("sample_rate") is None or params.get("channels") is None:
        raise APIError(400, "missing_audio_metadata", "format, sample_rate, and channels are required.")
    sample_rate = parse_int(params.get("sample_rate"), "sample_rate", code="invalid_query_parameter")
    channels = parse_int(params.get("channels"), "channels", code="invalid_query_parameter")
    stems = normalize_stems(params.get("stems"), state.instruments)
    response_format = str(params.get("response_format", "json")).lower()
    output_audio_format = str(params.get("output_audio_format", "pcm_f32le")).lower()
    validate_common_options(response_format, output_audio_format)
    raw = await _read_body(request, state)
    mix, seconds = decode_pcm(
        raw,
        audio_format,
        sample_rate,
        channels,
        state.sample_rate,
        state.config.max_audio_seconds,
    )
    return model, mix, stems, response_format, output_audio_format, seconds


async def _parse_request(request, state):
    content_type = _content_type(request)
    if content_type == "application/json":
        return await _parse_json_request(request, state)
    if content_type == "application/octet-stream":
        return await _parse_binary_request(request, state)
    raise APIError(415, "unsupported_content_type", "Content-Type must be application/json or application/octet-stream.")


def _run_separation_sync(state, mix, stems):
    if state.separator.model_type == "vr":
        results = state.separator.separate(mix, pbar=False)
    else:
        results = state.separator.separate(mix, pbar=False, stems=stems)
    return results


async def _run_separation(state, mix, stems):
    acquired = await state.limiter.acquire()
    if not acquired:
        raise APIError(429, "server_overloaded", "Inference queue is full.")
    try:
        async with state.inference_lock:
            task = asyncio.to_thread(_run_separation_sync, state, mix, stems)
            if state.config.request_timeout_seconds:
                try:
                    return await asyncio.wait_for(task, timeout=state.config.request_timeout_seconds)
                except asyncio.TimeoutError:
                    raise APIError(504, "separation_timeout", "Separation request timed out.")
            return await task
    finally:
        await state.limiter.release()


def create_app(config):
    state = load_state(config)
    app = FastAPI(title="pymss server", version="1")
    app.state.pymss_state = state

    @app.exception_handler(APIError)
    async def handle_api_error(_request, exc):
        return _error_response(exc)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": True,
            "model": state.served_model_names[0],
            "device": state.device,
        }

    @app.get("/v1/models")
    async def list_served_models(request: Request):
        _check_auth(request, state)
        return {
            "object": "list",
            "data": [model_card(state, model_id) for model_id in state.served_model_names],
        }

    @app.get("/v1/models/{model}")
    async def get_served_model(model: str, request: Request):
        _check_auth(request, state)
        _require_served_model(state, model)
        return model_card(state, model)

    @app.post("/v1/audio/separations")
    async def separate_audio(request: Request):
        _check_auth(request, state)
        model, mix, stems, response_format, output_audio_format, input_seconds = await _parse_request(request, state)
        try:
            results = await _run_separation(state, mix, stems)
        except APIError:
            raise
        except Exception as exc:
            state.logger.exception("Separation failed")
            raise APIError(500, "separation_failed", str(exc), error_type="server_error")

        try:
            if response_format == "json":
                return json_response(state, model, results, stems, input_seconds)

            content = zip_response(state, model, results, stems, input_seconds, output_audio_format)
            return Response(content=content, media_type="application/zip")
        except APIError:
            raise
        except Exception as exc:
            state.logger.exception("Encoding separation response failed")
            raise APIError(500, "separation_failed", str(exc), error_type="server_error")

    return app


def run_server(config):
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps.
        raise RuntimeError("Install server dependencies with `pip install pymss[server]` or `uv sync --extra server`.") from exc

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
