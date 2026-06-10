from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse


WEBUI_STATIC_DIR = Path(__file__).with_name("webui_static")


def _missing_assets_response():
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "WebUI static assets are not built. Build the WebUI according to https://github.com/pymss-project/pymss/blob/main/README.md, then start the server again.",
                "type": "server_error",
                "param": None,
                "code": "webui_assets_missing",
            }
        },
    )


def _index_path(static_dir):
    return Path(static_dir) / "index.html"


def _asset_path(static_dir, asset_path):
    assets_root = (Path(static_dir) / "assets").resolve()
    candidate = (assets_root / asset_path).resolve()
    try:
        candidate.relative_to(assets_root)
    except ValueError:
        raise HTTPException(status_code=404)
    return candidate


def _index_response(static_dir):
    index = _index_path(static_dir)
    if not index.is_file():
        return _missing_assets_response()
    return FileResponse(index, media_type="text/html; charset=utf-8")


def register_webui_routes(app, static_dir=None):
    static_dir = Path(static_dir) if static_dir is not None else WEBUI_STATIC_DIR

    @app.get("/ui")
    async def redirect_webui_root():
        return RedirectResponse(url="/ui/", status_code=307)

    @app.get("/ui/")
    async def webui_index():
        return _index_response(static_dir)

    @app.get("/ui/assets/{asset_path:path}")
    async def webui_asset(asset_path: str):
        path = _asset_path(static_dir, asset_path)
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path)

    @app.get("/ui/{_spa_path:path}")
    async def webui_spa_fallback(_spa_path: str):
        return _index_response(static_dir)
