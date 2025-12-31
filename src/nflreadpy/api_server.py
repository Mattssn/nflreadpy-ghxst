"""Simple HTTP API for nflreadpy loaders.

This module exposes a lightweight HTTP server that can be run directly or via
Docker to serve nflreadpy loader outputs as JSON. It avoids external web
framework dependencies to keep the deployment footprint small while still
providing structured responses suitable for tools like n8n.
"""

from __future__ import annotations

import inspect
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import polars as pl
from pydantic import BaseModel, Field, ValidationError

from nflreadpy.datasets import (
    player_name_mapping,
    team_abbr_mapping,
    team_abbr_mapping_norelocate,
)
from nflreadpy.load_injuries import load_injuries
from nflreadpy.load_pbp import load_pbp
from nflreadpy.load_players import load_players
from nflreadpy.load_schedules import load_schedules
from nflreadpy.load_teams import load_teams

LoaderFunction = Callable[..., Any]

SUPPORTED_LOADERS: dict[str, LoaderFunction] = {
    "load_schedules": load_schedules,
    "load_pbp": load_pbp,
    "load_injuries": load_injuries,
    "load_players": load_players,
    "load_teams": load_teams,
    "team_abbr_mapping": team_abbr_mapping,
    "team_abbr_mapping_norelocate": team_abbr_mapping_norelocate,
    "player_name_mapping": player_name_mapping,
}


class LoadRequest(BaseModel):
    """Request model for loader invocations."""

    loader: str = Field(description="Name of the loader to run.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the loader function.",
    )
    limit: int | None = Field(
        default=None,
        description="Optional number of rows to include in the response.",
    )


class LoaderNotFoundError(LookupError):
    """Raised when the requested loader is not registered."""


class LoaderParameterError(ValueError):
    """Raised when parameters passed to a loader are invalid."""


def describe_loader(name: str, func: LoaderFunction) -> dict[str, Any]:
    signature = inspect.signature(func)
    parameters = []
    for parameter in signature.parameters.values():
        param_info: dict[str, Any] = {"name": parameter.name}
        if parameter.annotation is not inspect._empty:
            param_info["type"] = str(parameter.annotation)
        if parameter.default is not inspect._empty:
            param_info["default"] = parameter.default
        parameters.append(param_info)

    return {
        "name": name,
        "doc": (inspect.getdoc(func) or "").splitlines()[0],
        "parameters": parameters,
    }


def get_registered_loaders() -> list[dict[str, Any]]:
    return [describe_loader(name, func) for name, func in SUPPORTED_LOADERS.items()]


def _coerce_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if limit < 0:
        raise LoaderParameterError("limit must be non-negative")
    return limit


def _normalize_result(result: Any, limit: int | None) -> tuple[list[Any], int]:
    if isinstance(result, pl.LazyFrame):
        result = result.collect()

    if isinstance(result, pl.DataFrame):
        if limit is not None:
            result = result.head(limit)
        data = result.to_dicts()
        return data, len(data)

    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
        data_list = list(result)
        if limit is not None:
            data_list = data_list[:limit]
        return data_list, len(data_list)

    return [result], 1


def execute_loader(request_model: LoadRequest) -> tuple[list[Any], int]:
    loader = SUPPORTED_LOADERS.get(request_model.loader)
    if loader is None:
        raise LoaderNotFoundError(f"Unsupported loader '{request_model.loader}'")

    limit = _coerce_limit(request_model.limit)

    try:
        result = loader(**request_model.params)
    except TypeError as exc:
        msg = "Invalid parameters for loader: " + str(exc)
        raise LoaderParameterError(msg) from exc

    return _normalize_result(result, limit)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pl.Date, pl.Datetime)):
        return str(value)
    return str(value)


def _load_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    if not raw_body:
        return {}
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise LoaderParameterError(f"Invalid JSON body: {exc}") from exc


def _send_json(
    handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: Any
) -> None:
    body = json.dumps(payload, default=_json_default).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class LoaderRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/health":
            _send_json(self, HTTPStatus.OK, {"status": "ok"})
            return
        if parsed_path.path == "/loaders":
            _send_json(self, HTTPStatus.OK, {"loaders": get_registered_loaders()})
            return

        _send_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/load":
            _send_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            body = _load_body(self)
            request_model = LoadRequest.model_validate(body)
            data, row_count = execute_loader(request_model)
        except LoaderNotFoundError as exc:
            _send_json(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        except LoaderParameterError as exc:
            _send_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except ValidationError as exc:
            _send_json(self, HTTPStatus.UNPROCESSABLE_ENTITY, exc.errors())
            return
        except Exception as exc:  # pragma: no cover - defensive guard
            _send_json(
                self,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Loader execution failed", "detail": str(exc)},
            )
            return

        response_payload = {
            "loader": request_model.loader,
            "row_count": row_count,
            "data": data,
        }
        _send_json(self, HTTPStatus.OK, response_payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if os.getenv("NFLREADPY_QUIET", "false").lower() == "true":
            return
        super().log_message(format, *args)


def run_server(host: str | None = None, port: int | None = None) -> None:
    resolved_host = host or os.getenv("NFLREADPY_HOST", "0.0.0.0")
    resolved_port = port or int(os.getenv("NFLREADPY_PORT", "8000"))

    server = ThreadingHTTPServer((resolved_host, resolved_port), LoaderRequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
