from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


@pytest.fixture
def server_policy(monkeypatch):
    wrapper_module = SimpleNamespace(PolicyServerWrapper=object)
    websocket_module = SimpleNamespace(WebsocketPolicyServer=object)
    monkeypatch.setitem(
        sys.modules,
        "deployment.model_server.policy_wrapper",
        wrapper_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "deployment.model_server.tools.websocket_policy_server",
        websocket_module,
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "deployment/model_server/server_policy.py"
    )
    spec = importlib.util.spec_from_file_location("server_policy_cli_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_cli_defaults_to_all_interfaces_and_allows_localhost_override(
    server_policy,
) -> None:
    parser = server_policy.build_argparser()

    assert parser.parse_args([]).host == "0.0.0.0"
    assert parser.parse_args(["--host", "127.0.0.1"]).host == "127.0.0.1"


def test_model_asset_paths_are_optional_cli_overrides(server_policy) -> None:
    args = server_policy.build_argparser().parse_args(
        ["--base-vlm-path", "/models/qwen", "--dino-model-path", "/models/dino"]
    )

    assert args.base_vlm_path == "/models/qwen"
    assert args.dino_model_path == "/models/dino"


def test_main_passes_host_to_websocket_server_without_loading_model(
    server_policy,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeWrapper:
        metadata = {"contract": "test"}

        def __init__(self, **kwargs) -> None:
            calls["wrapper"] = kwargs

    class FakeServer:
        def __init__(self, **kwargs) -> None:
            calls["server"] = kwargs

        def serve_forever(self) -> None:
            calls["served"] = True

    monkeypatch.setattr(server_policy, "PolicyServerWrapper", FakeWrapper)
    monkeypatch.setattr(server_policy, "WebsocketPolicyServer", FakeServer)
    args = SimpleNamespace(
        ckpt_path="/tmp/checkpoint.pt",
        base_vlm_path="/models/qwen",
        dino_model_path="/models/dino",
        host="127.0.0.1",
        port=12345,
        seed=None,
        use_bf16=True,
        idle_timeout=60,
    )

    server_policy.main(args)

    assert calls["wrapper"] == {
        "ckpt_path": "/tmp/checkpoint.pt",
        "device": "cuda",
        "use_bf16": True,
        "base_vlm_path": "/models/qwen",
        "dino_model_path": "/models/dino",
    }

    assert calls["server"] == {
        "policy": calls["server"]["policy"],
        "host": "127.0.0.1",
        "port": 12345,
        "idle_timeout": 60,
        "metadata": {"contract": "test"},
    }
    assert isinstance(calls["server"]["policy"], FakeWrapper)
    assert calls["served"] is True
