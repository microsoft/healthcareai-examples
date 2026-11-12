# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline guards scoped to this test directory; no SDK or image-module stubs."""

import socket
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def toolkit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Network, credential initialization, and AML lookup are forbidden."
        )

    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    import os

    for name in tuple(os.environ):
        if name.startswith(
            ("AZURE", "MI2_", "CXRREPORTGEN_", "MIP_", "GIGAPATH_", "GIGATIME_")
        ):
            monkeypatch.delenv(name)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, forbidden)

    import dotenv
    import requests

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)

    import azure.identity as identity
    from healthcareai_toolkit import clients, settings
    from healthcareai_toolkit.clients.medimage import medimagebase
    from healthcareai_toolkit.util import connection_managers

    monkeypatch.setattr(identity, "DefaultAzureCredential", forbidden)
    monkeypatch.setattr(connection_managers, "MLClient", forbidden)
    monkeypatch.setattr(settings, "_constants", dict.fromkeys(settings.keys()))
    return SimpleNamespace(
        clients=clients,
        settings=settings,
        base=medimagebase,
        managers=connection_managers,
        identity=identity,
        forbidden=forbidden,
    )
