from omicau.ui import server


def test_preferred_port_is_checked_by_binding_not_connecting(monkeypatch):
    class ProbeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, address):
            assert address == ("127.0.0.1", 8765)

        def connect_ex(self, address):
            raise AssertionError("preferred-port check must not make a network connection")

    monkeypatch.setattr(server.socket, "socket", lambda *args: ProbeSocket())

    assert server._free_port(8765) == 8765
