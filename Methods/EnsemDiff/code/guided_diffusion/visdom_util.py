import os


class NullVisdom:
    def image(self, *args, **kwargs):
        return None

    def line(self, *args, **kwargs):
        return None


def make_visdom(port=8850):
    enabled = os.environ.get("ENSEMDIFF_USE_VISDOM", "").lower() in {"1", "true", "yes"}
    if not enabled:
        return NullVisdom()

    from visdom import Visdom

    return Visdom(port=port, use_incoming_socket=False)
