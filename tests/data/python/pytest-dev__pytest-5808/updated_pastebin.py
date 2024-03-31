""" submit failure or test session information to a pastebin service. """
import tempfile

import pytest

def pytest_addoption(parser):
    group = parser.getgroup("terminal reporting")
    group._addoption(
        "--pastebin",
        metavar="mode",
        action="store",
        dest="pastebin",
        default=None,
        choices=["failed", "all"],
        help="send failed|all info to bpaste.net pastebin service.",
    )

@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # ... existing code

def pytest_unconfigure(config):
    # ... existing code

def create_new_paste(contents):
    """
    Creates a new paste using bpaste.net service.

    :contents: paste contents as utf-8 encoded bytes
    :returns: url to the pasted contents or error message
    """
    import re
    from urllib.request import urlopen
    from urllib.parse import urlencode

    params = {"code": contents, "lexer": "text", "expiry": "1week"}
    url = "https://bpaste.net"
    try:
        response = (
            urlopen(url, data=urlencode(params).encode("ascii")).read().decode("utf-8")
        )
    except OSError as exc_info:  # urllib errors
        return "bad response: %s" % exc_info
    m = re.search(r'href="/raw/(\w+)"', response)
    if m:
        return "{}/show/{}".format(url, m.group(1))
    else:
        return "bad response: invalid format ('" + response + "')"

def pytest_terminal_summary(terminalreporter):
    # ... existing code
