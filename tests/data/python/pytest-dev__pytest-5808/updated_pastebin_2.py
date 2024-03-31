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
    if config.option.pastebin == "all":
        tr = config.pluginmanager.getplugin("terminalreporter")
        if tr is not None:
            config._pastebinfile = tempfile.TemporaryFile("w+b")
            oldwrite = tr._tw.write

            def tee_write(s, **kwargs):
                oldwrite(s, **kwargs)
                if isinstance(s, str):
                    s = s.encode("utf-8")
                config._pastebinfile.write(s)

            tr._tw.write = tee_write


def pytest_unconfigure(config):
    if hasattr(config, "_pastebinfile"):
        config._pastebinfile.seek(0)
        sessionlog = config._pastebinfile.read()
        config._pastebinfile.close()
        del config._pastebinfile
        tr = config.pluginmanager.getplugin("terminalreporter")
        del tr._tw.__dict__["write"]
        tr.write_sep("=", "Sending information to Paste Service")
        pastebinurl = create_new_paste(sessionlog)
        tr.write_line("pastebin session-log: %s\n" % pastebinurl)


def create_new_paste(contents):
    import re
    from urllib.request import urlopen
    from urllib.parse import urlencode

    params = {"code": contents, "lexer": "text", "expiry": "1week"}
    url = "https://bpaste.net"
    try:
        response = (
            urlopen(url, data=urlencode(params).encode("ascii")).read().decode("utf-8")
        )
    except OSError as exc_info:
        return "bad response: %s" % exc_info
    m = re.search(r'href="/raw/(\w+)"', response)
    if m:
        return "{}/show/{}".format(url, m.group(1))
    else:
        return "bad response: invalid format ('" + response + "')"


def pytest_terminal_summary(terminalreporter):
    import _pytest.config

    if terminalreporter.config.option.pastebin != "failed":
        return
    tr = terminalreporter
    if "failed" in tr.stats:
        terminalreporter.write_sep("=", "Sending information to Paste Service")
        for rep in terminalreporter.stats.get("failed"):
            try:
                msg = rep.longrepr.reprtraceback.reprentries[-1].reprfileloc
            except AttributeError:
                msg = tr._getfailureheadline(rep)
            tw = _pytest.config.create_terminal_writer(
                terminalreporter.config, stringio=True
            )
            rep.toterminal(tw)
            s = tw.stringio.getvalue()
            assert len(s)
            pastebinurl = create_new_paste(s)
            tr.write_line("{} --> {}".format(msg, pastebinurl))
