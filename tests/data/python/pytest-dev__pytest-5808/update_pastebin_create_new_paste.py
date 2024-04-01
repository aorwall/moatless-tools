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