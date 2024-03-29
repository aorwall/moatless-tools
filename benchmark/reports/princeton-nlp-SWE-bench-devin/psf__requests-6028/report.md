# psf__requests-6028

| **psf/requests** | `0192aac24123735b3eaf9b08df46429bb770c283` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 35738 |
| **Any found context length** | 35738 |
| **Avg pos** | 106.0 |
| **Min pos** | 106 |
| **Max pos** | 106 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/requests/utils.py b/requests/utils.py
--- a/requests/utils.py
+++ b/requests/utils.py
@@ -974,6 +974,10 @@ def prepend_scheme_if_needed(url, new_scheme):
     if not netloc:
         netloc, path = path, netloc
 
+    if auth:
+        # parse_url doesn't provide the netloc with auth
+        # so we'll add it ourselves.
+        netloc = '@'.join([auth, netloc])
     if scheme is None:
         scheme = new_scheme
     if path is None:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| requests/utils.py | 977 | 977 | 106 | 1 | 35738


## Problem Statement

```
Proxy authentication bug
<!-- Summary. -->

When using proxies in python 3.8.12, I get an error 407. Using any other version of python works fine. I am assuming it could be to do with this https://docs.python.org/3/whatsnew/3.8.html#notable-changes-in-python-3-8-12.

<!-- What you expected. -->

I should get a status of 200.

<!-- What happened instead. -->

I get a status code of 407.

\`\`\`python
import requests


r = requests.get('https://example.org/', proxies=proxies) # You will need a proxy to test with, I am using a paid service.
print(r.status_code)

\`\`\`

## System Information

\`\`\`json
{
  "chardet": {
    "version": null
  },
  "charset_normalizer": {
    "version": "2.0.9"
  },
  "cryptography": {
    "version": ""
  },
  "idna": {
    "version": "3.3"
  },
  "implementation": {
    "name": "CPython",
    "version": "3.8.12"
  },
  "platform": {
    "release": "5.13.0-7620-generic",
    "system": "Linux"
  },
  "pyOpenSSL": {
    "openssl_version": "",
    "version": null
  },
  "requests": {
    "version": "2.27.0"
  },
  "system_ssl": {
    "version": "101010cf"
  },
  "urllib3": {
    "version": "1.26.7"
  },
  "using_charset_normalizer": true,
  "using_pyopenssl": false
}
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 requests/utils.py** | 738 | 796| 456 | 456 | 7696 | 
| 2 | 2 requests/__init__.py | 87 | 153| 520 | 976 | 8914 | 
| 3 | 2 requests/__init__.py | 57 | 85| 377 | 1353 | 8914 | 
| 4 | 3 requests/status_codes.py | 25 | 103| 879 | 2232 | 10078 | 
| 5 | **3 requests/utils.py** | 1 | 112| 749 | 2981 | 10078 | 
| 6 | 4 requests/packages.py | 1 | 27| 219 | 3200 | 10297 | 
| 7 | 5 requests/auth.py | 100 | 125| 206 | 3406 | 12684 | 
| 8 | 6 requests/adapters.py | 438 | 539| 687 | 4093 | 16985 | 
| 9 | 7 requests/exceptions.py | 30 | 134| 508 | 4601 | 17666 | 
| 10 | 8 requests/compat.py | 1 | 82| 505 | 5106 | 18171 | 
| 11 | 8 requests/__init__.py | 1 | 55| 115 | 5221 | 18171 | 
| 12 | **8 requests/utils.py** | 799 | 834| 211 | 5432 | 18171 | 
| 13 | 9 requests/sessions.py | 272 | 299| 134 | 5566 | 24419 | 
| 14 | 10 setup.py | 68 | 119| 495 | 6061 | 25388 | 
| 15 | 10 requests/adapters.py | 1 | 53| 392 | 6453 | 25388 | 
| 16 | 11 requests/help.py | 1 | 65| 410 | 6863 | 26242 | 
| 17 | 11 requests/adapters.py | 373 | 393| 168 | 7031 | 26242 | 
| 18 | **11 requests/utils.py** | 837 | 861| 213 | 7244 | 26242 | 
| 19 | 11 requests/adapters.py | 293 | 328| 285 | 7529 | 26242 | 
| 20 | 12 requests/__version__.py | 1 | 15| 174 | 7703 | 26416 | 
| 21 | 12 requests/help.py | 68 | 136| 444 | 8147 | 26416 | 
| 22 | 12 requests/adapters.py | 330 | 357| 251 | 8398 | 26416 | 
| 23 | **12 requests/utils.py** | 179 | 233| 425 | 8823 | 26416 | 
| 24 | 12 setup.py | 35 | 66| 274 | 9097 | 26416 | 
| 25 | 12 requests/auth.py | 278 | 306| 244 | 9341 | 26416 | 
| 26 | 12 requests/sessions.py | 119 | 142| 243 | 9584 | 26416 | 
| 27 | **12 requests/utils.py** | 885 | 925| 252 | 9836 | 26416 | 
| 28 | 13 requests/cookies.py | 51 | 63| 133 | 9969 | 30466 | 
| 29 | 13 requests/auth.py | 216 | 232| 195 | 10164 | 30466 | 
| 30 | 13 requests/adapters.py | 167 | 202| 283 | 10447 | 30466 | 
| 31 | 13 requests/sessions.py | 691 | 718| 250 | 10697 | 30466 | 
| 32 | 13 requests/auth.py | 127 | 215| 800 | 11497 | 30466 | 
| 33 | 13 requests/auth.py | 28 | 69| 342 | 11839 | 30466 | 
| 34 | 13 requests/auth.py | 1 | 25| 116 | 11955 | 30466 | 
| 35 | 13 requests/auth.py | 234 | 276| 365 | 12320 | 30466 | 
| 36 | 14 requests/models.py | 1 | 59| 463 | 12783 | 37925 | 
| 37 | 14 requests/status_codes.py | 105 | 124| 145 | 12928 | 37925 | 
| 38 | 14 requests/adapters.py | 204 | 254| 437 | 13365 | 37925 | 
| 39 | 14 requests/adapters.py | 85 | 145| 546 | 13911 | 37925 | 
| 40 | 15 docs/conf.py | 1 | 114| 832 | 14743 | 40880 | 
| 41 | 15 requests/status_codes.py | 1 | 23| 139 | 14882 | 40880 | 
| 42 | 15 requests/cookies.py | 65 | 94| 208 | 15090 | 40880 | 
| 43 | 15 requests/exceptions.py | 1 | 27| 173 | 15263 | 40880 | 
| 44 | 15 requests/models.py | 756 | 776| 154 | 15417 | 40880 | 
| 45 | 15 requests/models.py | 544 | 564| 165 | 15582 | 40880 | 
| 46 | 15 requests/sessions.py | 301 | 321| 199 | 15781 | 40880 | 
| 47 | 15 requests/models.py | 688 | 700| 137 | 15918 | 40880 | 
| 48 | 15 requests/auth.py | 72 | 97| 163 | 16081 | 40880 | 
| 49 | 16 requests/api.py | 90 | 102| 126 | 16207 | 42495 | 
| 50 | 16 requests/models.py | 330 | 358| 220 | 16427 | 42495 | 
| 51 | 16 requests/models.py | 360 | 444| 710 | 17137 | 42495 | 
| 52 | 16 requests/models.py | 702 | 715| 130 | 17267 | 42495 | 
| 53 | 16 requests/sessions.py | 254 | 270| 168 | 17435 | 42495 | 
| 54 | **16 requests/utils.py** | 985 | 1003| 131 | 17566 | 42495 | 
| 55 | 16 docs/conf.py | 115 | 233| 979 | 18545 | 42495 | 
| 56 | 16 requests/cookies.py | 97 | 115| 134 | 18679 | 42495 | 
| 57 | 16 setup.py | 1 | 33| 200 | 18879 | 42495 | 
| 58 | 16 requests/models.py | 717 | 737| 193 | 19072 | 42495 | 
| 59 | **16 requests/utils.py** | 568 | 601| 197 | 19269 | 42495 | 
| 60 | **16 requests/utils.py** | 928 | 957| 347 | 19616 | 42495 | 
| 61 | 16 requests/sessions.py | 324 | 415| 722 | 20338 | 42495 | 
| 62 | 16 requests/sessions.py | 720 | 772| 346 | 20684 | 42495 | 
| 63 | 16 requests/api.py | 64 | 87| 198 | 20882 | 42495 | 
| 64 | 16 requests/adapters.py | 56 | 82| 261 | 21143 | 42495 | 
| 65 | 17 requests/certs.py | 1 | 19| 103 | 21246 | 42598 | 
| 66 | 17 requests/models.py | 919 | 974| 420 | 21666 | 42598 | 
| 67 | **17 requests/utils.py** | 670 | 713| 237 | 21903 | 42598 | 
| 68 | 17 requests/cookies.py | 1 | 49| 293 | 22196 | 42598 | 
| 69 | 18 docs/_themes/flask_theme_support.py | 1 | 87| 1276 | 23472 | 43874 | 
| 70 | 18 requests/models.py | 446 | 529| 593 | 24065 | 43874 | 
| 71 | 18 requests/models.py | 678 | 686| 108 | 24173 | 43874 | 
| 72 | **18 requests/utils.py** | 864 | 882| 106 | 24279 | 43874 | 
| 73 | 18 requests/sessions.py | 1 | 47| 335 | 24614 | 43874 | 
| 74 | 18 docs/conf.py | 235 | 387| 1022 | 25636 | 43874 | 
| 75 | 18 requests/adapters.py | 256 | 291| 280 | 25916 | 43874 | 
| 76 | **18 requests/utils.py** | 515 | 537| 155 | 26071 | 43874 | 
| 77 | 18 requests/cookies.py | 135 | 168| 239 | 26310 | 43874 | 
| 78 | 18 requests/models.py | 778 | 792| 149 | 26459 | 43874 | 
| 79 | 18 requests/models.py | 312 | 328| 156 | 26615 | 43874 | 
| 80 | 18 requests/sessions.py | 96 | 117| 241 | 26856 | 43874 | 
| 81 | 19 requests/_internal_utils.py | 1 | 43| 232 | 27088 | 44107 | 
| 82 | 19 requests/api.py | 135 | 160| 224 | 27312 | 44107 | 
| 83 | 19 requests/models.py | 596 | 676| 643 | 27955 | 44107 | 
| 84 | 19 requests/sessions.py | 533 | 564| 276 | 28231 | 44107 | 
| 85 | 19 requests/models.py | 845 | 881| 265 | 28496 | 44107 | 
| 86 | 19 requests/adapters.py | 395 | 436| 453 | 28949 | 44107 | 
| 87 | 19 requests/models.py | 825 | 843| 144 | 29093 | 44107 | 
| 88 | 19 requests/sessions.py | 144 | 252| 899 | 29992 | 44107 | 
| 89 | 19 requests/models.py | 566 | 593| 274 | 30266 | 44107 | 
| 90 | 19 requests/models.py | 176 | 197| 157 | 30423 | 44107 | 
| 91 | **19 requests/utils.py** | 1027 | 1057| 236 | 30659 | 44107 | 
| 92 | 19 requests/api.py | 120 | 132| 143 | 30802 | 44107 | 
| 93 | 19 requests/models.py | 274 | 310| 282 | 31084 | 44107 | 
| 94 | **19 requests/utils.py** | 540 | 565| 157 | 31241 | 44107 | 
| 95 | 19 requests/sessions.py | 613 | 689| 544 | 31785 | 44107 | 
| 96 | 19 requests/sessions.py | 457 | 531| 747 | 32532 | 44107 | 
| 97 | **19 requests/utils.py** | 383 | 415| 272 | 32804 | 44107 | 
| 98 | 19 requests/api.py | 105 | 117| 149 | 32953 | 44107 | 
| 99 | **19 requests/utils.py** | 1006 | 1024| 163 | 33116 | 44107 | 
| 100 | **19 requests/utils.py** | 115 | 176| 507 | 33623 | 44107 | 
| 101 | 19 requests/sessions.py | 591 | 611| 206 | 33829 | 44107 | 
| 102 | **19 requests/utils.py** | 470 | 487| 194 | 34023 | 44107 | 
| 103 | 19 requests/sessions.py | 579 | 589| 126 | 34149 | 44107 | 
| 104 | 19 requests/models.py | 200 | 271| 584 | 34733 | 44107 | 
| 105 | 19 requests/cookies.py | 218 | 335| 799 | 35532 | 44107 | 
| **-> 106 <-** | **19 requests/utils.py** | 960 | 982| 206 | 35738 | 44107 | 
| 107 | 19 requests/api.py | 1 | 61| 734 | 36472 | 44107 | 
| 108 | 19 requests/adapters.py | 147 | 165| 203 | 36675 | 44107 | 
| 109 | 19 requests/cookies.py | 118 | 132| 150 | 36825 | 44107 | 
| 110 | 19 requests/models.py | 62 | 109| 294 | 37119 | 44107 | 
| 111 | 20 requests/structures.py | 1 | 86| 579 | 37698 | 44801 | 
| 112 | 20 requests/cookies.py | 401 | 438| 286 | 37984 | 44801 | 
| 113 | 21 requests/hooks.py | 1 | 35| 176 | 38160 | 44978 | 
| 114 | 21 requests/models.py | 883 | 917| 340 | 38500 | 44978 | 
| 115 | 21 requests/cookies.py | 171 | 199| 255 | 38755 | 44978 | 
| 116 | **21 requests/utils.py** | 633 | 652| 203 | 38958 | 44978 | 
| 117 | 21 requests/sessions.py | 566 | 577| 153 | 39111 | 44978 | 
| 118 | **21 requests/utils.py** | 604 | 630| 213 | 39324 | 44978 | 
| 119 | 21 requests/cookies.py | 356 | 374| 187 | 39511 | 44978 | 
| 120 | 21 requests/sessions.py | 50 | 93| 324 | 39835 | 44978 | 
| 121 | **21 requests/utils.py** | 655 | 667| 180 | 40015 | 44978 | 
| 122 | 21 requests/cookies.py | 376 | 399| 272 | 40287 | 44978 | 
| 123 | 21 requests/adapters.py | 359 | 371| 141 | 40428 | 44978 | 
| 124 | **21 requests/utils.py** | 351 | 380| 256 | 40684 | 44978 | 
| 125 | 21 requests/models.py | 531 | 542| 153 | 40837 | 44978 | 
| 126 | 21 requests/models.py | 794 | 823| 196 | 41033 | 44978 | 
| 127 | 21 requests/cookies.py | 441 | 474| 258 | 41291 | 44978 | 
| 128 | 21 requests/sessions.py | 417 | 455| 308 | 41599 | 44978 | 
| 129 | 21 requests/models.py | 111 | 173| 509 | 42108 | 44978 | 
| 130 | 21 requests/structures.py | 89 | 106| 114 | 42222 | 44978 | 
| 131 | **21 requests/utils.py** | 490 | 512| 168 | 42390 | 44978 | 
| 132 | 21 requests/models.py | 739 | 754| 182 | 42572 | 44978 | 
| 133 | **21 requests/utils.py** | 236 | 278| 416 | 42988 | 44978 | 
| 134 | **21 requests/utils.py** | 418 | 441| 286 | 43274 | 44978 | 
| 135 | 21 requests/cookies.py | 508 | 526| 170 | 43444 | 44978 | 
| 136 | **21 requests/utils.py** | 322 | 348| 182 | 43626 | 44978 | 
| 137 | **21 requests/utils.py** | 444 | 467| 140 | 43766 | 44978 | 
| 138 | 21 requests/cookies.py | 337 | 354| 193 | 43959 | 44978 | 
| 139 | 21 requests/cookies.py | 477 | 505| 246 | 44205 | 44978 | 
| 140 | 21 requests/cookies.py | 201 | 216| 152 | 44357 | 44978 | 
| 141 | 21 requests/cookies.py | 529 | 550| 163 | 44520 | 44978 | 
| 142 | **21 requests/utils.py** | 716 | 735| 140 | 44660 | 44978 | 
| 143 | **21 requests/utils.py** | 281 | 319| 282 | 44942 | 44978 | 


### Hint

```
Hi @flameaway, it’s hard to tell what exactly is happening here without more info. Could you verify this issue occurs in both Requests 2.26.0 and urllib3 1.25.11?

It could very well be related to the ipaddress change, I’d just like to rule out other potential factors before we start down that path.
Requests 2.26.0 returns status 200. Either version of urllib (1.25.11, 1.26.7) work with it. Requests 2.27.0 returns the 407 error with either urllib version.
Thanks for confirming that! It sounds like this may be localized to today's release (2.27.0) We made some minor refactorings to how we handle proxies on redirects in https://github.com/psf/requests/pull/5924. I'm not seeing anything off immediately, so this will need some digging. For the meantime, using 2.26.0 is likely the short term solution.

I just want to clarify one more comment.

> When using proxies in python 3.8.12, I get an error 407. Using any other version of python works fine.

Does this mean 2.27.0 works on all other Python versions besides 3.8.12, or did you only test 2.27.0 with 3.8.12? I want to confirm we're not dealing with a requests release issue AND a python release issue.
> Does this mean 2.27.0 works on all other Python versions besides 3.8.12, or did you only test 2.27.0 with 3.8.12? I want to confirm we're not dealing with a requests release issue AND a python release issue.

It seems to only be having issues on 2.27.0. I didn't realize, but python 3.9.7 defaulted to installing requests 2.26.0. 
Confirming that this error also occurs with requests 2.27.0 and Python 3.8.9
To be clear, there is way too little information in here as it stands to be able to debug this from our end.
Did a bisect and found: 
\`\`\`
ef59aa0227bf463f0ed3d752b26db9b3acc64afb is the first bad commit
commit ef59aa0227bf463f0ed3d752b26db9b3acc64afb
Author: Nate Prewitt <Nate.Prewitt@gmail.com>
Date:   Thu Aug 26 22:06:48 2021 -0700

    Move from urlparse to parse_url for prepending schemes

 requests/utils.py   | 21 +++++++++++++++------
 tests/test_utils.py |  1 +
 2 files changed, 16 insertions(+), 6 deletions(-)
\`\`\`

I'm using a proxy from QuotaGuard, so it has auth.
So after doing some digging, in my case the params passed to `urlunparse` in `prepend_scheme_if_needed` went from:
scheme: `http`
netloc: `user:pwd@host:port`
To:
scheme: `http`
netloc: `host:port`
So the auth is lost from netloc here. The auth is still parsed and stored in the auth var, however.

Adding this to `prepend_scheme_if_needed` resolves, but unaware of any other issues that might cause:
\`\`\`
if auth:
    netloc = '@'.join([auth, netloc])
\`\`\`
Same issue here.
Since 2.27.0 with Python 3.8

I confirm @adamp01 investigation with mine. `user:pwd` seem to be lost during proxy parsing. I always get a 
`Tunnel connection failed: 407 Proxy Authentication Required`
Thanks for confirming @racam and @adamp01. We switched to using urllib3’s parser for proxies because of some recent changes to the standard lib `urlparse` around schemes. It looks like the two differ on their definition of `netloc`. I’m working on a patch to try to get this resolved.
Thank you for helping debug this @racam and @adamp01 
```

## Patch

```diff
diff --git a/requests/utils.py b/requests/utils.py
--- a/requests/utils.py
+++ b/requests/utils.py
@@ -974,6 +974,10 @@ def prepend_scheme_if_needed(url, new_scheme):
     if not netloc:
         netloc, path = path, netloc
 
+    if auth:
+        # parse_url doesn't provide the netloc with auth
+        # so we'll add it ourselves.
+        netloc = '@'.join([auth, netloc])
     if scheme is None:
         scheme = new_scheme
     if path is None:

```

## Test Patch

```diff
diff --git a/tests/test_utils.py b/tests/test_utils.py
--- a/tests/test_utils.py
+++ b/tests/test_utils.py
@@ -602,6 +602,14 @@ def test_parse_header_links(value, expected):
         ('example.com/path', 'http://example.com/path'),
         ('//example.com/path', 'http://example.com/path'),
         ('example.com:80', 'http://example.com:80'),
+        (
+            'http://user:pass@example.com/path?query',
+            'http://user:pass@example.com/path?query'
+        ),
+        (
+            'http://user@example.com/path?query',
+            'http://user@example.com/path?query'
+        )
     ))
 def test_prepend_scheme_if_needed(value, expected):
     assert prepend_scheme_if_needed(value, 'http') == expected

```


## Code snippets

### 1 - requests/utils.py:

Start line: 738, End line: 796

```python
def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """
    # Prioritize lowercase environment variables over uppercase
    # to keep a consistent behaviour with other http projects (curl, wget).
    get_proxy = lambda k: os.environ.get(k) or os.environ.get(k.upper())

    # First check whether no_proxy is defined. If it is, check that the URL
    # we're getting isn't in the no_proxy list.
    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy('no_proxy')
    parsed = urlparse(url)

    if parsed.hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls.
        return True

    if no_proxy:
        # We need to check whether we match here. We need to see if we match
        # the end of the hostname, both with and without the port.
        no_proxy = (
            host for host in no_proxy.replace(' ', '').split(',') if host
        )

        if is_ipv4_address(parsed.hostname):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed.hostname, proxy_ip):
                        return True
                elif parsed.hostname == proxy_ip:
                    # If no_proxy ip was defined in plain IP notation instead of cidr notation &
                    # matches the IP of the index
                    return True
        else:
            host_with_port = parsed.hostname
            if parsed.port:
                host_with_port += ':{}'.format(parsed.port)

            for host in no_proxy:
                if parsed.hostname.endswith(host) or host_with_port.endswith(host):
                    # The URL does match something in no_proxy, so we don't want
                    # to apply the proxies on this URL.
                    return True

    with set_environ('no_proxy', no_proxy_arg):
        # parsed.hostname can be `None` in cases such as a file URI.
        try:
            bypass = proxy_bypass(parsed.hostname)
        except (TypeError, socket.gaierror):
            bypass = False

    if bypass:
        return True

    return False
```
### 2 - requests/__init__.py:

Start line: 87, End line: 153

```python
def _check_cryptography(cryptography_version):
    # cryptography < 1.3.4
    try:
        cryptography_version = list(map(int, cryptography_version.split('.')))
    except ValueError:
        return

    if cryptography_version < [1, 3, 4]:
        warning = 'Old version of cryptography ({}) may cause slowdown.'.format(cryptography_version)
        warnings.warn(warning, RequestsDependencyWarning)

# Check imported dependencies for compatibility.
try:
    check_compatibility(urllib3.__version__, chardet_version, charset_normalizer_version)
except (AssertionError, ValueError):
    warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
                  "version!".format(urllib3.__version__, chardet_version, charset_normalizer_version),
                  RequestsDependencyWarning)

# Attempt to enable urllib3's fallback for SNI support
# if the standard library doesn't support SNI or the
# 'ssl' library isn't available.
try:
    try:
        import ssl
    except ImportError:
        ssl = None

    if not getattr(ssl, "HAS_SNI", False):
        from urllib3.contrib import pyopenssl
        pyopenssl.inject_into_urllib3()

        # Check cryptography version
        from cryptography import __version__ as cryptography_version
        _check_cryptography(cryptography_version)
except ImportError:
    pass

# urllib3's DependencyWarnings should be silenced.
from urllib3.exceptions import DependencyWarning
warnings.simplefilter('ignore', DependencyWarning)

from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __build__, __author__, __author_email__, __license__
from .__version__ import __copyright__, __cake__

from . import utils
from . import packages
from .models import Request, Response, PreparedRequest
from .api import request, get, head, post, patch, put, delete, options
from .sessions import session, Session
from .status_codes import codes
from .exceptions import (
    RequestException, Timeout, URLRequired,
    TooManyRedirects, HTTPError, ConnectionError,
    FileModeWarning, ConnectTimeout, ReadTimeout, JSONDecodeError
)

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())

# FileModeWarnings go off per the default.
warnings.simplefilter('default', FileModeWarning, append=True)
```
### 3 - requests/__init__.py:

Start line: 57, End line: 85

```python
def check_compatibility(urllib3_version, chardet_version, charset_normalizer_version):
    urllib3_version = urllib3_version.split('.')
    assert urllib3_version != ['dev']  # Verify urllib3 isn't installed from git.

    # Sometimes, urllib3 only reports its version as 16.1.
    if len(urllib3_version) == 2:
        urllib3_version.append('0')

    # Check urllib3 for compatibility.
    major, minor, patch = urllib3_version  # noqa: F811
    major, minor, patch = int(major), int(minor), int(patch)
    # urllib3 >= 1.21.1, <= 1.26
    assert major == 1
    assert minor >= 21
    assert minor <= 26

    # Check charset_normalizer for compatibility.
    if chardet_version:
        major, minor, patch = chardet_version.split('.')[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # chardet_version >= 3.0.2, < 5.0.0
        assert (3, 0, 2) <= (major, minor, patch) < (5, 0, 0)
    elif charset_normalizer_version:
        major, minor, patch = charset_normalizer_version.split('.')[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # charset_normalizer >= 2.0.0 < 3.0.0
        assert (2, 0, 0) <= (major, minor, patch) < (3, 0, 0)
    else:
        raise Exception("You need either charset_normalizer or chardet installed")
```
### 4 - requests/status_codes.py:

Start line: 25, End line: 103

```python
_codes = {

    # Informational.
    100: ('continue',),
    101: ('switching_protocols',),
    102: ('processing',),
    103: ('checkpoint',),
    122: ('uri_too_long', 'request_uri_too_long'),
    200: ('ok', 'okay', 'all_ok', 'all_okay', 'all_good', '\\o/', '✓'),
    201: ('created',),
    202: ('accepted',),
    203: ('non_authoritative_info', 'non_authoritative_information'),
    204: ('no_content',),
    205: ('reset_content', 'reset'),
    206: ('partial_content', 'partial'),
    207: ('multi_status', 'multiple_status', 'multi_stati', 'multiple_stati'),
    208: ('already_reported',),
    226: ('im_used',),

    # Redirection.
    300: ('multiple_choices',),
    301: ('moved_permanently', 'moved', '\\o-'),
    302: ('found',),
    303: ('see_other', 'other'),
    304: ('not_modified',),
    305: ('use_proxy',),
    306: ('switch_proxy',),
    307: ('temporary_redirect', 'temporary_moved', 'temporary'),
    308: ('permanent_redirect',
          'resume_incomplete', 'resume',),  # These 2 to be removed in 3.0

    # Client Error.
    400: ('bad_request', 'bad'),
    401: ('unauthorized',),
    402: ('payment_required', 'payment'),
    403: ('forbidden',),
    404: ('not_found', '-o-'),
    405: ('method_not_allowed', 'not_allowed'),
    406: ('not_acceptable',),
    407: ('proxy_authentication_required', 'proxy_auth', 'proxy_authentication'),
    408: ('request_timeout', 'timeout'),
    409: ('conflict',),
    410: ('gone',),
    411: ('length_required',),
    412: ('precondition_failed', 'precondition'),
    413: ('request_entity_too_large',),
    414: ('request_uri_too_large',),
    415: ('unsupported_media_type', 'unsupported_media', 'media_type'),
    416: ('requested_range_not_satisfiable', 'requested_range', 'range_not_satisfiable'),
    417: ('expectation_failed',),
    418: ('im_a_teapot', 'teapot', 'i_am_a_teapot'),
    421: ('misdirected_request',),
    422: ('unprocessable_entity', 'unprocessable'),
    423: ('locked',),
    424: ('failed_dependency', 'dependency'),
    425: ('unordered_collection', 'unordered'),
    426: ('upgrade_required', 'upgrade'),
    428: ('precondition_required', 'precondition'),
    429: ('too_many_requests', 'too_many'),
    431: ('header_fields_too_large', 'fields_too_large'),
    444: ('no_response', 'none'),
    449: ('retry_with', 'retry'),
    450: ('blocked_by_windows_parental_controls', 'parental_controls'),
    451: ('unavailable_for_legal_reasons', 'legal_reasons'),
    499: ('client_closed_request',),

    # Server Error.
    500: ('internal_server_error', 'server_error', '/o\\', '✗'),
    501: ('not_implemented',),
    502: ('bad_gateway',),
    503: ('service_unavailable', 'unavailable'),
    504: ('gateway_timeout',),
    505: ('http_version_not_supported', 'http_version'),
    506: ('variant_also_negotiates',),
    507: ('insufficient_storage',),
    509: ('bandwidth_limit_exceeded', 'bandwidth'),
    510: ('not_extended',),
    511: ('network_authentication_required', 'network_auth', 'network_authentication'),
}
```
### 5 - requests/utils.py:

Start line: 1, End line: 112

```python
# -*- coding: utf-8 -*-

"""
requests.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Requests
that are also useful for external consumption.
"""

import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from urllib3.util import make_headers
from urllib3.util import parse_url

from .__version__ import __version__
from . import certs
# to_native_string is unused here, but imported here for backwards compatibility
from ._internal_utils import to_native_string
from .compat import parse_http_list as _parse_list_header
from .compat import (
    quote, urlparse, bytes, str, unquote, getproxies,
    proxy_bypass, urlunparse, basestring, integer_types, is_py3,
    proxy_bypass_environment, getproxies_environment, Mapping)
from .cookies import cookiejar_from_dict
from .structures import CaseInsensitiveDict
from .exceptions import (
    InvalidURL, InvalidHeader, FileModeWarning, UnrewindableBodyError)

NETRC_FILES = ('.netrc', '_netrc')

DEFAULT_CA_BUNDLE_PATH = certs.where()

DEFAULT_PORTS = {'http': 80, 'https': 443}

# Ensure that ', ' is used to preserve previous delimiter behavior.
DEFAULT_ACCEPT_ENCODING = ", ".join(
    re.split(r",\s*", make_headers(accept_encoding=True)["accept-encoding"])
)


if sys.platform == 'win32':
    # provide a proxy_bypass version on Windows without DNS lookups

    def proxy_bypass_registry(host):
        try:
            if is_py3:
                import winreg
            else:
                import _winreg as winreg
        except ImportError:
            return False

        try:
            internetSettings = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                r'Software\Microsoft\Windows\CurrentVersion\Internet Settings')
            # ProxyEnable could be REG_SZ or REG_DWORD, normalizing it
            proxyEnable = int(winreg.QueryValueEx(internetSettings,
                                              'ProxyEnable')[0])
            # ProxyOverride is almost always a string
            proxyOverride = winreg.QueryValueEx(internetSettings,
                                                'ProxyOverride')[0]
        except OSError:
            return False
        if not proxyEnable or not proxyOverride:
            return False

        # make a check value list from the registry entry: replace the
        # '<local>' string by the localhost entry and the corresponding
        # canonical entry.
        proxyOverride = proxyOverride.split(';')
        # now check if we match one of the registry values.
        for test in proxyOverride:
            if test == '<local>':
                if '.' not in host:
                    return True
            test = test.replace(".", r"\.")     # mask dots
            test = test.replace("*", r".*")     # change glob sequence
            test = test.replace("?", r".")      # change glob char
            if re.match(test, host, re.I):
                return True
        return False

    def proxy_bypass(host):  # noqa
        """Return True, if the host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.
        """
        if getproxies_environment():
            return proxy_bypass_environment(host)
        else:
            return proxy_bypass_registry(host)


def dict_to_sequence(d):
    """Returns an internal sequence dictionary update."""

    if hasattr(d, 'items'):
        d = d.items()

    return d
```
### 6 - requests/packages.py:

Start line: 1, End line: 27

```python
import sys

try:
    import chardet
except ImportError:
    import charset_normalizer as chardet
    import warnings

    warnings.filterwarnings('ignore', 'Trying to detect', module='charset_normalizer')

# This code exists for backwards compatibility reasons.
# I don't like it either. Just look the other way. :)

for package in ('urllib3', 'idna'):
    locals()[package] = __import__(package)
    # This traversal is apparently necessary such that the identities are
    # preserved (requests.packages.urllib3.* is urllib3.*)
    for mod in list(sys.modules):
        if mod == package or mod.startswith(package + '.'):
            sys.modules['requests.packages.' + mod] = sys.modules[mod]

target = chardet.__name__
for mod in list(sys.modules):
    if mod == target or mod.startswith(target + '.'):
        sys.modules['requests.packages.' + target.replace(target, 'chardet')] = sys.modules[mod]
# Kinda cool, though, right?
```
### 7 - requests/auth.py:

Start line: 100, End line: 125

```python
class HTTPProxyAuth(HTTPBasicAuth):
    """Attaches HTTP Proxy Authentication to a given Request object."""

    def __call__(self, r):
        r.headers['Proxy-Authorization'] = _basic_auth_str(self.username, self.password)
        return r


class HTTPDigestAuth(AuthBase):
    """Attaches HTTP Digest Authentication to the given Request object."""

    def __init__(self, username, password):
        self.username = username
        self.password = password
        # Keep state in per-thread local storage
        self._thread_local = threading.local()

    def init_per_thread_state(self):
        # Ensure state is initialized just once per-thread
        if not hasattr(self._thread_local, 'init'):
            self._thread_local.init = True
            self._thread_local.last_nonce = ''
            self._thread_local.nonce_count = 0
            self._thread_local.chal = {}
            self._thread_local.pos = None
            self._thread_local.num_401_calls = None
```
### 8 - requests/adapters.py:

Start line: 438, End line: 539

```python
class HTTPAdapter(BaseAdapter):

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        # ... other code

        try:
            if not chunked:
                resp = conn.urlopen(
                    method=request.method,
                    url=url,
                    body=request.body,
                    headers=request.headers,
                    redirect=False,
                    assert_same_host=False,
                    preload_content=False,
                    decode_content=False,
                    retries=self.max_retries,
                    timeout=timeout
                )

            # Send the request.
            else:
                if hasattr(conn, 'proxy_pool'):
                    conn = conn.proxy_pool

                low_conn = conn._get_conn(timeout=DEFAULT_POOL_TIMEOUT)

                try:
                    skip_host = 'Host' in request.headers
                    low_conn.putrequest(request.method,
                                        url,
                                        skip_accept_encoding=True,
                                        skip_host=skip_host)

                    for header, value in request.headers.items():
                        low_conn.putheader(header, value)

                    low_conn.endheaders()

                    for i in request.body:
                        low_conn.send(hex(len(i))[2:].encode('utf-8'))
                        low_conn.send(b'\r\n')
                        low_conn.send(i)
                        low_conn.send(b'\r\n')
                    low_conn.send(b'0\r\n\r\n')

                    # Receive the response from the server
                    try:
                        # For Python 2.7, use buffering of HTTP responses
                        r = low_conn.getresponse(buffering=True)
                    except TypeError:
                        # For compatibility with Python 3.3+
                        r = low_conn.getresponse()

                    resp = HTTPResponse.from_httplib(
                        r,
                        pool=conn,
                        connection=low_conn,
                        preload_content=False,
                        decode_content=False
                    )
                except:
                    # If we hit any problems here, clean up the connection.
                    # Then, reraise so that we can handle the actual exception.
                    low_conn.close()
                    raise

        except (ProtocolError, socket.error) as err:
            raise ConnectionError(err, request=request)

        except MaxRetryError as e:
            if isinstance(e.reason, ConnectTimeoutError):
                # TODO: Remove this in 3.0.0: see #2811
                if not isinstance(e.reason, NewConnectionError):
                    raise ConnectTimeout(e, request=request)

            if isinstance(e.reason, ResponseError):
                raise RetryError(e, request=request)

            if isinstance(e.reason, _ProxyError):
                raise ProxyError(e, request=request)

            if isinstance(e.reason, _SSLError):
                # This branch is for urllib3 v1.22 and later.
                raise SSLError(e, request=request)

            raise ConnectionError(e, request=request)

        except ClosedPoolError as e:
            raise ConnectionError(e, request=request)

        except _ProxyError as e:
            raise ProxyError(e)

        except (_SSLError, _HTTPError) as e:
            if isinstance(e, _SSLError):
                # This branch is for urllib3 versions earlier than v1.22
                raise SSLError(e, request=request)
            elif isinstance(e, ReadTimeoutError):
                raise ReadTimeout(e, request=request)
            elif isinstance(e, _InvalidHeader):
                raise InvalidHeader(e, request=request)
            else:
                raise

        return self.build_response(request, resp)
```
### 9 - requests/exceptions.py:

Start line: 30, End line: 134

```python
class InvalidJSONError(RequestException):
    """A JSON error occurred."""


class JSONDecodeError(InvalidJSONError, CompatJSONDecodeError):
    """Couldn't decode the text into json"""


class HTTPError(RequestException):
    """An HTTP error occurred."""


class ConnectionError(RequestException):
    """A Connection error occurred."""


class ProxyError(ConnectionError):
    """A proxy error occurred."""


class SSLError(ConnectionError):
    """An SSL error occurred."""


class Timeout(RequestException):
    """The request timed out.

    Catching this error will catch both
    :exc:`~requests.exceptions.ConnectTimeout` and
    :exc:`~requests.exceptions.ReadTimeout` errors.
    """


class ConnectTimeout(ConnectionError, Timeout):
    """The request timed out while trying to connect to the remote server.

    Requests that produced this error are safe to retry.
    """


class ReadTimeout(Timeout):
    """The server did not send any data in the allotted amount of time."""


class URLRequired(RequestException):
    """A valid URL is required to make a request."""


class TooManyRedirects(RequestException):
    """Too many redirects."""


class MissingSchema(RequestException, ValueError):
    """The URL scheme (e.g. http or https) is missing."""


class InvalidSchema(RequestException, ValueError):
    """The URL scheme provided is either invalid or unsupported."""


class InvalidURL(RequestException, ValueError):
    """The URL provided was somehow invalid."""


class InvalidHeader(RequestException, ValueError):
    """The header value provided was somehow invalid."""


class InvalidProxyURL(InvalidURL):
    """The proxy URL provided is invalid."""


class ChunkedEncodingError(RequestException):
    """The server declared chunked encoding but sent an invalid chunk."""


class ContentDecodingError(RequestException, BaseHTTPError):
    """Failed to decode response content."""


class StreamConsumedError(RequestException, TypeError):
    """The content for this response was already consumed."""


class RetryError(RequestException):
    """Custom retries logic failed"""


class UnrewindableBodyError(RequestException):
    """Requests encountered an error when trying to rewind a body."""

# Warnings


class RequestsWarning(Warning):
    """Base warning for Requests."""


class FileModeWarning(RequestsWarning, DeprecationWarning):
    """A file was opened in text mode, but Requests determined its binary length."""


class RequestsDependencyWarning(RequestsWarning):
    """An imported dependency doesn't match the expected version range."""
```
### 10 - requests/compat.py:

Start line: 1, End line: 82

```python
# -*- coding: utf-8 -*-

"""
requests.compat
~~~~~~~~~~~~~~~

This module handles import compatibility issues between Python 2 and
Python 3.
"""

try:
    import chardet
except ImportError:
    import charset_normalizer as chardet

import sys

# -------
# Pythons
# -------

# Syntax sugar.
_ver = sys.version_info

#: Python 2.x?
is_py2 = (_ver[0] == 2)

#: Python 3.x?
is_py3 = (_ver[0] == 3)

has_simplejson = False
try:
    import simplejson as json
    has_simplejson = True
except ImportError:
    import json

# ---------
# Specifics
# ---------

if is_py2:
    from urllib import (
        quote, unquote, quote_plus, unquote_plus, urlencode, getproxies,
        proxy_bypass, proxy_bypass_environment, getproxies_environment)
    from urlparse import urlparse, urlunparse, urljoin, urlsplit, urldefrag
    from urllib2 import parse_http_list
    import cookielib
    from Cookie import Morsel
    from StringIO import StringIO
    # Keep OrderedDict for backwards compatibility.
    from collections import Callable, Mapping, MutableMapping, OrderedDict

    builtin_str = str
    bytes = str
    str = unicode
    basestring = basestring
    numeric_types = (int, long, float)
    integer_types = (int, long)
    JSONDecodeError = ValueError

elif is_py3:
    from urllib.parse import urlparse, urlunparse, urljoin, urlsplit, urlencode, quote, unquote, quote_plus, unquote_plus, urldefrag
    from urllib.request import parse_http_list, getproxies, proxy_bypass, proxy_bypass_environment, getproxies_environment
    from http import cookiejar as cookielib
    from http.cookies import Morsel
    from io import StringIO
    # Keep OrderedDict for backwards compatibility.
    from collections import OrderedDict
    from collections.abc import Callable, Mapping, MutableMapping
    if has_simplejson:
        from simplejson import JSONDecodeError
    else:
        from json import JSONDecodeError

    builtin_str = str
    str = str
    bytes = bytes
    basestring = (str, bytes)
    numeric_types = (int, float)
    integer_types = (int,)
```
### 12 - requests/utils.py:

Start line: 799, End line: 834

```python
def get_environ_proxies(url, no_proxy=None):
    """
    Return a dict of environment proxies.

    :rtype: dict
    """
    if should_bypass_proxies(url, no_proxy=no_proxy):
        return {}
    else:
        return getproxies()


def select_proxy(url, proxies):
    """Select a proxy for the url, if applicable.

    :param url: The url being for the request
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    """
    proxies = proxies or {}
    urlparts = urlparse(url)
    if urlparts.hostname is None:
        return proxies.get(urlparts.scheme, proxies.get('all'))

    proxy_keys = [
        urlparts.scheme + '://' + urlparts.hostname,
        urlparts.scheme,
        'all://' + urlparts.hostname,
        'all',
    ]
    proxy = None
    for proxy_key in proxy_keys:
        if proxy_key in proxies:
            proxy = proxies[proxy_key]
            break

    return proxy
```
### 18 - requests/utils.py:

Start line: 837, End line: 861

```python
def resolve_proxies(request, proxies, trust_env=True):
    """This method takes proxy information from a request and configuration
    input to resolve a mapping of target proxies. This will consider settings
    such a NO_PROXY to strip proxy configurations.

    :param request: Request or PreparedRequest
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    :param trust_env: Boolean declaring whether to trust environment configs

    :rtype: dict
    """
    proxies = proxies if proxies is not None else {}
    url = request.url
    scheme = urlparse(url).scheme
    no_proxy = proxies.get('no_proxy')
    new_proxies = proxies.copy()

    if trust_env and not should_bypass_proxies(url, no_proxy=no_proxy):
        environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)

        proxy = environ_proxies.get(scheme, environ_proxies.get('all'))

        if proxy:
            new_proxies.setdefault(scheme, proxy)
    return new_proxies
```
### 23 - requests/utils.py:

Start line: 179, End line: 233

```python
def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""

    netrc_file = os.environ.get('NETRC')
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = ('~/{}'.format(f) for f in NETRC_FILES)

    try:
        from netrc import netrc, NetrcParseError

        netrc_path = None

        for f in netrc_locations:
            try:
                loc = os.path.expanduser(f)
            except KeyError:
                # os.path.expanduser can fail when $HOME is undefined and
                # getpwuid fails. See https://bugs.python.org/issue20164 &
                # https://github.com/psf/requests/issues/1846
                return

            if os.path.exists(loc):
                netrc_path = loc
                break

        # Abort early if there isn't one.
        if netrc_path is None:
            return

        ri = urlparse(url)

        # Strip port numbers from netloc. This weird `if...encode`` dance is
        # used for Python 3.2, which doesn't support unicode literals.
        splitstr = b':'
        if isinstance(url, str):
            splitstr = splitstr.decode('ascii')
        host = ri.netloc.split(splitstr)[0]

        try:
            _netrc = netrc(netrc_path).authenticators(host)
            if _netrc:
                # Return with login / password
                login_i = (0 if _netrc[0] else 1)
                return (_netrc[login_i], _netrc[2])
        except (NetrcParseError, IOError):
            # If there was a parsing error or a permissions issue reading the file,
            # we'll just skip netrc auth unless explicitly asked to raise errors.
            if raise_errors:
                raise

    # App Engine hackiness.
    except (ImportError, AttributeError):
        pass
```
### 27 - requests/utils.py:

Start line: 885, End line: 925

```python
def parse_header_links(value):
    """Return a list of parsed link headers proxies.

    i.e. Link: <http:/.../front.jpeg>; rel=front; type="image/jpeg",<http://.../back.jpeg>; rel=back;type="image/jpeg"

    :rtype: list
    """

    links = []

    replace_chars = ' \'"'

    value = value.strip(replace_chars)
    if not value:
        return links

    for val in re.split(', *<', value):
        try:
            url, params = val.split(';', 1)
        except ValueError:
            url, params = val, ''

        link = {'url': url.strip('<> \'"')}

        for param in params.split(';'):
            try:
                key, value = param.split('=')
            except ValueError:
                break

            link[key.strip(replace_chars)] = value.strip(replace_chars)

        links.append(link)

    return links


# Null bytes; no need to recreate these on each call to guess_json_utf
_null = '\x00'.encode('ascii')  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3
```
### 54 - requests/utils.py:

Start line: 985, End line: 1003

```python
def get_auth_from_url(url):
    """Given a url with authentication components, extract them into a tuple of
    username,password.

    :rtype: (str,str)
    """
    parsed = urlparse(url)

    try:
        auth = (unquote(parsed.username), unquote(parsed.password))
    except (AttributeError, TypeError):
        auth = ('', '')

    return auth


# Moved outside of function to avoid recompile every call
_CLEAN_HEADER_REGEX_BYTE = re.compile(b'^\\S[^\\r\\n]*$|^$')
_CLEAN_HEADER_REGEX_STR = re.compile(r'^\S[^\r\n]*$|^$')
```
### 59 - requests/utils.py:

Start line: 568, End line: 601

```python
def get_unicode_from_response(r):
    """Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Tried:

    1. charset from content-type
    2. fall back and replace all unicode characters

    :rtype: str
    """
    warnings.warn((
        'In requests 3.0, get_unicode_from_response will be removed. For '
        'more information, please see the discussion on issue #2266. (This'
        ' warning should only appear once.)'),
        DeprecationWarning)

    tried_encodings = []

    # Try charset from content-type
    encoding = get_encoding_from_headers(r.headers)

    if encoding:
        try:
            return str(r.content, encoding)
        except UnicodeError:
            tried_encodings.append(encoding)

    # Fall back:
    try:
        return str(r.content, encoding, errors='replace')
    except TypeError:
        return r.content
```
### 60 - requests/utils.py:

Start line: 928, End line: 957

```python
def guess_json_utf(data):
    """
    :rtype: str
    """
    # JSON always starts with two ASCII characters, so detection is as
    # easy as counting the nulls and from their location and count
    # determine the encoding. Also detect a BOM, if present.
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return 'utf-32'     # BOM included
    if sample[:3] == codecs.BOM_UTF8:
        return 'utf-8-sig'  # BOM included, MS style (discouraged)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return 'utf-16'     # BOM included
    nullcount = sample.count(_null)
    if nullcount == 0:
        return 'utf-8'
    if nullcount == 2:
        if sample[::2] == _null2:   # 1st and 3rd are null
            return 'utf-16-be'
        if sample[1::2] == _null2:  # 2nd and 4th are null
            return 'utf-16-le'
        # Did not detect 2 valid UTF-16 ascii-range characters
    if nullcount == 3:
        if sample[:3] == _null3:
            return 'utf-32-be'
        if sample[1:] == _null3:
            return 'utf-32-le'
        # Did not detect a valid UTF-32 ascii-range character
    return None
```
### 67 - requests/utils.py:

Start line: 670, End line: 713

```python
def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :rtype: str
    """
    bits = 0xffffffff ^ (1 << 32 - mask) - 1
    return socket.inet_ntoa(struct.pack('>I', bits))


def is_ipv4_address(string_ip):
    """
    :rtype: bool
    """
    try:
        socket.inet_aton(string_ip)
    except socket.error:
        return False
    return True


def is_valid_cidr(string_network):
    """
    Very simple check of the cidr format in no_proxy variable.

    :rtype: bool
    """
    if string_network.count('/') == 1:
        try:
            mask = int(string_network.split('/')[1])
        except ValueError:
            return False

        if mask < 1 or mask > 32:
            return False

        try:
            socket.inet_aton(string_network.split('/')[0])
        except socket.error:
            return False
    else:
        return False
    return True
```
### 72 - requests/utils.py:

Start line: 864, End line: 882

```python
def default_user_agent(name="python-requests"):
    """
    Return a string representing the default user agent.

    :rtype: str
    """
    return '%s/%s' % (name, __version__)


def default_headers():
    """
    :rtype: requests.structures.CaseInsensitiveDict
    """
    return CaseInsensitiveDict({
        'User-Agent': default_user_agent(),
        'Accept-Encoding': DEFAULT_ACCEPT_ENCODING,
        'Accept': '*/*',
        'Connection': 'keep-alive',
    })
```
### 76 - requests/utils.py:

Start line: 515, End line: 537

```python
def get_encoding_from_headers(headers):
    """Returns encodings from given HTTP Header Dict.

    :param headers: dictionary to extract encoding from.
    :rtype: str
    """

    content_type = headers.get('content-type')

    if not content_type:
        return None

    content_type, params = _parse_content_type_header(content_type)

    if 'charset' in params:
        return params['charset'].strip("'\"")

    if 'text' in content_type:
        return 'ISO-8859-1'

    if 'application/json' in content_type:
        # Assume UTF-8 based on RFC 4627: https://www.ietf.org/rfc/rfc4627.txt since the charset was unset
        return 'utf-8'
```
### 91 - requests/utils.py:

Start line: 1027, End line: 1057

```python
def urldefragauth(url):
    """
    Given a url remove the fragment and the authentication part.

    :rtype: str
    """
    scheme, netloc, path, params, query, fragment = urlparse(url)

    # see func:`prepend_scheme_if_needed`
    if not netloc:
        netloc, path = path, netloc

    netloc = netloc.rsplit('@', 1)[-1]

    return urlunparse((scheme, netloc, path, params, query, ''))


def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    body_seek = getattr(prepared_request.body, 'seek', None)
    if body_seek is not None and isinstance(prepared_request._body_position, integer_types):
        try:
            body_seek(prepared_request._body_position)
        except (IOError, OSError):
            raise UnrewindableBodyError("An error occurred when rewinding request "
                                        "body for redirect.")
    else:
        raise UnrewindableBodyError("Unable to rewind request body for redirect.")
```
### 94 - requests/utils.py:

Start line: 540, End line: 565

```python
def stream_decode_response_unicode(iterator, r):
    """Stream decodes a iterator."""

    if r.encoding is None:
        for item in iterator:
            yield item
        return

    decoder = codecs.getincrementaldecoder(r.encoding)(errors='replace')
    for chunk in iterator:
        rv = decoder.decode(chunk)
        if rv:
            yield rv
    rv = decoder.decode(b'', final=True)
    if rv:
        yield rv


def iter_slices(string, slice_length):
    """Iterate over slices of a string."""
    pos = 0
    if slice_length is None or slice_length <= 0:
        slice_length = len(string)
    while pos < len(string):
        yield string[pos:pos + slice_length]
        pos += slice_length
```
### 97 - requests/utils.py:

Start line: 383, End line: 415

```python
# From mitsuhiko/werkzeug (used with permission).
def parse_dict_header(value):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict:

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    :param value: a string with a dict header.
    :return: :class:`dict`
    :rtype: dict
    """
    result = {}
    for item in _parse_list_header(value):
        if '=' not in item:
            result[item] = None
            continue
        name, value = item.split('=', 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result
```
### 99 - requests/utils.py:

Start line: 1006, End line: 1024

```python
def check_header_validity(header):
    """Verifies that header value is a string which doesn't contain
    leading whitespace or return characters. This prevents unintended
    header injection.

    :param header: tuple, in the format (name, value).
    """
    name, value = header

    if isinstance(value, bytes):
        pat = _CLEAN_HEADER_REGEX_BYTE
    else:
        pat = _CLEAN_HEADER_REGEX_STR
    try:
        if not pat.match(value):
            raise InvalidHeader("Invalid return character or leading space in header: %s" % name)
    except TypeError:
        raise InvalidHeader("Value for header {%s: %s} must be of type str or "
                            "bytes, not %s" % (name, value, type(value)))
```
### 100 - requests/utils.py:

Start line: 115, End line: 176

```python
def super_len(o):
    total_length = None
    current_position = 0

    if hasattr(o, '__len__'):
        total_length = len(o)

    elif hasattr(o, 'len'):
        total_length = o.len

    elif hasattr(o, 'fileno'):
        try:
            fileno = o.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # AttributeError is a surprising exception, seeing as how we've just checked
            # that `hasattr(o, 'fileno')`.  It happens for objects obtained via
            # `Tarfile.extractfile()`, per issue 5229.
            pass
        else:
            total_length = os.fstat(fileno).st_size

            # Having used fstat to determine the file length, we need to
            # confirm that this file was opened up in binary mode.
            if 'b' not in o.mode:
                warnings.warn((
                    "Requests has determined the content-length for this "
                    "request using the binary size of the file: however, the "
                    "file has been opened in text mode (i.e. without the 'b' "
                    "flag in the mode). This may lead to an incorrect "
                    "content-length. In Requests 3.0, support will be removed "
                    "for files in text mode."),
                    FileModeWarning
                )

    if hasattr(o, 'tell'):
        try:
            current_position = o.tell()
        except (OSError, IOError):
            # This can happen in some weird situations, such as when the file
            # is actually a special file descriptor like stdin. In this
            # instance, we don't know what the length is, so set it to zero and
            # let requests chunk it instead.
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, 'seek') and total_length is None:
                # StringIO and BytesIO have seek but no usable fileno
                try:
                    # seek to end of file
                    o.seek(0, 2)
                    total_length = o.tell()

                    # seek back to current position to support
                    # partially read file-like objects
                    o.seek(current_position or 0)
                except (OSError, IOError):
                    total_length = 0

    if total_length is None:
        total_length = 0

    return max(0, total_length - current_position)
```
### 102 - requests/utils.py:

Start line: 470, End line: 487

```python
def get_encodings_from_content(content):
    """Returns encodings from given content string.

    :param content: bytestring to extract encodings from.
    """
    warnings.warn((
        'In requests 3.0, get_encodings_from_content will be removed. For '
        'more information, please see the discussion on issue #2266. (This'
        ' warning should only appear once.)'),
        DeprecationWarning)

    charset_re = re.compile(r'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
    pragma_re = re.compile(r'<meta.*?content=["\']*;?charset=(.+?)["\'>]', flags=re.I)
    xml_re = re.compile(r'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

    return (charset_re.findall(content) +
            pragma_re.findall(content) +
            xml_re.findall(content))
```
### 106 - requests/utils.py:

Start line: 960, End line: 982

```python
def prepend_scheme_if_needed(url, new_scheme):
    """Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :rtype: str
    """
    parsed = parse_url(url)
    scheme, auth, host, port, path, query, fragment = parsed

    # A defect in urlparse determines that there isn't a netloc present in some
    # urls. We previously assumed parsing was overly cautious, and swapped the
    # netloc and path. Due to a lack of tests on the original defect, this is
    # maintained with parse_url for backwards compatibility.
    netloc = parsed.netloc
    if not netloc:
        netloc, path = path, netloc

    if scheme is None:
        scheme = new_scheme
    if path is None:
        path = ''

    return urlunparse((scheme, netloc, path, '', query, fragment))
```
### 116 - requests/utils.py:

Start line: 633, End line: 652

```python
def requote_uri(uri):
    """Re-quote the given URI.

    This function passes the given URI through an unquote/quote cycle to
    ensure that it is fully and consistently quoted.

    :rtype: str
    """
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        # Unquote only the unreserved characters
        # Then quote only illegal characters (do not quote reserved,
        # unreserved, or '%')
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        # We couldn't unquote the given URI, so let's try quoting it, but
        # there may be unquoted '%'s in the URI. We need to make sure they're
        # properly quoted so they do not cause issues elsewhere.
        return quote(uri, safe=safe_without_percent)
```
### 118 - requests/utils.py:

Start line: 604, End line: 630

```python
# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + "0123456789-._~")


def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.

    :rtype: str
    """
    parts = uri.split('%')
    for i in range(1, len(parts)):
        h = parts[i][0:2]
        if len(h) == 2 and h.isalnum():
            try:
                c = chr(int(h, 16))
            except ValueError:
                raise InvalidURL("Invalid percent-escape sequence: '%s'" % h)

            if c in UNRESERVED_SET:
                parts[i] = c + parts[i][2:]
            else:
                parts[i] = '%' + parts[i]
        else:
            parts[i] = '%' + parts[i]
    return ''.join(parts)
```
### 121 - requests/utils.py:

Start line: 655, End line: 667

```python
def address_in_network(ip, net):
    """This function allows you to check if an IP belongs to a network subnet

    Example: returns True if ip = 192.168.1.1 and net = 192.168.1.0/24
             returns False if ip = 192.168.1.1 and net = 192.168.100.0/24

    :rtype: bool
    """
    ipaddr = struct.unpack('=L', socket.inet_aton(ip))[0]
    netaddr, bits = net.split('/')
    netmask = struct.unpack('=L', socket.inet_aton(dotted_netmask(int(bits))))[0]
    network = struct.unpack('=L', socket.inet_aton(netaddr))[0] & netmask
    return (ipaddr & netmask) == (network & netmask)
```
### 124 - requests/utils.py:

Start line: 351, End line: 380

```python
# From mitsuhiko/werkzeug (used with permission).
def parse_list_header(value):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    :rtype: list
    """
    result = []
    for item in _parse_list_header(value):
        if item[:1] == item[-1:] == '"':
            item = unquote_header_value(item[1:-1])
        result.append(item)
    return result
```
### 131 - requests/utils.py:

Start line: 490, End line: 512

```python
def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of
         parameters
    """

    tokens = header.split(';')
    content_type, params = tokens[0].strip(), tokens[1:]
    params_dict = {}
    items_to_strip = "\"' "

    for param in params:
        param = param.strip()
        if param:
            key, value = param, True
            index_of_equals = param.find("=")
            if index_of_equals != -1:
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1:].strip(items_to_strip)
            params_dict[key.lower()] = value
    return content_type, params_dict
```
### 133 - requests/utils.py:

Start line: 236, End line: 278

```python
def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    name = getattr(obj, 'name', None)
    if (name and isinstance(name, basestring) and name[0] != '<' and
            name[-1] != '>'):
        return os.path.basename(name)


def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    if os.path.exists(path):
        # this is already a valid path, no need to do anything further
        return path

    # find the first valid part of the provided path and treat that as a zip archive
    # assume the rest of the path is the name of a member in the archive
    archive, member = os.path.split(path)
    while archive and not os.path.exists(archive):
        archive, prefix = os.path.split(archive)
        if not prefix:
            # If we don't check for an empty prefix after the split (in other words, archive remains unchanged after the split),
            # we _can_ end up in an infinite loop on a rare corner case affecting a small number of users
            break
        member = '/'.join([prefix, member])

    if not zipfile.is_zipfile(archive):
        return path

    zip_file = zipfile.ZipFile(archive)
    if member not in zip_file.namelist():
        return path

    # we have a valid zip archive and a valid member of that archive
    tmp = tempfile.gettempdir()
    extracted_path = os.path.join(tmp, member.split('/')[-1])
    if not os.path.exists(extracted_path):
        # use read + write to avoid the creating nested folders, we only want the file, avoids mkdir racing condition
        with atomic_open(extracted_path) as file_handler:
            file_handler.write(zip_file.read(member))
    return extracted_path
```
### 134 - requests/utils.py:

Start line: 418, End line: 441

```python
# From mitsuhiko/werkzeug (used with permission).
def unquote_header_value(value, is_filename=False):
    r"""Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.

    :param value: the header value to unquote.
    :rtype: str
    """
    if value and value[0] == value[-1] == '"':
        # this is not the real unquoting, but fixing this so that the
        # RFC is met will result in bugs with internet explorer and
        # probably some other browsers as well.  IE for example is
        # uploading files with "C:\foo\bar.txt" as filename
        value = value[1:-1]

        # if this is a filename and the starting characters look like
        # a UNC path, then just return the value without quotes.  Using the
        # replace sequence below on a UNC path has the effect of turning
        # the leading double slash into a single slash and then
        # _fix_ie_filename() doesn't work correctly.  See #458.
        if not is_filename or value[:2] != '\\\\':
            return value.replace('\\\\', '\\').replace('\\"', '"')
    return value
```
### 136 - requests/utils.py:

Start line: 322, End line: 348

```python
def to_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. If it can be, return a list of tuples, e.g.,

    ::

        >>> to_key_val_list([('key', 'val')])
        [('key', 'val')]
        >>> to_key_val_list({'key': 'val'})
        [('key', 'val')]
        >>> to_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples

    :rtype: list
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError('cannot encode objects that are not 2-tuples')

    if isinstance(value, Mapping):
        value = value.items()

    return list(value)
```
### 137 - requests/utils.py:

Start line: 444, End line: 467

```python
def dict_from_cookiejar(cj):
    """Returns a key/value dictionary from a CookieJar.

    :param cj: CookieJar object to extract cookies from.
    :rtype: dict
    """

    cookie_dict = {}

    for cookie in cj:
        cookie_dict[cookie.name] = cookie.value

    return cookie_dict


def add_dict_to_cookiejar(cj, cookie_dict):
    """Returns a CookieJar from a key/value dictionary.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """

    return cookiejar_from_dict(cookie_dict, cj)
```
### 142 - requests/utils.py:

Start line: 716, End line: 735

```python
@contextlib.contextmanager
def set_environ(env_name, value):
    """Set the environment variable 'env_name' to 'value'

    Save previous value, yield, and then restore the previous value stored in
    the environment variable 'env_name'.

    If 'value' is None, do nothing"""
    value_changed = value is not None
    if value_changed:
        old_value = os.environ.get(env_name)
        os.environ[env_name] = value
    try:
        yield
    finally:
        if value_changed:
            if old_value is None:
                del os.environ[env_name]
            else:
                os.environ[env_name] = old_value
```
### 143 - requests/utils.py:

Start line: 281, End line: 319

```python
@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""
    replacer = os.rename if sys.version_info[0] == 2 else os.replace
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        with os.fdopen(tmp_descriptor, 'wb') as tmp_handler:
            yield tmp_handler
        replacer(tmp_name, filename)
    except BaseException:
        os.remove(tmp_name)
        raise


def from_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. Unless it can not be represented as such, return an
    OrderedDict, e.g.,

    ::

        >>> from_key_val_list([('key', 'val')])
        OrderedDict([('key', 'val')])
        >>> from_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples
        >>> from_key_val_list({'key': 'val'})
        OrderedDict([('key', 'val')])

    :rtype: OrderedDict
    """
    if value is None:
        return None

    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError('cannot encode objects that are not 2-tuples')

    return OrderedDict(value)
```
