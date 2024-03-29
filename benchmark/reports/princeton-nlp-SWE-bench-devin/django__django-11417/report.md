# django__django-11417

| **django/django** | `3dca8738cbbbb5674f795169e5ea25e2002f2d71` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 588 |
| **Avg pos** | 2.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/django/core/mail/message.py b/django/core/mail/message.py
--- a/django/core/mail/message.py
+++ b/django/core/mail/message.py
@@ -2,15 +2,15 @@
 from email import (
     charset as Charset, encoders as Encoders, generator, message_from_string,
 )
-from email.errors import InvalidHeaderDefect, NonASCIILocalPartDefect
+from email.errors import HeaderParseError
 from email.header import Header
-from email.headerregistry import Address
+from email.headerregistry import Address, parser
 from email.message import Message
 from email.mime.base import MIMEBase
 from email.mime.message import MIMEMessage
 from email.mime.multipart import MIMEMultipart
 from email.mime.text import MIMEText
-from email.utils import formatdate, getaddresses, make_msgid, parseaddr
+from email.utils import formatdate, getaddresses, make_msgid
 from io import BytesIO, StringIO
 from pathlib import Path
 
@@ -71,56 +71,44 @@ def forbid_multi_line_headers(name, val, encoding):
     return name, val
 
 
-def split_addr(addr, encoding):
-    """
-    Split the address into local part and domain and encode them.
-
-    When non-ascii characters are present in the local part, it must be
-    MIME-word encoded. The domain name must be idna-encoded if it contains
-    non-ascii characters.
-    """
-    if '@' in addr:
-        localpart, domain = addr.split('@', 1)
-        # Try to get the simplest encoding - ascii if possible so that
-        # to@example.com doesn't become =?utf-8?q?to?=@example.com. This
-        # makes unit testing a bit easier and more readable.
-        try:
-            localpart.encode('ascii')
-        except UnicodeEncodeError:
-            localpart = Header(localpart, encoding).encode()
-        domain = domain.encode('idna').decode('ascii')
-    else:
-        localpart = Header(addr, encoding).encode()
-        domain = ''
-    return (localpart, domain)
-
-
 def sanitize_address(addr, encoding):
     """
     Format a pair of (name, address) or an email address string.
     """
+    address = None
     if not isinstance(addr, tuple):
-        addr = parseaddr(addr)
-    nm, addr = addr
-    localpart, domain = None, None
+        addr = force_str(addr)
+        try:
+            token, rest = parser.get_mailbox(addr)
+        except (HeaderParseError, ValueError, IndexError):
+            raise ValueError('Invalid address "%s"' % addr)
+        else:
+            if rest:
+                # The entire email address must be parsed.
+                raise ValueError(
+                    'Invalid adddress; only %s could be parsed from "%s"'
+                    % (token, addr)
+                )
+            nm = token.display_name or ''
+            localpart = token.local_part
+            domain = token.domain or ''
+    else:
+        nm, address = addr
+        localpart, domain = address.rsplit('@', 1)
+
     nm = Header(nm, encoding).encode()
+    # Avoid UTF-8 encode, if it's possible.
     try:
-        addr.encode('ascii')
-    except UnicodeEncodeError:  # IDN or non-ascii in the local part
-        localpart, domain = split_addr(addr, encoding)
-
-    # An `email.headerregistry.Address` object is used since
-    # email.utils.formataddr() naively encodes the name as ascii (see #25986).
-    if localpart and domain:
-        address = Address(nm, username=localpart, domain=domain)
-        return str(address)
-
+        localpart.encode('ascii')
+    except UnicodeEncodeError:
+        localpart = Header(localpart, encoding).encode()
     try:
-        address = Address(nm, addr_spec=addr)
-    except (InvalidHeaderDefect, NonASCIILocalPartDefect):
-        localpart, domain = split_addr(addr, encoding)
-        address = Address(nm, username=localpart, domain=domain)
-    return str(address)
+        domain.encode('ascii')
+    except UnicodeEncodeError:
+        domain = domain.encode('idna').decode('ascii')
+
+    parsed_address = Address(nm, username=localpart, domain=domain)
+    return str(parsed_address)
 
 
 class MIMEMixin:

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| django/core/mail/message.py | 5 | 13 | 2 | 1 | 588
| django/core/mail/message.py | 74 | 123 | - | 1 | -


## Problem Statement

```
Update mail backend to use modern standard library parsing approach.
Description
	
 django.core.mail.message.sanitize_address uses email.utils.parseaddr from the standard lib. On Python 3, email.headerregistry.parser.get_mailbox() does the same, and is less error-prone.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 django/core/mail/message.py** | 98 | 123| 238 | 238 | 3725 | 
| **-> 2 <-** | **1 django/core/mail/message.py** | 1 | 52| 350 | 588 | 3725 | 
| 3 | **1 django/core/mail/message.py** | 251 | 276| 322 | 910 | 3725 | 
| 4 | **1 django/core/mail/message.py** | 74 | 95| 201 | 1111 | 3725 | 
| 5 | 2 django/core/mail/utils.py | 1 | 21| 113 | 1224 | 3838 | 
| 6 | 3 django/core/mail/__init__.py | 1 | 35| 320 | 1544 | 4852 | 
| 7 | 4 django/core/mail/backends/__init__.py | 1 | 2| 0 | 1544 | 4860 | 
| 8 | **4 django/core/mail/message.py** | 184 | 192| 115 | 1659 | 4860 | 
| 9 | 5 django/core/mail/backends/smtp.py | 1 | 39| 364 | 2023 | 5888 | 
| 10 | **5 django/core/mail/message.py** | 341 | 356| 127 | 2150 | 5888 | 
| 11 | 5 django/core/mail/backends/smtp.py | 116 | 131| 137 | 2287 | 5888 | 
| 12 | 6 django/core/mail/backends/dummy.py | 1 | 11| 0 | 2287 | 5931 | 
| 13 | 7 django/core/mail/backends/console.py | 1 | 43| 281 | 2568 | 6213 | 
| 14 | 8 django/core/mail/backends/locmem.py | 1 | 31| 183 | 2751 | 6397 | 
| 15 | 9 django/core/validators.py | 233 | 277| 330 | 3081 | 10717 | 
| 16 | 10 django/core/mail/backends/filebased.py | 1 | 68| 547 | 3628 | 11265 | 
| 17 | 10 django/core/validators.py | 210 | 230| 139 | 3767 | 11265 | 
| 18 | 10 django/core/validators.py | 189 | 208| 157 | 3924 | 11265 | 
| 19 | 10 django/core/mail/__init__.py | 89 | 101| 130 | 4054 | 11265 | 
| 20 | **10 django/core/mail/message.py** | 162 | 181| 218 | 4272 | 11265 | 
| 21 | **10 django/core/mail/message.py** | 388 | 412| 185 | 4457 | 11265 | 
| 22 | **10 django/core/mail/message.py** | 126 | 138| 125 | 4582 | 11265 | 
| 23 | **10 django/core/mail/message.py** | 140 | 159| 189 | 4771 | 11265 | 
| 24 | **10 django/core/mail/message.py** | 55 | 71| 171 | 4942 | 11265 | 
| 25 | 11 django/core/mail/backends/base.py | 1 | 60| 323 | 5265 | 11588 | 
| 26 | 12 django/db/models/fields/__init__.py | 1637 | 1658| 183 | 5448 | 28582 | 
| 27 | 12 django/core/validators.py | 152 | 187| 388 | 5836 | 28582 | 
| 28 | 13 django/utils/log.py | 79 | 134| 439 | 6275 | 30190 | 
| 29 | 14 django/utils/http.py | 1 | 72| 693 | 6968 | 34146 | 
| 30 | 15 django/contrib/auth/forms.py | 240 | 262| 196 | 7164 | 37204 | 
| 31 | 16 django/utils/html.py | 287 | 300| 154 | 7318 | 40307 | 
| 32 | 17 django/contrib/admindocs/utils.py | 1 | 23| 133 | 7451 | 42207 | 
| 33 | 18 django/core/management/commands/sendtestemail.py | 1 | 24| 186 | 7637 | 42509 | 
| 34 | **18 django/core/mail/message.py** | 195 | 249| 398 | 8035 | 42509 | 
| 35 | 19 django/contrib/auth/migrations/0003_alter_user_email_max_length.py | 1 | 17| 0 | 8035 | 42587 | 
| 36 | **19 django/core/mail/message.py** | 358 | 386| 273 | 8308 | 42587 | 
| 37 | 19 django/core/mail/__init__.py | 38 | 60| 205 | 8513 | 42587 | 
| 38 | 19 django/core/mail/backends/smtp.py | 75 | 114| 263 | 8776 | 42587 | 
| 39 | 20 django/contrib/auth/base_user.py | 1 | 44| 298 | 9074 | 43471 | 
| 40 | **20 django/core/mail/message.py** | 325 | 339| 127 | 9201 | 43471 | 
| 41 | **20 django/core/mail/message.py** | 278 | 291| 128 | 9329 | 43471 | 
| 42 | 20 django/core/mail/__init__.py | 104 | 117| 133 | 9462 | 43471 | 
| 43 | 21 django/http/response.py | 1 | 25| 144 | 9606 | 47755 | 
| 44 | 22 django/contrib/auth/middleware.py | 85 | 110| 192 | 9798 | 48771 | 
| 45 | **22 django/core/mail/message.py** | 415 | 455| 314 | 10112 | 48771 | 
| 46 | 22 django/core/mail/backends/smtp.py | 41 | 73| 281 | 10393 | 48771 | 
| 47 | 23 django/conf/global_settings.py | 145 | 263| 876 | 11269 | 54375 | 
| 48 | 24 django/contrib/messages/utils.py | 1 | 13| 0 | 11269 | 54425 | 
| 49 | 24 django/utils/html.py | 255 | 285| 322 | 11591 | 54425 | 
| 50 | 24 django/utils/log.py | 1 | 76| 492 | 12083 | 54425 | 
| 51 | 25 django/http/request.py | 359 | 382| 185 | 12268 | 59145 | 
| 52 | 26 django/contrib/admindocs/views.py | 402 | 415| 127 | 12395 | 62455 | 
| 53 | 26 django/core/validators.py | 1 | 25| 190 | 12585 | 62455 | 
| 54 | 27 django/utils/formats.py | 235 | 258| 202 | 12787 | 64547 | 
| 55 | 27 django/utils/http.py | 193 | 215| 166 | 12953 | 64547 | 
| 56 | 27 django/core/management/commands/sendtestemail.py | 26 | 41| 121 | 13074 | 64547 | 
| 57 | 27 django/utils/html.py | 302 | 345| 443 | 13517 | 64547 | 
| 58 | 27 django/utils/http.py | 157 | 190| 319 | 13836 | 64547 | 
| 59 | 28 django/contrib/auth/backends.py | 150 | 211| 444 | 14280 | 66122 | 
| 60 | 29 django/contrib/auth/password_validation.py | 1 | 32| 206 | 14486 | 67606 | 
| 61 | 30 django/contrib/auth/models.py | 290 | 355| 463 | 14949 | 70609 | 
| 62 | 31 django/contrib/messages/storage/__init__.py | 1 | 13| 0 | 14949 | 70679 | 
| 63 | 32 django/contrib/postgres/utils.py | 1 | 30| 218 | 15167 | 70897 | 
| 64 | 32 django/contrib/admindocs/utils.py | 26 | 38| 144 | 15311 | 70897 | 
| 65 | 33 django/http/multipartparser.py | 645 | 690| 399 | 15710 | 75924 | 
| 66 | 33 django/contrib/auth/models.py | 1 | 30| 200 | 15910 | 75924 | 
| 67 | 34 django/contrib/auth/migrations/0007_alter_validators_add_error_messages.py | 1 | 25| 137 | 16047 | 76061 | 
| 68 | 35 django/contrib/messages/storage/cookie.py | 144 | 167| 190 | 16237 | 77393 | 
| 69 | 35 django/core/validators.py | 280 | 292| 116 | 16353 | 77393 | 
| 70 | 36 django/contrib/auth/validators.py | 1 | 26| 165 | 16518 | 77559 | 
| 71 | 37 django/core/management/commands/makemessages.py | 155 | 167| 111 | 16629 | 83119 | 
| 72 | 38 django/contrib/messages/middleware.py | 1 | 27| 174 | 16803 | 83294 | 
| 73 | 39 django/contrib/messages/__init__.py | 1 | 5| 0 | 16803 | 83330 | 
| 74 | 40 django/conf/locale/bg/formats.py | 5 | 22| 131 | 16934 | 83505 | 
| 75 | 40 django/http/request.py | 1 | 35| 245 | 17179 | 83505 | 
| 76 | 41 django/conf/locale/hr/formats.py | 5 | 21| 218 | 17397 | 84388 | 
| 77 | 41 django/core/mail/__init__.py | 63 | 86| 226 | 17623 | 84388 | 
| 78 | 42 django/contrib/auth/__init__.py | 1 | 58| 393 | 18016 | 85955 | 
| 79 | 43 django/core/servers/basehttp.py | 97 | 116| 178 | 18194 | 87667 | 
| 80 | 44 django/contrib/auth/admin.py | 1 | 22| 188 | 18382 | 89393 | 
| 81 | 45 django/conf/locale/sr/formats.py | 5 | 22| 293 | 18675 | 90242 | 
| 82 | 46 django/conf/locale/uk/formats.py | 5 | 38| 460 | 19135 | 90747 | 
| 83 | 46 django/contrib/auth/forms.py | 1 | 20| 137 | 19272 | 90747 | 
| 84 | 47 django/template/defaultfilters.py | 411 | 420| 111 | 19383 | 96812 | 
| 85 | 48 django/db/backends/utils.py | 134 | 150| 141 | 19524 | 98709 | 
| 86 | 49 django/db/backends/base/operations.py | 482 | 523| 284 | 19808 | 104102 | 
| 87 | 50 django/contrib/auth/migrations/0004_alter_user_username_opts.py | 1 | 24| 150 | 19958 | 104252 | 
| 88 | 51 django/conf/locale/ml/formats.py | 5 | 41| 663 | 20621 | 104960 | 
| 89 | 52 django/core/checks/messages.py | 26 | 50| 259 | 20880 | 105533 | 
| 90 | 53 django/utils/ipv6.py | 1 | 47| 295 | 21175 | 105829 | 
| 91 | 53 django/core/management/commands/makemessages.py | 283 | 362| 816 | 21991 | 105829 | 
| 92 | 54 django/conf/locale/sq/formats.py | 5 | 22| 128 | 22119 | 106001 | 
| 93 | 54 django/core/management/commands/makemessages.py | 363 | 392| 231 | 22350 | 106001 | 
| 94 | 55 django/http/__init__.py | 1 | 22| 197 | 22547 | 106198 | 
| 95 | 56 django/contrib/auth/hashers.py | 564 | 597| 242 | 22789 | 110985 | 
| 96 | 57 django/conf/locale/is/formats.py | 5 | 22| 130 | 22919 | 111160 | 
| 97 | 58 django/conf/locale/sr_Latn/formats.py | 5 | 22| 293 | 23212 | 112009 | 
| 98 | **58 django/core/mail/message.py** | 293 | 323| 271 | 23483 | 112009 | 
| 99 | 59 django/conf/locale/cs/formats.py | 5 | 43| 640 | 24123 | 112694 | 
| 100 | 60 django/conf/locale/sk/formats.py | 5 | 30| 348 | 24471 | 113087 | 
| 101 | 61 django/conf/locale/sl/formats.py | 5 | 20| 208 | 24679 | 113936 | 
| 102 | 61 django/contrib/messages/storage/cookie.py | 1 | 24| 184 | 24863 | 113936 | 
| 103 | 62 django/conf/locale/id/formats.py | 5 | 50| 708 | 25571 | 114689 | 
| 104 | 62 django/contrib/auth/base_user.py | 47 | 140| 585 | 26156 | 114689 | 
| 105 | 63 django/utils/encoding.py | 102 | 115| 130 | 26286 | 116985 | 
| 106 | 64 django/contrib/messages/views.py | 1 | 19| 0 | 26286 | 117081 | 
| 107 | 65 django/conf/locale/bs/formats.py | 5 | 22| 139 | 26425 | 117264 | 
| 108 | 66 django/contrib/auth/views.py | 187 | 203| 122 | 26547 | 119916 | 
| 109 | 67 django/forms/fields.py | 1158 | 1170| 113 | 26660 | 128860 | 
| 110 | 68 django/core/files/storage.py | 1 | 22| 158 | 26818 | 131696 | 
| 111 | 68 django/core/servers/basehttp.py | 119 | 154| 280 | 27098 | 131696 | 
| 112 | 69 django/conf/locale/ru/formats.py | 5 | 33| 402 | 27500 | 132143 | 
| 113 | 70 django/conf/locale/ka/formats.py | 5 | 22| 298 | 27798 | 133050 | 
| 114 | 71 django/conf/locale/de_CH/formats.py | 5 | 35| 416 | 28214 | 133511 | 
| 115 | 72 django/contrib/messages/constants.py | 1 | 22| 0 | 28214 | 133607 | 
| 116 | 73 django/conf/locale/hi/formats.py | 5 | 22| 125 | 28339 | 133776 | 
| 117 | 73 django/http/response.py | 134 | 155| 176 | 28515 | 133776 | 
| 118 | 74 django/conf/locale/bn/formats.py | 5 | 33| 294 | 28809 | 134114 | 
| 119 | 74 django/utils/formats.py | 1 | 57| 377 | 29186 | 134114 | 
| 120 | 74 django/core/validators.py | 73 | 108| 554 | 29740 | 134114 | 
| 121 | 75 django/conf/locale/mn/formats.py | 5 | 22| 120 | 29860 | 134278 | 
| 122 | 75 django/contrib/auth/middleware.py | 1 | 24| 193 | 30053 | 134278 | 
| 123 | 76 django/conf/locale/el/formats.py | 5 | 36| 508 | 30561 | 134831 | 
| 124 | 77 django/conf/locale/az/formats.py | 5 | 33| 399 | 30960 | 135275 | 
| 125 | 78 django/conf/locale/en_GB/formats.py | 5 | 40| 708 | 31668 | 136028 | 
| 126 | 79 django/http/cookie.py | 1 | 27| 188 | 31856 | 136217 | 
| 127 | 80 django/middleware/common.py | 118 | 147| 277 | 32133 | 137728 | 
| 128 | 81 django/conf/locale/da/formats.py | 5 | 27| 250 | 32383 | 138023 | 
| 129 | 81 django/contrib/auth/middleware.py | 47 | 83| 360 | 32743 | 138023 | 
| 130 | 82 django/conf/locale/mk/formats.py | 5 | 43| 672 | 33415 | 138740 | 
| 131 | 83 django/core/cache/backends/memcached.py | 1 | 36| 248 | 33663 | 140458 | 
| 132 | 83 django/utils/log.py | 162 | 196| 290 | 33953 | 140458 | 
| 133 | 84 django/utils/deprecation.py | 76 | 98| 158 | 34111 | 141150 | 
| 134 | 84 django/db/models/fields/__init__.py | 1888 | 1965| 567 | 34678 | 141150 | 
| 135 | 85 django/core/management/commands/compilemessages.py | 58 | 115| 504 | 35182 | 142416 | 
| 136 | 86 django/contrib/admin/options.py | 1045 | 1068| 234 | 35416 | 160769 | 
| 137 | 86 django/core/management/commands/makemessages.py | 633 | 663| 286 | 35702 | 160769 | 
| 138 | 86 django/core/validators.py | 110 | 149| 403 | 36105 | 160769 | 
| 139 | 87 django/core/signing.py | 145 | 170| 255 | 36360 | 162478 | 
| 140 | 88 django/conf/locale/nb/formats.py | 5 | 40| 646 | 37006 | 163169 | 
| 141 | 89 django/conf/locale/en_AU/formats.py | 5 | 40| 708 | 37714 | 163922 | 
| 142 | 90 django/conf/locale/fy/formats.py | 22 | 22| 0 | 37714 | 164074 | 
| 143 | 90 django/middleware/common.py | 1 | 32| 247 | 37961 | 164074 | 
| 144 | 91 django/conf/locale/de/formats.py | 5 | 29| 323 | 38284 | 164442 | 
| 145 | 92 django/conf/locale/et/formats.py | 5 | 22| 133 | 38417 | 164619 | 
| 146 | 93 django/conf/locale/pl/formats.py | 5 | 30| 339 | 38756 | 165003 | 
| 147 | 94 django/utils/translation/__init__.py | 197 | 269| 440 | 39196 | 167329 | 
| 148 | 95 django/views/debug.py | 154 | 177| 176 | 39372 | 171547 | 
| 149 | 96 django/forms/forms.py | 380 | 400| 210 | 39582 | 175620 | 
| 150 | 97 django/conf/locale/kn/formats.py | 5 | 22| 123 | 39705 | 175787 | 
| 151 | 98 django/conf/locale/cy/formats.py | 5 | 36| 582 | 40287 | 176414 | 
| 152 | 99 django/forms/utils.py | 149 | 179| 229 | 40516 | 177651 | 
| 153 | 100 django/conf/locale/lv/formats.py | 5 | 47| 735 | 41251 | 178431 | 
| 154 | 101 django/conf/locale/sv/formats.py | 5 | 39| 534 | 41785 | 179010 | 
| 155 | 102 django/conf/locale/gd/formats.py | 5 | 22| 144 | 41929 | 179198 | 
| 156 | 102 django/http/multipartparser.py | 1 | 40| 195 | 42124 | 179198 | 
| 157 | 103 django/conf/locale/en/formats.py | 5 | 41| 663 | 42787 | 179906 | 
| 158 | 104 django/contrib/messages/storage/base.py | 43 | 80| 274 | 43061 | 181104 | 
| 159 | 104 django/utils/http.py | 398 | 460| 318 | 43379 | 181104 | 
| 160 | 105 django/conf/locale/ro/formats.py | 5 | 36| 262 | 43641 | 181411 | 
| 161 | 106 django/conf/locale/te/formats.py | 5 | 22| 123 | 43764 | 181578 | 
| 162 | 107 django/conf/locale/km/formats.py | 5 | 22| 164 | 43928 | 181786 | 
| 163 | 108 django/conf/locale/ca/formats.py | 5 | 31| 287 | 44215 | 182118 | 
| 164 | 109 django/db/utils.py | 1 | 48| 150 | 44365 | 184136 | 
| 165 | 110 django/conf/locale/ta/formats.py | 5 | 22| 125 | 44490 | 184305 | 
| 166 | 111 django/conf/locale/hu/formats.py | 5 | 32| 323 | 44813 | 184673 | 
| 167 | 111 django/conf/locale/sr/formats.py | 23 | 44| 511 | 45324 | 184673 | 
| 168 | 112 django/forms/models.py | 350 | 380| 233 | 45557 | 196143 | 
| 169 | 113 django/utils/archive.py | 185 | 227| 286 | 45843 | 197646 | 
| 170 | 114 django/conf/locale/ga/formats.py | 5 | 22| 124 | 45967 | 197814 | 
| 171 | 114 django/core/checks/messages.py | 53 | 76| 161 | 46128 | 197814 | 
| 172 | 114 django/contrib/auth/hashers.py | 406 | 416| 127 | 46255 | 197814 | 


## Patch

```diff
diff --git a/django/core/mail/message.py b/django/core/mail/message.py
--- a/django/core/mail/message.py
+++ b/django/core/mail/message.py
@@ -2,15 +2,15 @@
 from email import (
     charset as Charset, encoders as Encoders, generator, message_from_string,
 )
-from email.errors import InvalidHeaderDefect, NonASCIILocalPartDefect
+from email.errors import HeaderParseError
 from email.header import Header
-from email.headerregistry import Address
+from email.headerregistry import Address, parser
 from email.message import Message
 from email.mime.base import MIMEBase
 from email.mime.message import MIMEMessage
 from email.mime.multipart import MIMEMultipart
 from email.mime.text import MIMEText
-from email.utils import formatdate, getaddresses, make_msgid, parseaddr
+from email.utils import formatdate, getaddresses, make_msgid
 from io import BytesIO, StringIO
 from pathlib import Path
 
@@ -71,56 +71,44 @@ def forbid_multi_line_headers(name, val, encoding):
     return name, val
 
 
-def split_addr(addr, encoding):
-    """
-    Split the address into local part and domain and encode them.
-
-    When non-ascii characters are present in the local part, it must be
-    MIME-word encoded. The domain name must be idna-encoded if it contains
-    non-ascii characters.
-    """
-    if '@' in addr:
-        localpart, domain = addr.split('@', 1)
-        # Try to get the simplest encoding - ascii if possible so that
-        # to@example.com doesn't become =?utf-8?q?to?=@example.com. This
-        # makes unit testing a bit easier and more readable.
-        try:
-            localpart.encode('ascii')
-        except UnicodeEncodeError:
-            localpart = Header(localpart, encoding).encode()
-        domain = domain.encode('idna').decode('ascii')
-    else:
-        localpart = Header(addr, encoding).encode()
-        domain = ''
-    return (localpart, domain)
-
-
 def sanitize_address(addr, encoding):
     """
     Format a pair of (name, address) or an email address string.
     """
+    address = None
     if not isinstance(addr, tuple):
-        addr = parseaddr(addr)
-    nm, addr = addr
-    localpart, domain = None, None
+        addr = force_str(addr)
+        try:
+            token, rest = parser.get_mailbox(addr)
+        except (HeaderParseError, ValueError, IndexError):
+            raise ValueError('Invalid address "%s"' % addr)
+        else:
+            if rest:
+                # The entire email address must be parsed.
+                raise ValueError(
+                    'Invalid adddress; only %s could be parsed from "%s"'
+                    % (token, addr)
+                )
+            nm = token.display_name or ''
+            localpart = token.local_part
+            domain = token.domain or ''
+    else:
+        nm, address = addr
+        localpart, domain = address.rsplit('@', 1)
+
     nm = Header(nm, encoding).encode()
+    # Avoid UTF-8 encode, if it's possible.
     try:
-        addr.encode('ascii')
-    except UnicodeEncodeError:  # IDN or non-ascii in the local part
-        localpart, domain = split_addr(addr, encoding)
-
-    # An `email.headerregistry.Address` object is used since
-    # email.utils.formataddr() naively encodes the name as ascii (see #25986).
-    if localpart and domain:
-        address = Address(nm, username=localpart, domain=domain)
-        return str(address)
-
+        localpart.encode('ascii')
+    except UnicodeEncodeError:
+        localpart = Header(localpart, encoding).encode()
     try:
-        address = Address(nm, addr_spec=addr)
-    except (InvalidHeaderDefect, NonASCIILocalPartDefect):
-        localpart, domain = split_addr(addr, encoding)
-        address = Address(nm, username=localpart, domain=domain)
-    return str(address)
+        domain.encode('ascii')
+    except UnicodeEncodeError:
+        domain = domain.encode('idna').decode('ascii')
+
+    parsed_address = Address(nm, username=localpart, domain=domain)
+    return str(parsed_address)
 
 
 class MIMEMixin:

```

## Test Patch

```diff
diff --git a/tests/mail/tests.py b/tests/mail/tests.py
--- a/tests/mail/tests.py
+++ b/tests/mail/tests.py
@@ -748,10 +748,30 @@ def test_sanitize_address(self):
                 'utf-8',
                 '=?utf-8?q?to=40other=2Ecom?= <to@example.com>',
             ),
+            (
+                ('To Example', 'to@other.com@example.com'),
+                'utf-8',
+                '=?utf-8?q?To_Example?= <"to@other.com"@example.com>',
+            ),
         ):
             with self.subTest(email_address=email_address, encoding=encoding):
                 self.assertEqual(sanitize_address(email_address, encoding), expected_result)
 
+    def test_sanitize_address_invalid(self):
+        for email_address in (
+            # Invalid address with two @ signs.
+            'to@other.com@example.com',
+            # Invalid address without the quotes.
+            'to@other.com <to@example.com>',
+            # Other invalid addresses.
+            '@',
+            'to@',
+            '@example.com',
+        ):
+            with self.subTest(email_address=email_address):
+                with self.assertRaises(ValueError):
+                    sanitize_address(email_address, encoding='utf-8')
+
 
 @requires_tz_support
 class MailTimeZoneTests(SimpleTestCase):

```


## Code snippets

### 1 - django/core/mail/message.py:

Start line: 98, End line: 123

```python
def sanitize_address(addr, encoding):
    """
    Format a pair of (name, address) or an email address string.
    """
    if not isinstance(addr, tuple):
        addr = parseaddr(addr)
    nm, addr = addr
    localpart, domain = None, None
    nm = Header(nm, encoding).encode()
    try:
        addr.encode('ascii')
    except UnicodeEncodeError:  # IDN or non-ascii in the local part
        localpart, domain = split_addr(addr, encoding)

    # An `email.headerregistry.Address` object is used since
    # email.utils.formataddr() naively encodes the name as ascii (see #25986).
    if localpart and domain:
        address = Address(nm, username=localpart, domain=domain)
        return str(address)

    try:
        address = Address(nm, addr_spec=addr)
    except (InvalidHeaderDefect, NonASCIILocalPartDefect):
        localpart, domain = split_addr(addr, encoding)
        address = Address(nm, username=localpart, domain=domain)
    return str(address)
```
### 2 - django/core/mail/message.py:

Start line: 1, End line: 52

```python
import mimetypes
from email import (
    charset as Charset, encoders as Encoders, generator, message_from_string,
)
from email.errors import InvalidHeaderDefect, NonASCIILocalPartDefect
from email.header import Header
from email.headerregistry import Address
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.message import MIMEMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, getaddresses, make_msgid, parseaddr
from io import BytesIO, StringIO
from pathlib import Path

from django.conf import settings
from django.core.mail.utils import DNS_NAME
from django.utils.encoding import force_str

# Don't BASE64-encode UTF-8 messages so that we avoid unwanted attention from
# some spam filters.
utf8_charset = Charset.Charset('utf-8')
utf8_charset.body_encoding = None  # Python defaults to BASE64
utf8_charset_qp = Charset.Charset('utf-8')
utf8_charset_qp.body_encoding = Charset.QP

# Default MIME type to use on attachments (if it is not explicitly given
# and cannot be guessed).
DEFAULT_ATTACHMENT_MIME_TYPE = 'application/octet-stream'

RFC5322_EMAIL_LINE_LENGTH_LIMIT = 998


class BadHeaderError(ValueError):
    pass


# Header names that contain structured address data (RFC #5322)
ADDRESS_HEADERS = {
    'from',
    'sender',
    'reply-to',
    'to',
    'cc',
    'bcc',
    'resent-from',
    'resent-sender',
    'resent-to',
    'resent-cc',
    'resent-bcc',
}
```
### 3 - django/core/mail/message.py:

Start line: 251, End line: 276

```python
class EmailMessage:

    def message(self):
        encoding = self.encoding or settings.DEFAULT_CHARSET
        msg = SafeMIMEText(self.body, self.content_subtype, encoding)
        msg = self._create_message(msg)
        msg['Subject'] = self.subject
        msg['From'] = self.extra_headers.get('From', self.from_email)
        self._set_list_header_if_not_empty(msg, 'To', self.to)
        self._set_list_header_if_not_empty(msg, 'Cc', self.cc)
        self._set_list_header_if_not_empty(msg, 'Reply-To', self.reply_to)

        # Email header names are case-insensitive (RFC 2045), so we have to
        # accommodate that when doing comparisons.
        header_names = [key.lower() for key in self.extra_headers]
        if 'date' not in header_names:
            # formatdate() uses stdlib methods to format the date, which use
            # the stdlib/OS concept of a timezone, however, Django sets the
            # TZ environment variable based on the TIME_ZONE setting which
            # will get picked up by formatdate().
            msg['Date'] = formatdate(localtime=settings.EMAIL_USE_LOCALTIME)
        if 'message-id' not in header_names:
            # Use cached DNS_NAME for performance
            msg['Message-ID'] = make_msgid(domain=DNS_NAME)
        for name, value in self.extra_headers.items():
            if name.lower() != 'from':  # From is already handled
                msg[name] = value
        return msg
```
### 4 - django/core/mail/message.py:

Start line: 74, End line: 95

```python
def split_addr(addr, encoding):
    """
    Split the address into local part and domain and encode them.

    When non-ascii characters are present in the local part, it must be
    MIME-word encoded. The domain name must be idna-encoded if it contains
    non-ascii characters.
    """
    if '@' in addr:
        localpart, domain = addr.split('@', 1)
        # Try to get the simplest encoding - ascii if possible so that
        # to@example.com doesn't become =?utf-8?q?to?=@example.com. This
        # makes unit testing a bit easier and more readable.
        try:
            localpart.encode('ascii')
        except UnicodeEncodeError:
            localpart = Header(localpart, encoding).encode()
        domain = domain.encode('idna').decode('ascii')
    else:
        localpart = Header(addr, encoding).encode()
        domain = ''
    return (localpart, domain)
```
### 5 - django/core/mail/utils.py:

Start line: 1, End line: 21

```python
"""
Email message and email sending related helper functions.
"""

import socket


# Cache the hostname, but do it lazily: socket.getfqdn() can take a couple of
# seconds, which slows down the restart of the server.
class CachedDnsName:
    def __str__(self):
        return self.get_fqdn()

    def get_fqdn(self):
        if not hasattr(self, '_fqdn'):
            self._fqdn = socket.getfqdn()
        return self._fqdn


DNS_NAME = CachedDnsName()
```
### 6 - django/core/mail/__init__.py:

Start line: 1, End line: 35

```python
"""
Tools for sending email.
"""
from django.conf import settings
# Imported for backwards compatibility and for the sake
# of a cleaner namespace. These symbols used to be in
# django/core/mail.py before the introduction of email
# backends and the subsequent reorganization (See #10355)
from django.core.mail.message import (
    DEFAULT_ATTACHMENT_MIME_TYPE, BadHeaderError, EmailMessage,
    EmailMultiAlternatives, SafeMIMEMultipart, SafeMIMEText,
    forbid_multi_line_headers, make_msgid,
)
from django.core.mail.utils import DNS_NAME, CachedDnsName
from django.utils.module_loading import import_string

__all__ = [
    'CachedDnsName', 'DNS_NAME', 'EmailMessage', 'EmailMultiAlternatives',
    'SafeMIMEText', 'SafeMIMEMultipart', 'DEFAULT_ATTACHMENT_MIME_TYPE',
    'make_msgid', 'BadHeaderError', 'forbid_multi_line_headers',
    'get_connection', 'send_mail', 'send_mass_mail', 'mail_admins',
    'mail_managers',
]


def get_connection(backend=None, fail_silently=False, **kwds):
    """Load an email backend and return an instance of it.

    If backend is None (default), use settings.EMAIL_BACKEND.

    Both fail_silently and other keyword arguments are used in the
    constructor of the backend.
    """
    klass = import_string(backend or settings.EMAIL_BACKEND)
    return klass(fail_silently=fail_silently, **kwds)
```
### 7 - django/core/mail/backends/__init__.py:

Start line: 1, End line: 2

```python

```
### 8 - django/core/mail/message.py:

Start line: 184, End line: 192

```python
class SafeMIMEMultipart(MIMEMixin, MIMEMultipart):

    def __init__(self, _subtype='mixed', boundary=None, _subparts=None, encoding=None, **_params):
        self.encoding = encoding
        MIMEMultipart.__init__(self, _subtype, boundary, _subparts, **_params)

    def __setitem__(self, name, val):
        name, val = forbid_multi_line_headers(name, val, self.encoding)
        MIMEMultipart.__setitem__(self, name, val)
```
### 9 - django/core/mail/backends/smtp.py:

Start line: 1, End line: 39

```python
"""SMTP email backend class."""
import smtplib
import ssl
import threading

from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend
from django.core.mail.message import sanitize_address
from django.core.mail.utils import DNS_NAME


class EmailBackend(BaseEmailBackend):
    """
    A wrapper that manages the SMTP network connection.
    """
    def __init__(self, host=None, port=None, username=None, password=None,
                 use_tls=None, fail_silently=False, use_ssl=None, timeout=None,
                 ssl_keyfile=None, ssl_certfile=None,
                 **kwargs):
        super().__init__(fail_silently=fail_silently)
        self.host = host or settings.EMAIL_HOST
        self.port = port or settings.EMAIL_PORT
        self.username = settings.EMAIL_HOST_USER if username is None else username
        self.password = settings.EMAIL_HOST_PASSWORD if password is None else password
        self.use_tls = settings.EMAIL_USE_TLS if use_tls is None else use_tls
        self.use_ssl = settings.EMAIL_USE_SSL if use_ssl is None else use_ssl
        self.timeout = settings.EMAIL_TIMEOUT if timeout is None else timeout
        self.ssl_keyfile = settings.EMAIL_SSL_KEYFILE if ssl_keyfile is None else ssl_keyfile
        self.ssl_certfile = settings.EMAIL_SSL_CERTFILE if ssl_certfile is None else ssl_certfile
        if self.use_ssl and self.use_tls:
            raise ValueError(
                "EMAIL_USE_TLS/EMAIL_USE_SSL are mutually exclusive, so only set "
                "one of those settings to True.")
        self.connection = None
        self._lock = threading.RLock()

    @property
    def connection_class(self):
        return smtplib.SMTP_SSL if self.use_ssl else smtplib.SMTP
```
### 10 - django/core/mail/message.py:

Start line: 341, End line: 356

```python
class EmailMessage:

    def _create_message(self, msg):
        return self._create_attachments(msg)

    def _create_attachments(self, msg):
        if self.attachments:
            encoding = self.encoding or settings.DEFAULT_CHARSET
            body_msg = msg
            msg = SafeMIMEMultipart(_subtype=self.mixed_subtype, encoding=encoding)
            if self.body or body_msg.is_multipart():
                msg.attach(body_msg)
            for attachment in self.attachments:
                if isinstance(attachment, MIMEBase):
                    msg.attach(attachment)
                else:
                    msg.attach(self._create_attachment(*attachment))
        return msg
```
### 20 - django/core/mail/message.py:

Start line: 162, End line: 181

```python
class SafeMIMEText(MIMEMixin, MIMEText):

    def __init__(self, _text, _subtype='plain', _charset=None):
        self.encoding = _charset
        MIMEText.__init__(self, _text, _subtype=_subtype, _charset=_charset)

    def __setitem__(self, name, val):
        name, val = forbid_multi_line_headers(name, val, self.encoding)
        MIMEText.__setitem__(self, name, val)

    def set_payload(self, payload, charset=None):
        if charset == 'utf-8' and not isinstance(charset, Charset.Charset):
            has_long_lines = any(
                len(l.encode()) > RFC5322_EMAIL_LINE_LENGTH_LIMIT
                for l in payload.splitlines()
            )
            # Quoted-Printable encoding has the side effect of shortening long
            # lines, if any (#22561).
            charset = utf8_charset_qp if has_long_lines else utf8_charset
        MIMEText.set_payload(self, payload, charset=charset)
```
### 21 - django/core/mail/message.py:

Start line: 388, End line: 412

```python
class EmailMessage:

    def _create_attachment(self, filename, content, mimetype=None):
        """
        Convert the filename, content, mimetype triple into a MIME attachment
        object.
        """
        attachment = self._create_mime_attachment(content, mimetype)
        if filename:
            try:
                filename.encode('ascii')
            except UnicodeEncodeError:
                filename = ('utf-8', '', filename)
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        return attachment

    def _set_list_header_if_not_empty(self, msg, header, values):
        """
        Set msg's header, either from self.extra_headers, if present, or from
        the values argument.
        """
        if values:
            try:
                value = self.extra_headers[header]
            except KeyError:
                value = ', '.join(str(v) for v in values)
            msg[header] = value
```
### 22 - django/core/mail/message.py:

Start line: 126, End line: 138

```python
class MIMEMixin:
    def as_string(self, unixfrom=False, linesep='\n'):
        """Return the entire formatted message as a string.
        Optional `unixfrom' when True, means include the Unix From_ envelope
        header.

        This overrides the default as_string() implementation to not mangle
        lines that begin with 'From '. See bug #13433 for details.
        """
        fp = StringIO()
        g = generator.Generator(fp, mangle_from_=False)
        g.flatten(self, unixfrom=unixfrom, linesep=linesep)
        return fp.getvalue()
```
### 23 - django/core/mail/message.py:

Start line: 140, End line: 159

```python
class MIMEMixin:

    def as_bytes(self, unixfrom=False, linesep='\n'):
        """Return the entire formatted message as bytes.
        Optional `unixfrom' when True, means include the Unix From_ envelope
        header.

        This overrides the default as_bytes() implementation to not mangle
        lines that begin with 'From '. See bug #13433 for details.
        """
        fp = BytesIO()
        g = generator.BytesGenerator(fp, mangle_from_=False)
        g.flatten(self, unixfrom=unixfrom, linesep=linesep)
        return fp.getvalue()


class SafeMIMEMessage(MIMEMixin, MIMEMessage):

    def __setitem__(self, name, val):
        # message/rfc822 attachments must be ASCII
        name, val = forbid_multi_line_headers(name, val, 'ascii')
        MIMEMessage.__setitem__(self, name, val)
```
### 24 - django/core/mail/message.py:

Start line: 55, End line: 71

```python
def forbid_multi_line_headers(name, val, encoding):
    """Forbid multi-line headers to prevent header injection."""
    encoding = encoding or settings.DEFAULT_CHARSET
    val = str(val)  # val may be lazy
    if '\n' in val or '\r' in val:
        raise BadHeaderError("Header values can't contain newlines (got %r for header %r)" % (val, name))
    try:
        val.encode('ascii')
    except UnicodeEncodeError:
        if name.lower() in ADDRESS_HEADERS:
            val = ', '.join(sanitize_address(addr, encoding) for addr in getaddresses((val,)))
        else:
            val = Header(val, encoding).encode()
    else:
        if name.lower() == 'subject':
            val = Header(val).encode()
    return name, val
```
### 34 - django/core/mail/message.py:

Start line: 195, End line: 249

```python
class EmailMessage:
    """A container for email information."""
    content_subtype = 'plain'
    mixed_subtype = 'mixed'
    encoding = None     # None => use settings default

    def __init__(self, subject='', body='', from_email=None, to=None, bcc=None,
                 connection=None, attachments=None, headers=None, cc=None,
                 reply_to=None):
        """
        Initialize a single email message (which can be sent to multiple
        recipients).
        """
        if to:
            if isinstance(to, str):
                raise TypeError('"to" argument must be a list or tuple')
            self.to = list(to)
        else:
            self.to = []
        if cc:
            if isinstance(cc, str):
                raise TypeError('"cc" argument must be a list or tuple')
            self.cc = list(cc)
        else:
            self.cc = []
        if bcc:
            if isinstance(bcc, str):
                raise TypeError('"bcc" argument must be a list or tuple')
            self.bcc = list(bcc)
        else:
            self.bcc = []
        if reply_to:
            if isinstance(reply_to, str):
                raise TypeError('"reply_to" argument must be a list or tuple')
            self.reply_to = list(reply_to)
        else:
            self.reply_to = []
        self.from_email = from_email or settings.DEFAULT_FROM_EMAIL
        self.subject = subject
        self.body = body or ''
        self.attachments = []
        if attachments:
            for attachment in attachments:
                if isinstance(attachment, MIMEBase):
                    self.attach(attachment)
                else:
                    self.attach(*attachment)
        self.extra_headers = headers or {}
        self.connection = connection

    def get_connection(self, fail_silently=False):
        from django.core.mail import get_connection
        if not self.connection:
            self.connection = get_connection(fail_silently=fail_silently)
        return self.connection
```
### 36 - django/core/mail/message.py:

Start line: 358, End line: 386

```python
class EmailMessage:

    def _create_mime_attachment(self, content, mimetype):
        """
        Convert the content, mimetype pair into a MIME attachment object.

        If the mimetype is message/rfc822, content may be an
        email.Message or EmailMessage object, as well as a str.
        """
        basetype, subtype = mimetype.split('/', 1)
        if basetype == 'text':
            encoding = self.encoding or settings.DEFAULT_CHARSET
            attachment = SafeMIMEText(content, subtype, encoding)
        elif basetype == 'message' and subtype == 'rfc822':
            # Bug #18967: per RFC2046 s5.2.1, message/rfc822 attachments
            # must not be base64 encoded.
            if isinstance(content, EmailMessage):
                # convert content into an email.Message first
                content = content.message()
            elif not isinstance(content, Message):
                # For compatibility with existing code, parse the message
                # into an email.Message object if it is not one already.
                content = message_from_string(force_str(content))

            attachment = SafeMIMEMessage(content, subtype)
        else:
            # Encode non-text attachments with base64.
            attachment = MIMEBase(basetype, subtype)
            attachment.set_payload(content)
            Encoders.encode_base64(attachment)
        return attachment
```
### 40 - django/core/mail/message.py:

Start line: 325, End line: 339

```python
class EmailMessage:

    def attach_file(self, path, mimetype=None):
        """
        Attach a file from the filesystem.

        Set the mimetype to DEFAULT_ATTACHMENT_MIME_TYPE if it isn't specified
        and cannot be guessed.

        For a text/* mimetype (guessed or specified), decode the file's content
        as UTF-8. If that fails, set the mimetype to
        DEFAULT_ATTACHMENT_MIME_TYPE and don't decode the content.
        """
        path = Path(path)
        with path.open('rb') as file:
            content = file.read()
            self.attach(path.name, content, mimetype)
```
### 41 - django/core/mail/message.py:

Start line: 278, End line: 291

```python
class EmailMessage:

    def recipients(self):
        """
        Return a list of all recipients of the email (includes direct
        addressees as well as Cc and Bcc entries).
        """
        return [email for email in (self.to + self.cc + self.bcc) if email]

    def send(self, fail_silently=False):
        """Send the email message."""
        if not self.recipients():
            # Don't bother creating the network connection if there's nobody to
            # send to.
            return 0
        return self.get_connection(fail_silently).send_messages([self])
```
### 45 - django/core/mail/message.py:

Start line: 415, End line: 455

```python
class EmailMultiAlternatives(EmailMessage):
    """
    A version of EmailMessage that makes it easy to send multipart/alternative
    messages. For example, including text and HTML versions of the text is
    made easier.
    """
    alternative_subtype = 'alternative'

    def __init__(self, subject='', body='', from_email=None, to=None, bcc=None,
                 connection=None, attachments=None, headers=None, alternatives=None,
                 cc=None, reply_to=None):
        """
        Initialize a single email message (which can be sent to multiple
        recipients).
        """
        super().__init__(
            subject, body, from_email, to, bcc, connection, attachments,
            headers, cc, reply_to,
        )
        self.alternatives = alternatives or []

    def attach_alternative(self, content, mimetype):
        """Attach an alternative content representation."""
        assert content is not None
        assert mimetype is not None
        self.alternatives.append((content, mimetype))

    def _create_message(self, msg):
        return self._create_attachments(self._create_alternatives(msg))

    def _create_alternatives(self, msg):
        encoding = self.encoding or settings.DEFAULT_CHARSET
        if self.alternatives:
            body_msg = msg
            msg = SafeMIMEMultipart(_subtype=self.alternative_subtype, encoding=encoding)
            if self.body:
                msg.attach(body_msg)
            for alternative in self.alternatives:
                msg.attach(self._create_mime_attachment(*alternative))
        return msg
```
### 98 - django/core/mail/message.py:

Start line: 293, End line: 323

```python
class EmailMessage:

    def attach(self, filename=None, content=None, mimetype=None):
        """
        Attach a file with the given filename and content. The filename can
        be omitted and the mimetype is guessed, if not provided.

        If the first parameter is a MIMEBase subclass, insert it directly
        into the resulting message attachments.

        For a text/* mimetype (guessed or specified), when a bytes object is
        specified as content, decode it as UTF-8. If that fails, set the
        mimetype to DEFAULT_ATTACHMENT_MIME_TYPE and don't decode the content.
        """
        if isinstance(filename, MIMEBase):
            assert content is None
            assert mimetype is None
            self.attachments.append(filename)
        else:
            assert content is not None
            mimetype = mimetype or mimetypes.guess_type(filename)[0] or DEFAULT_ATTACHMENT_MIME_TYPE
            basetype, subtype = mimetype.split('/', 1)

            if basetype == 'text':
                if isinstance(content, bytes):
                    try:
                        content = content.decode()
                    except UnicodeDecodeError:
                        # If mimetype suggests the file is text but it's
                        # actually binary, read() raises a UnicodeDecodeError.
                        mimetype = DEFAULT_ATTACHMENT_MIME_TYPE

            self.attachments.append((filename, content, mimetype))
```
