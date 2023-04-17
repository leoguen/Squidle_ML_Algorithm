import json
from urllib.parse import urljoin
from time import time
import requests
from io import StringIO
import pandas as pd

DEFAULT_HOST = "https://squidle.org"
JSON_SEPS = (',', ':')


def query_filter(name, op, val=None):
    f = dict(name=name, op=op)
    if val is not None:
        f['val'] = val
    return f


class SQAPIException(Exception):
    def __init__(self, msg, status_code=400, reason="ERROR", url=None):
        self.message = msg
        self.status_code = status_code
        self.reason = reason
        self.url = url

    def __str__(self):
        return "HTTPException ({}, {}): {}".format(self.reason, self.status_code, self.message)


class Request:
    def __init__(self, endpoint, method, id=None, qsparams=None, urlparams=None, files=None, data=None, headers=None, sqapi=None, template=None, json_data=None):
        self._endpoint = endpoint
        self.method = method
        self.files = files
        self.data = data
        self.json_data = json_data
        self.qsparams = qsparams or dict()
        self._urlparams = urlparams or dict()
        self._headers = headers or dict()
        self.id = id
        self._sqapi = sqapi
        if template is not None:
            self.template(template)

    def execute(self, build_url=True, with_api=None, verbosity=None, icon_ok="✅", icon_error="❌"):
        sqapi = with_api or self._sqapi
        if build_url:
            url = self.url(sqapi.host)
        else:
            url = urljoin(sqapi.host, self.endpoint)  # just assume that the endpoint is the full URL

        verbosity = sqapi.verbosity if verbosity is None else verbosity
        tic = time()
        r_string = f"HTTP-{self.method}: {url if verbosity > 1 else self.endpoint}"
        if verbosity > 0: print(f"{r_string} ...", end="")
        self._headers.update({"X-auth-token": self._sqapi.api_key})
        r = sqapi.session.request(self.method, url, headers=self.headers, files=self.files, data=self.data, json=self.json_data)
        if verbosity > 0: print(f" {icon_ok if r.ok else icon_error} | {r.status_code} | Done in {time() - tic:.2f} s")
        if not r.ok:
            try:
                # Try to parse error
                e = r.json()
                message = e.get("message")
                for k, v in e.get("validation_errors", {}).items():
                    message += " | {}: {}".format(k, v)
            except Exception as e:
                # otherwise print response string
                message = r_string
            print(f"{icon_error} | {message}")
            print(r.content)
            raise SQAPIException(f"{self.method} ERROR {message}", status_code=r.status_code, reason=r.reason, url=r.url)
        return r

    def url(self, host):
        if self.id is not None:
            self._urlparams['id'] = self.id

        url = urljoin(host, self.endpoint)

        params = [f"{k}={v}" for k, v in self.qsparams.items()]
        if len(params) > 0:
            url += f"{'&' if '?' in url else '?'}{'&'.join(params)}"

        return url

    @property
    def headers(self):
        return self._headers

    @property
    def endpoint(self):
        return self._endpoint.format(**self._urlparams)

    def template(self, template):
        self._headers.update({"X-template": template, "Accept": "text/html"})
        return self


class Get(Request):
    def __init__(self, endpoint, filters=None, order_by=None, page=None,
                 results_per_page=None, single=None, limit=None, offset=None, **kwarg):
        super().__init__(endpoint, method="GET", **kwarg)
        self._filters = filters or []
        self._order_by = order_by or []
        self._q = dict()
        self._page = page
        self._results_per_page = results_per_page
        self.single = single
        self.limit = limit
        self.offset = offset

    def filter(self, name, op, val=None):
        self._filters.append(query_filter(name, op, val=val))
        return self

    def filters_or(self, filters):
        self._filters.append({"or": filters})
        return self

    def filters_and(self, filters):
        self._filters.append({"and": filters})
        return self

    def order_by(self, field, direction="asc"):
        self._order_by.append(dict(field=field, direction=direction))
        return self
    
    def page(self, page):
        self._page = page
        return self

    def results_per_page(self, results_per_page):
        self._results_per_page = results_per_page
        return self

    def url(self, host):
        if self._filters: self._q["filters"] = self._filters
        if self._order_by: self._q["order_by"] = self._order_by
        if self.single is not None: self._q["single"] = self.single
        if self.limit is not None: self._q["limit"] = self.limit
        if self.offset is not None: self._q["offset"] = self.offset
        if self._q:
            self.qsparams["q"] = json.dumps(self._q, separators=JSON_SEPS)
        if self._page is not None:
            self.qsparams["page"] = self._page
        if self._results_per_page is not None:
            self.qsparams["results_per_page"] = self._results_per_page

        return super().url(host)


class Export(Get):
    def __init__(self, endpoint, include_columns=None, fileops=None, **kwargs):
        super().__init__(endpoint, **kwargs)
        self._file_operations = fileops or []
        self._f = dict()
        if isinstance(include_columns, list):
            self.qsparams["include_columns"] = json.dumps(include_columns or [], separators=JSON_SEPS)

    def file_op(self, method, module=None, **kwargs):
        op = dict(method=method)
        if module is not None:
            op['module'] = module
        if kwargs:
            op['kwargs'] = kwargs
        self._file_operations.append(op)
        return self

    def url(self, host):
        if self._file_operations:
            self._f["operations"] = self._file_operations
        if self._f:
            self.qsparams["f"] = json.dumps(self._f, separators=JSON_SEPS)

        return super().url(host)


class SaveFile(Export):
    def __init__(self, endpoint, fileops=None, save=None, **kwargs):
        super().__init__(endpoint, fileops=fileops, **kwargs)
        self._save = save or {}

    def save(self, collection, update_existing=False, match_on=None, create_missing=True):
        self._save = dict(
            collection=collection, update_existing=update_existing, match_on=match_on, create_missing=create_missing)
        return self

    def url(self, host):
        if self._save:
            self.qsparams['save'] = json.dumps(self._save, separators=JSON_SEPS)
        return super().url(host)


class Post(Request):
    def __init__(self, endpoint, data=None, json_data=None, headers=None, **kwargs):
        super().__init__(endpoint, "POST", data=data, json_data=json_data, headers=headers or {"Accept": "application/json"}, **kwargs)


class Patch(Request):
    def __init__(self, endpoint, data=None, json_data=None, **kwargs):
        super().__init__(endpoint, "PATCH", data=data, json_data=json_data, headers={"Accept": "application/json"}, **kwargs)


class SQAPI:
    def __init__(self, host=DEFAULT_HOST, api_key=None, verbosity=2, noauth=False):
        self.host = host or DEFAULT_HOST
        self.api_key = api_key
        self.verbosity = verbosity
        self.session = requests.Session()
        self.current_user = None
        # self.session.headers.update({"X-auth-token": self.api_key})

        # Get current user info and login session using API key
        if not noauth:
            self.connect(host, api_key)

    def login(self, host, username, password, *args, **keargs):
        self.host = host or self.host
        r = self.post("/api/users/login", data=json.dumps(dict(username=username, password=password))).execute()
        if r.ok:
            login_response = r.json()
            print(login_response)
            if login_response.get("api_token") is None:
                raise SQAPIException(f"Your API token does not appear to be set. You may need to login to the SQ+ "
                                     f"instance at {host} to activate it.")
            self.connect(host=host, api_key=login_response.get("api_token"))

    def connect(self, host=None, api_key=None, *args, **kwargs):
        self.host = host or self.host
        self.api_key = api_key or self.api_key
        self.current_user = self.get(
            "/api/users/login", headers={"X-auth-token": self.api_key}
        ).execute(verbosity=0).json()

    def status(self):
        return dict(
            host=self.host,
            current_user=self.current_user,
            verbosity=self.verbosity
        )

    def get(self, endpoint, *args, **kwargs):
        return Get(endpoint, sqapi=self, *args, **kwargs)

    def export(self, endpoint, include_columns=None, fileops=None, *args, **kwargs):
        return Export(endpoint, sqapi=self, page=None, results_per_page=None, include_columns=include_columns, *args, **kwargs)

    def post(self, endpoint, data=None, json_data=None, **kwargs):
        return Post(endpoint, sqapi=self, data=data, json_data=json_data, **kwargs)

    def patch(self, endpoint, data=None, json_data=None, **kwargs):
        return Patch(endpoint, sqapi=self, data=data, json_data=json_data, **kwargs)

    def upload_file(self, endpoint, file_path, data=None, **kwargs):
        files = {'file': (file_path, open(file_path, 'rb'), 'text/x-spam')}
        return Post(endpoint, sqapi=self, files=files, data=data, headers=None, **kwargs)

    def save_file(self, endpoint, id=None, include_columns=None, fileops=None, save=None, **kwargs):
        return SaveFile(endpoint, sqapi=self, id=id, include_columns=include_columns, fileops=fileops, save=save, **kwargs)

    def api_ref_link(self, resource, title='{resource} API reference'):
        return (f"<a href='{self.host}/api/help?template=api_help_page.html#{resource}' target='api_reference'>"
                f"{title.format(resource=resource)}"
                f"</a>")





