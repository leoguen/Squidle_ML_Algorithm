import argparse
from time import time
from urllib.parse import urljoin, unquote
import json
import requests
import os

from .ui import UIComponents

DEFAULT_URL = "http://localhost:5000"
DEFAULT_API_PATH = "api"



class HTTPException(Exception):
    def __init__(self, msg, status_code=400, reason="ERROR", url=None):
        self.message = msg
        self.status_code = status_code
        self.reason = reason
        self.url = url

    def __str__(self):
        return "HTTPException ({}, {}): {}".format(self.reason, self.status_code, self.message)


SQAPIargparser = argparse.ArgumentParser()
SQAPIargparser.add_argument("--api_token", type=str, help="The API key of the user to act on behalf of", required=True)
SQAPIargparser.add_argument("--url", type=str, help="The base URL of the server (default: {})".format(DEFAULT_URL), default=DEFAULT_URL)


class SQAPIBase:
    def __init__(self, api_token=None, url=DEFAULT_URL, api_path=DEFAULT_API_PATH, **kwargs):

        self.api_token = api_token  #or self.cliargs.api_token
        self.base_url = url  #or self.cliargs.url or DEFAULT_URL
        self.api_path = api_path  #or DEFAULT_API_PATH

        assert self.api_token is not None, "Unauthorised access. No `api_token` has been set."

        # Set current user based on API token
        params = dict(q=dict(filters=[dict(name="api_token", op="eq", val=self.api_token)], single=True))
        self.current_user = self.request("GET", resource="users", querystring_params=params)

    def select_resource_object_ui(self, resource, list_format="{name}", actions=None, *args, **kwargs):
        """

        :param resource:
        :param list_format:
        :param actions:
        :param do_get_single:
        :param args:
        :param kwargs:
        :return:
        """

        r = self.request(resource=resource, *args, **kwargs)
        objects = r.get("objects")
        return UIComponents.select_object_list(objects, title="Choose a {}:".format(resource.upper()), list_format=list_format, actions=actions)

    def get_create(self, data=None, querystring_params=None, match_on=None, *args, **kwargs):
        """

        :param data:
        :param querystring_params:
        :param match_on:
        :param args:
        :param kwargs:
        :return:  response obj, is_new (whether or not object was created or retrieved)
        """
        obj = self.get_if_exists(data=data, querystring_params=querystring_params, match_on=match_on, *args, **kwargs)
        if obj is not None:
            return obj, False
        else:
            return self.request("POST", data_json=data, *args, **kwargs), True

    def get_if_exists(self, data=None, querystring_params=None, match_on=None, *args, **kwargs):
        """

        :param data:
        :param querystring_params:
        :param match_on:
        :param args:
        :param kwargs:
        :return:
        """
        try:
            if querystring_params is None: querystring_params = {}
            if match_on is None: match_on = data.keys()
            querystring_params['q'] = dict(filters=[dict(name=k, op="eq", val=data.get(k)) for k in match_on], single=True)
            return self.request("GET", querystring_params=querystring_params, icon_error="✖", *args, **kwargs)
        except HTTPException as e:
            if e.status_code == requests.codes.NOT_FOUND:   # if not found, fail silently and return none, otherwise raise error
                return None
            raise e

    def confirm_delete(self, resource, id, metadata=None):
        """

        :param resource:
        :param id:
        :param metadata:
        :return:
        """
        assert id is not None, "ID is not specified"
        print("\n* DELETE {} **************************************\n".format(resource.upper()))
        if isinstance(metadata, dict):
            print(json.dumps(metadata, indent=2))
            # for k, v in metadata.items():
            #     if isinstance(v, (str, bool, float, int)):
            #         print(" - {}: {}".format(k,v))
        if input("\nAre you 100% sure that you want to delete this {0}?\n"
                 "ALL associated children / dependents will also be deleted. \n"
                 "This is NOT reversible! \n"
                 "* DELETE {0} **************************************\n"
                 "If so, type 'YES' => ".format(resource.upper())) == "YES":
            return self.request("DELETE", resource=resource + "/{id}", resource_params=dict(id=id))

    def get_create_file(self, resource, querystring_params=None, match_on=None, file_path=None, file_url=None, preprocess_file=None, fileparams=None, **data):
        """

        :param preprocess_file: callable function with arg `file_path` where
        :param resource:
        :param querystring_params:
        :param match_on:
        :param file_path:
        :param file_url:
        :param data:
        :return: response obj
        :return: is_new (bool: whether or not object was created or retrieved)
        """
        obj = self.get_if_exists(data=data, querystring_params=querystring_params, match_on=match_on, resource=resource)
        if obj is not None:
            return obj, False
        else:
            return self.create_file(resource, file_path=file_path, file_url=file_url, preprocess_file=preprocess_file, fileparams=fileparams, **data), True

    def create_file(self, resource, file_path=None, file_url=None, preprocess_file=None, fileparams=None, **data):
        """

        :param resource:
        :param file_path:
        :param file_url:
        :param preprocess_file:
        :param data:
        :return:
        """
        assert (file_path is not None or file_url is not None) and not (file_path is not None and file_url is not None), \
            "Either `file_path` or `file_url` must be set, but not both."
        if callable(preprocess_file):                                  # if process file is set, process the file and return a path to a local processed file
            file_path = preprocess_file(file_path=file_path, file_url=file_url, fileparams=fileparams)
        if file_path is not None:
            files = {'file': (os.path.basename(file_path), open(file_path, 'rb'), 'text/x-spam')}
            return self.request("POST", resource=resource + "/data", files=files, data=data)
        elif file_url is not None:
            data["file_url"] = file_url
            return self.request("POST", data_json=data, resource=resource)

    def send_user_email(self, subject, message, email_addresses=None, user_ids=None, usernames=None):
        users = []
        for e in email_addresses or []:
            users.append(self.get_if_exists(resource="users", data=dict(email=e)))
        for i in user_ids or []:
            users.append(self.get_if_exists(resource="users", data=dict(id=i)))
        for u in usernames or []:
            users.append(self.get_if_exists(resource="users", data=dict(username=u)))

        for u in users:
            if u is not None:
                self.request("POST", resource="users/{}/email".format(u.get("id")), icon_ok="📩",
                             data_json=dict(subject=subject, message=message))

    def request(self, method="GET", resource=None, data_json=None, data=None, querystring_params=None, files=None,
                icon_ok="✅", icon_error="❌", resource_params=None):
        """

        :param method:
        :param resource:
        :param data_json:
        :param data:
        :param querystring_params:
        :param files:
        :param icon_ok:
        :param icon_error:
        :param resource_params:
        :return:
        """
        tic = time()
        url = self._build_url(resource=resource, querystring_params=querystring_params, resource_params=resource_params)

        headers = {'auth-token': self.api_token, "Accept": "application/json"}
        r = requests.request(method, url, headers=headers, json=data_json, data=data, files=files)
        r_string = "HTTP {} ({}, {}, {:.3f}s): {}".format(method, r.reason, r.status_code, time()-tic, unquote(r.url))
        if r.ok:
            print(" {} {}".format(icon_ok, r_string))
            if r.status_code == requests.codes.NO_CONTENT:  # 204, no content (eg: DELETE)
                return r.content
            try: return r.json()            # first try get json response (most API requests)
            except: return r.content        # otherwise just return content (this should be rare, if ever)
        else:
            try:
                # Try to parse error
                e = r.json()
                message = e.get("message") or r_string
                for k,v in e.get("validation_errors", {}).items():
                    message += " | {}: {}".format(k,v)
                print(" {} {} ({})".format(icon_error, r_string, message))
            except Exception as e:
                # otherwise print response string
                message = r_string
                print(" {} {}".format(icon_error, r_string))
            raise HTTPException("HTTP {} ERROR {}".format(method, message), status_code=r.status_code, reason=r.reason, url=r.url)

    def _build_url(self, resource=None, querystring_params=None, resource_params=None):
        """

        :param resource:
        :param querystring_params:
        :param resource_params:
        :return:
        """
        url = urljoin(self.base_url, "/".join([self.api_path, resource]))
        url = url.format(**(resource_params or {}))
        if querystring_params is not None:
            qs = "&".join(["{}={}".format(k, json.dumps(v, separators=(',', ':')) if isinstance(v, dict) else v) for k,v in querystring_params.items()])
            url = "{}?{}".format(url, qs)
        return url
