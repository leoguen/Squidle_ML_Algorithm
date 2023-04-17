import os
import argparse
from abc import abstractmethod, ABC

import oyaml
from parse import parse
import fnmatch
import urllib.parse
import inspect

WILDCARD = "*"
PATHSEP = "/"
_DATASOURCE_REGISTRY = {}
CREDENTIALS_FILE = ".credentials.yaml"


argparser = argparse.ArgumentParser()
argparser.add_argument("--urlbase_browse", type=str, required=True,
                       help="Url base pattern to list repository objects, "
                            "eg: http://path/to/browse/{path}")
argparser.add_argument("--campaign_search", type=str, required=True,
                       help="Campaign path pattern. "
                            "eg: MYBUCKET/SA*")
argparser.add_argument("--deployment_search", type=str, required=True,
                       help="Deployment path pattern. "
                            "eg: {campaign[path]}SOMEAUV/sa2_*")
argparser.add_argument("--datafile_search", type=str, required=False, default=None,
                       help="[Optional] Data file path pattern to match. "
                            "eg: {deployment[path]}data/nav_*_data.csv")
argparser.add_argument("--mediadir_search", type=str, required=False, default=None,
                       help="[Optional] Media dir pattern to match. "
                            "eg: {deployment[path]}i*_frames")
argparser.add_argument("--thumbdir_search", type=str, required=False, default=None,
                       help="[Optional] Thumbnail dir pattern match. "
                            "eg: {deployment[path]thumbs}")
argparser.add_argument("--datafile_pattern", type=str, required=True,
                       help="URL path pattern to defile link to datafile for download, eg: "
                            "http://path.to.dfile/{campaign[basename]}/data/{deployment[basename]}/{search[basename]}")
argparser.add_argument("--media_pattern", type=str, required=True,
                       help="Media path pattern to match. "
                            "eg: {deployment[path]}/{search[basename]}/{{media.key}}")
argparser.add_argument("--thumb_pattern", type=str, required=False,
                       help="Thumbnail path pattern to match. "
                            "eg: {deployment[path]}/{search[basename]}/{{media.key}}")
# argparser.add_argument("--validate_searchs", required=False, action="store_false", default=True,
#                        help="(OPTIONAL) if set, patterns will be searched & matched even without wildcards. "
#                             "By default, if a pattern does not contain a wildcard, the path will be built naively.")


class SafeWildcardDict(dict):
    def __missing__(self, key):
        return WILDCARD


class SafeNoneDict(dict):
    def __missing__(self, key):
        return None


class NestedNoneDict(SafeNoneDict):
    def get(self, field, default=None):
        field_list = field.split(".")
        ret = self
        for f in field_list[:-1]:
            ret = dict.get(self, f, {})
        return dict.get(ret, field_list[-1], default)


def register_datasource_plugin(_class, name=None):
    """

    :param _class:
    :param name:
    :return:
    """
    global _DATASOURCE_REGISTRY
    assert inspect.isclass(_class), "Plugin needs a valid class"
    if name is None:
        name = _class.__name__
    assert name not in _DATASOURCE_REGISTRY, "Duplicate plugin name: '{}'. Plugin names need to be unique.".format(name)
    _DATASOURCE_REGISTRY[name] = _class
    print("Registered datasource plugin: {}".format(name))


def get_datasource_plugin(datasource_type=None):
    global _DATASOURCE_REGISTRY
    # ds = get_datasource(datasource.get("datasource_type", {}).get("name"))(cliargs=self.sqapi.cliargs, **datasource)
    name = datasource_type.get("name") if isinstance(datasource_type, dict) else datasource_type
    assert name in _DATASOURCE_REGISTRY, "'{}' is not a registered datasource type.".format(name)
    DatasourceClass = _DATASOURCE_REGISTRY.get(name)
    # DatasourceClass.add_arguments()
    return DatasourceClass


class DataSource(ABC):
    def __init__(self, urlbase_browse, campaign_search, deployment_search, datafile_pattern, media_pattern,
                 thumb_pattern=None, datafile_search=None, mediadir_search=None, thumbdir_search=None, user_id=None,
                 credential_key=None, pathsep=PATHSEP, sqapi=None, id=None, platform=None, name=None, datafile_operations=None, **kw):
                 # validate_searchs=False, **kw):
        """

        :param urlbase_browse:
        :param campaign_search:
        :param deployment_search:
        :param datafile_pattern:
        :param media_pattern:
        :param thumb_pattern:
        :param datafile_search:
        :param mediadir_search:
        :param thumbdir_search:
        :param credential_key:
        :param pathsep:
        :param cliargs:
        :param kw:
        """
        self.urlbase_browse = urlbase_browse
        self.credential_key = credential_key
        self.campaign_search = campaign_search
        self.deployment_search = deployment_search
        self.datafile_search = datafile_search
        self.mediadir_search = mediadir_search
        self.thumbdir_search = thumbdir_search
        # self.validate_searchs = validate_searchs
        self.pathsep = pathsep
        self.datafile_pattern = datafile_pattern
        self.media_pattern = media_pattern
        self.thumb_pattern = thumb_pattern
        self.sqapi = sqapi
        self.id = id
        self.name = name
        self.platform=platform or {}
        self.datafile_operations = datafile_operations or []
        self.user_id=user_id
        self._extra_kwargs = kw
        self.credentials = dict()

        if os.path.isfile(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE) as f:
                self.credentials = oyaml.safe_load(f).get(self.credential_key, {})


    # @classmethod
    # def add_arguments(cls):
    #     """Optionally override to add arguments specific to the parent class"""
    #     pass

    def list_campaigns(self):
        """

        :return:
        """
        search_url = self.get_url(self.campaign_search)
        # print("List campaigns: {}".format(search_url))
        return [self.get_campaign_info(c) for c in self.get_matched_objects(search_url)]

    def list_deployments(self, campaign=SafeWildcardDict()):
        """

        :param campaign:
        :return:
        """
        search_url = self.get_url(self.deployment_search, campaign=campaign)
        # print("List deployments: {}".format(search_url))
        return [self.get_deployment_info(d) for d in self.get_matched_objects(search_url)]

    def preprocess_deployment_assets(self, campaign=SafeWildcardDict(), deployment=SafeWildcardDict(), *args, **kwargs):
        deployment["fileparams"] = dict()  # optionally set the fileparams to be used by  `self.preprocess_file` method
        return campaign, deployment

    def postprocess_deployment_assets(self, dpl=None, *args, **kwargs):
        return dpl

    def preprocess_file(self, file_path=None, file_url=None, fileparams=None):
        return file_path  # return local file_path if there is a file_path

    def get_deployment_assets(self, campaign=SafeWildcardDict(), deployment=SafeWildcardDict()):
        """

        :param campaign:
        :param deployment:
        :return:
        """
        campaign, deployment = self.preprocess_deployment_assets(campaign=campaign, deployment=deployment)
        datafiles = self.get_deployment_asset_path(self.datafile_pattern, path_search=self.datafile_search, campaign=campaign, deployment=deployment)
        # mediadirs = self.get_deployment_asset_path(self.media_pattern, path_search=self.mediadir_search, campaign=campaign, deployment=deployment)
        # thumbdirs = self.get_deployment_asset_path(self.thumb_pattern, path_search=self.thumbdir_search, campaign=campaign, deployment=deployment)
        # datafiles = self.get_matched_objects_with_links(self.datafile_search, campaign=campaign, deployment=deployment)
        # mediadirs = self.get_matched_objects_with_links(self.mediadir_search, campaign=campaign, deployment=deployment)
        # thumbdirs = self.get_matched_objects_with_links(self.thumbdir_search, campaign=campaign, deployment=deployment)
        dpl = Deployment(datafiles=datafiles, campaign=campaign, deployment=deployment)
        dpl = self.postprocess_deployment_assets(dpl=dpl)
        return dpl

    def get_deployment_asset_path(self, path_pattern, path_search=None, campaign=SafeWildcardDict(), deployment=SafeWildcardDict()):
        """

        :param path_pattern:
        :param path_search:
        :param campaign:
        :param deployment:
        :return:
        """
        if path_pattern is None:
            return []
        if path_search:
            match_url = self.get_url(path_search, campaign=campaign, deployment=deployment)
            matches = self.get_matched_objects(match_url)
            for i in matches:
                i["path"] = path_pattern.format(campaign=campaign, deployment=deployment, search=i, self=self.__dict__)
        else:
            matches = [{"path": path_pattern.format(campaign=campaign, deployment=deployment, self=self.__dict__)}]
        return matches

    def get_matched_objects(self, search_url):
        """

        :param search_url: url to search for, will match patterns containing wildcards.
        :return: list of relative paths to matched objects
        """
        path, fname, subdir = self.decompose_wildcard_path(search_url)
        items = []
        url = self.get_url(path)
        # print(" * SEARCH: {} | PATH: {} | FNAME: {} | URL: {}".format(search_url, path, fname, url))
        print(" ✱ SEARCH: {}".format(search_url))
        for i in self.list_object_paths(url):
            object_name = i.get("basename")
            # print(f"path:{path}, oname: {oname}")
            if self.fnmatch(object_name, fname) or not fname:
                if subdir is None:
                    items.append(i)
                else:
                    subdir_path = i.get("path") + subdir[1:] if i.get("path")[-1] == self.pathsep and subdir[0] == self.pathsep else subdir
                    subdir_url = self.get_url(subdir_path)
                    items += self.get_matched_objects(subdir_url)  # recursively traverse into subdirectories

        return items

    # def get_matched_objects_with_links(self, path_search, campaign=SafeWildcardDict(), deployment=SafeWildcardDict(), **kw):
    #
    #     if path_search is None:  # if none, return empty list
    #         return []
    #     else:
    #         search_url = self.get_url(path_search, campaign=campaign, deployment=deployment, **kw)
    #         matched_object_paths = self.get_matched_objects(search_url)
    #     # elif self.validate_searchs or WILDCARD in path_search:  # if validate=True or path wildcard, do search
    #     #     search_url = self.get_url(path_search, campaign=campaign, deployment=deployment, **kw)
    #     #     matched_object_paths = self.get_matched_object_paths(search_url)
    #     # else:  # if not path pattern, just construct url naively
    #     #     matched_object_paths = [path_search]
    #
    #     matched_objects = []
    #     for p in matched_object_paths:
    #         url = self.get_url(p.get("path"), url_pattern=self.urlbase_download, campaign=campaign, deployment=deployment, **kw)
    #         # print(f" * REMOTE FILE URL: {url}")
    #         # p["path"] = self.get_object_path(url, self.urlbase_download)
    #         # p["basename"] = self.get_object_basename(p["path"])
    #         matched_objects.append(dict(url=url, **p))
    #     return matched_objects

    @abstractmethod
    def list_object_paths(self, url):
        """

        :param url:
        :return:
        """
        raise NotImplementedError("This method is not implemented in Base Class. "
                                  "Needs to be defined in derived class. Returns list of object paths.")

    def get_campaign_info(self, obj):
        """

        :param obj:
        :return:
        """
        return SafeNoneDict(name=obj.get("basename"), key=obj.get("basename"), **obj)

    def get_deployment_info(self, obj):
        """

        :param obj:
        :return:
        """
        return SafeNoneDict(name=obj.get("basename"), key=obj.get("basename"), **obj)

    def get_object_path(self, url, url_pattern=None):
        """

        :param url:
        :param url_pattern:
        :return:
        """
        if url_pattern is None:
            url_pattern = self.urlbase_browse
        try:
            r = parse(url_pattern, url)
            return r.named.get("path", url) if hasattr(r, "named") else url
        except Exception as e:
            print("Couldn't parse! {e}\npattern: {url_pattern}, path: {url}".format(e=e,url_pattern=url_pattern, url=url))
            return url

    def get_object_basename(self, path, ignore_trailing_pathsep=True):
        """

        :param path:
        :param ignore_trailing_pathsep:
        :return:
        """
        if path is None:
            return None
        # path = self.get_object_path(path)
        if ignore_trailing_pathsep and path.endswith(self.pathsep):
            path = path[:-1]  # remove trailing slash (if present)
        path_components = path.split(self.pathsep)
        return urllib.parse.unquote_plus(path_components[-1])

    def get_object_dirname(self, path):
        """

        :param path:
        :return:
        """
        if path is None:
            return None
        path_components = path.split(self.pathsep)
        dirname = self.pathsep.join(path_components[0:-1])
        return urllib.parse.unquote_plus(dirname)+self.pathsep if dirname else ""

    def get_url(self, path_pattern, url_pattern=None, campaign=SafeWildcardDict(), deployment=SafeWildcardDict(), **kw):
        """

        :param path_pattern:
        :param url_pattern:
        :param campaign:
        :param deployment:
        :param kw:
        :return:
        """
        if url_pattern is None:
            url_pattern = self.urlbase_browse
        path = path_pattern.format(campaign=campaign, deployment=deployment, **kw)
        path = path.replace(self.pathsep + self.pathsep, self.pathsep)  # replace any // with /
        url = url_pattern.format(path=path)
        # print(f"list_path: {path}")
        return url

    def parse_pattern(self, textstring, pattern):
        """

        :param textstring:
        :param pattern:
        :return:
        """
        # path_pattern = pattern.replace("{base}", self.filebase).replace("{filebase}", self.filebase)
        # print(f"PathPattern: {path_pattern}, Txt: '{textstring}', Pattern: '{pattern}'")
        result = parse(format=pattern.replace(WILDCARD, "{}").strip(), string=textstring.strip())
        # print(result)
        return result.named

    def filter_pattern(self, items, pattern):
        """

        :param items:
        :param pattern:
        :return:
        """
        return [i for i in items if self.fnmatch(i, pattern)]

    def fnmatch(self, item, pattern):
        """

        :param item:
        :param pattern:
        :return:
        """
        # print(f"Match: {item} to {pattern}")
        return fnmatch.fnmatch(item, pattern)

    def decompose_wildcard_path(self, search_url):
        """

        :param search_url:
        :return:
        """
        # example: path = "http://path/to/dir*/subdir/prefix*suffix.csv"
        # Get position of first occurrence of a wild card
        path_pattern = self.get_object_path(search_url)
        wildcard_index = path_pattern.find(WILDCARD)
        subdir = None
        if wildcard_index >= 0:
            part1 = path_pattern[0:wildcard_index]  # contains url+prefix, eg: http://path/to/dir
            part2 = path_pattern[wildcard_index:]  # contains all remaining, eg: */subdir/prefix*suffix.csv
            subdir_index = part2.find(self.pathsep)  # if subdir present, resplit things
            if subdir_index >= 0:
                fname = self.get_object_basename(part1, ignore_trailing_pathsep=False) + part2[0:subdir_index]  # eg: dir*
                subdir = part2[subdir_index:]  # eg: /subdir/prefix*suffix.csv
            else:
                fname = self.get_object_basename(part1, ignore_trailing_pathsep=False) + part2
            path = self.get_object_dirname(part1)

        # No wildcards, so just get url and file to match
        else:
            path = self.get_object_dirname(path_pattern)
            fname = self.get_object_basename(path_pattern, ignore_trailing_pathsep=False)

        if path and not path.endswith(self.pathsep):
            path += self.pathsep
        # print(f"path: {path}, fname: {fname}, subdir: {subdir}")
        return path, fname, subdir

    def process_all_datafiles(self, func=None):
        """

        :param func:
        :return:
        """
        # if no func is supplied, just print datafile and mediadir
        if func is None:
            func = print_deployment_info

        # loop through campaigns
        for c in self.list_campaigns():
            print("CAMPAIGN: {}".format(c))
            # loop through deployments
            for d in self.list_deployments(campaign=c):
                # loop through datafiles files
                dpl = self.get_deployment_assets(campaign=c, deployment=d)
                func(dpl)


class Deployment:
    def __init__(self, datafiles, campaign, deployment, mediadirs=None, thumbdirs=None):
        """

        :param datafiles:
        :param campaign:
        :param deployment:
        :param mediadirs:
        :param thumbdirs:
        """
        self.datafiles = datafiles
        self.campaign = campaign
        self.deployment = deployment
        self.mediadirs = mediadirs or []
        self.thumbdirs = thumbdirs or []

    def dict(self):
        """

        :return:
        """
        return self.__dict__

    def get_campaign(self, **add_fields):
        ret = NestedNoneDict(
            name=self.campaign.get("name"),
            key=self.campaign.get("key"),
            path=self.campaign.get("path"),
            **add_fields
        )
        return ret

    def get_deployment(self, campaign_id, platform_id, datasource_id, **add_fields):
        deployment = NestedNoneDict(
            name=self.deployment.get("name"),
            key=self.deployment.get("key"),
            path=self.deployment.get("path"),
            campaign_id=campaign_id,
            platform_id=platform_id,
            datasource_id=datasource_id,
            is_valid=False,
            **add_fields
        )
        return deployment

    def get_deployment_files(self, fileparams=None, **add_fields):
        """

        :param fileparams: DICT, special field passed to api.create_file that can be used for preprocessing files
        :param add_fields:
        :return:
        """
        files = []
        for f in self.datafiles:
            finfo = dict(
                name=os.path.basename(f.get("path")),
                updated_at=f.get("mtime"),
                fileparams = fileparams or {},
                **add_fields
            )
            if f.get("path").startswith("http"):
                finfo["file_url"] = f.get("path")
            else:
                finfo["file_path"] = f.get("path")
            files.append(finfo)
        return files
        # return [dict(name=os.path.basename(f.get("url")), file_url=f.get("url"), updated_at=f.get("mtime"), **add_fields) for f in self.datafiles]


def print_deployment_info(dpl):
    """

    :param dpl:
    :return:
    """
    print(" ╠ DEPLOYMENT: {}".format(dpl.deployment.get("key")))
    if len(dpl.datafiles) > 0:
        for d in dpl.datafiles:
            print(" ║ ├ DATAFILE URL: {}".format(d['url']))
    else:
        print(" ║ ├ DATAFILE ERROR!!!: No datafiles found.")
    if len(dpl.mediadirs) == 1:
        print(" ║ ├ MEDIADIR URL: {}".format(dpl.mediadirs[0]['url']))
    else:
        print(" ║ ├ MEDIADIR ERROR!!!: Expected exactly 1 mediadir. {} found.".format(len(dpl.mediadirs)))
    if len(dpl.thumbdirs) == 1:
        print(" ║ ├ THUMBDIR URL: {}".format(dpl.thumbdirs[0]['url']))
    elif len(dpl.thumbdirs) > 1:
        print(" ║ ├ THUMBDIR ERROR!!!: Expected no more than 1 thumbdir. {} found.".format(len(dpl.mediadirs)))