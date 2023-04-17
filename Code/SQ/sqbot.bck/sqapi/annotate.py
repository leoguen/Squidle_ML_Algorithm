import inspect
import json

import time

from sqapi.api import SQAPIBase, SQAPIargparser
from sqapi.media import SQMediaObject

DEFAULT_PROB_THRESH = 0.5
DEFAULT_POLL_DELAY = 600 #5
DEFAULT_ANNOTATOR_NAME = '{user[first_name]}-{user[last_name]}'

_ANNOTATOR_REGISTRY = {}


def register_annotator_plugin(_class, name=None):
    """

    :param _class:
    :param name:
    :return:
    """
    global _ANNOTATOR_REGISTRY
    assert inspect.isclass(_class), "Plugin needs a valid class"
    if name is None:
        name = _class.__name__
    assert name not in _ANNOTATOR_REGISTRY, "Duplicate plugin name: '{}'. Plugin names need to be unique.".format(name)
    _ANNOTATOR_REGISTRY[name] = _class
    print("Registered annotator plugin: {}".format(name))


def get_annotator_plugin(annotator=None):
    """

    :param annotator:
    :return:
    """
    global _ANNOTATOR_REGISTRY
    # ds = get_annotator(annotator.get("annotator_type", {}).get("name"))(cliargs=self.sqapi.cliargs, **annotator)
    name = annotator.get("name") if isinstance(annotator, dict) else annotator
    assert name in _ANNOTATOR_REGISTRY, "'{}' is not a registered annotator type. Valid annotators include: {}".format(
        name, get_annotator_keys())
    AnnotatorClass = _ANNOTATOR_REGISTRY.get(name)
    AnnotatorClass.add_arguments()
    return AnnotatorClass


def get_annotator_keys():
    """

    :return:
    """
    global _ANNOTATOR_REGISTRY
    return list(_ANNOTATOR_REGISTRY.keys())


class Annotator:
    def __init__(self, sqapi=None, annotator_name=None, prob_thresh=DEFAULT_PROB_THRESH, poll_delay=DEFAULT_POLL_DELAY,
                 label_map_file=None, annotation_set_id=None, affiliation_group_id=None, user_group_id=None, after_date=None, email_results=None, **sqapi_kwargs):
        self.sqapi = sqapi or SQAPIBase(**sqapi_kwargs)
        self.annotator_name = annotator_name.format(user=self.sqapi.current_user)  #(annotator_name or self.sqapi.cliargs.annotator_name).format(user=self.sqapi.current_user)
        self.prob_thresh = prob_thresh #or self.sqapi.cliargs.prob_thresh
        self.poll_delay = poll_delay #or self.sqapi.cliargs.poll_delay
        self.label_map_file = label_map_file #or self.sqapi.cliargs.label_map_file
        self.annotation_set_id = annotation_set_id #or self.sqapi.cliargs.annotation_set_id
        self.affiliation_group_id = affiliation_group_id #or self.sqapi.cliargs.affiliation_group_id
        self.user_group_id = user_group_id #or self.sqapi.cliargs.user_group_id
        self.after_date = after_date
        self.email_results = email_results
        self.labels = []
        self.label2id = None

    @classmethod
    def add_arguments(cls):
        """

        :return:
        """
        SQAPIargparser.add_argument("--annotation_set_id", type=int, help="If supplied will process that annotation_set.", required=False, default=None)
        SQAPIargparser.add_argument("--affiliation_group_id", type=int, help="If supplied will process all visible datasets from all members in that group.", required=False, default=None)
        SQAPIargparser.add_argument("--user_group_id", type=int, help="If supplied will process all visible datasets shared in that group.", required=False, default=None)
        SQAPIargparser.add_argument("--annotator_name", type=str, help="Name of Annotator/Algorithm (default: '{}')".format(DEFAULT_ANNOTATOR_NAME), default=DEFAULT_ANNOTATOR_NAME)
        SQAPIargparser.add_argument("--prob_thresh", type=float, help="Probability threshold for positive prediction (default: {})".format(DEFAULT_PROB_THRESH), default=DEFAULT_PROB_THRESH)
        SQAPIargparser.add_argument("--poll_delay", type=float, help="Number of seconds delay between checking for shared datasets (default: {}). To run only once, set to '-1'".format(DEFAULT_POLL_DELAY), default=DEFAULT_POLL_DELAY)
        SQAPIargparser.add_argument("--label_map_file", type=str, help="Path to label mapping JSON file", required=True)
        SQAPIargparser.add_argument("--after_date", type=str, default=None)
        SQAPIargparser.add_argument("-e", '--email_results', action='store_true', help="(optional) Whether or not to email a notification that the classifier is complete.")
        # SQAPIBase.add_argument("-e", '--email_results', action='append', default=[], type=str, help="(optional) GEODATA: email sync results to user(s) with matching email address(es) (all must be registered email on server).")

    def run(self, page=1, results_per_page=200, email_results=None):
        """

        :param email_results:
        :param page:
        :param results_per_page:
        :return:
        """
        annotation_sets = self.get_annotation_sets(
            annotation_set_id=self.annotation_set_id, affiliation_group_id=self.affiliation_group_id, after_date=self.after_date,
            user_group_id=self.user_group_id, page=page, results_per_page=results_per_page)

        print(f"\nFOUND: {annotation_sets.get('num_results')} annotation_sets | processing page {annotation_sets.get('page')}/{annotation_sets.get('total_pages')}...\n")

        # process current page of annotation_sets
        i = 0
        for a in annotation_sets.get("objects"):
            try:
                i += 1
                print(f"{'*'*80}\nProcessing ANNOTATION_SET: {(annotation_sets.get('page')-1)*results_per_page+i} / {annotation_sets.get('num_results')} | ID:{a.get('id')} > {a.get('name')} [{a.get('user',{}).get('username')}] ...\n{'*'*80}\n")
                media_count, point_count, annotation_count = self.process_annotation_set(a)
                if self.email_results if email_results is None else email_results:
                    self.email_annotation_set_user(a, media_count, point_count, annotation_count)
            except Exception as e:
                pass

        # paginate annotation_sets, if more than one page returned
        if annotation_sets.get("page") < annotation_sets.get("total_pages"):
            self.run(page=page+1, results_per_page=results_per_page, email_results=email_results)

        # if poll_delay set, keep alive and rerun every "poll_delay" seconds.
        if isinstance(self.poll_delay, (float, int)) and self.poll_delay > 0:
            time.sleep(self.poll_delay)
            self.run(page=page+1, results_per_page=results_per_page, email_results=email_results)

    def get_annotation_sets(self, annotation_set_id=None, affiliation_group_id=None, user_group_id=None, page=1,
                            results_per_page=200, filter_processed_annotation_sets=True, after_date=None):
        """

        :param annotation_set_id:
        :param affiliation_group_id:
        :param user_group_id:
        :param page:
        :param results_per_page:
        :param filter_processed_annotation_sets:
        :return:
        """
        or_filters = []
        if isinstance(annotation_set_id, int):
            or_filters.append(dict(name="id",op="eq",val=annotation_set_id))
        if isinstance(affiliation_group_id, int):
            or_filters.append(dict(name="user", op="has", val=dict(name="affiliations_usergroups",op="any",val=dict(name="group_id",op="eq",val=affiliation_group_id))))
        if isinstance(user_group_id, int):
            or_filters.append(dict(name="usergroups", op="any", val=dict(name="id", op="eq", val=user_group_id)))

        date_filter = [dict(name="created_at", op="gt", val=after_date)] if after_date else []

        q = dict(filters=[{"or": or_filters}]+date_filter) if or_filters else dict()
        if filter_processed_annotation_sets:
            user_id = self.sqapi.current_user.get("id")
            q['filters'].append({"not": dict(name="children", op="any", val=dict(name="user_id", op="eq", val=user_id))})
        return self.sqapi.request("GET", resource="annotation_set", querystring_params=dict(q=q, page=page, results_per_page=results_per_page))

    def process_annotation_set(self, annotation_set_data):
        """

        :param annotation_set_data:
        :return:
        """
        code2label = self.get_label_mappings(annotation_set_data)
        new_annotation_set = self.create_supplemental_annotation_set(parent_data=annotation_set_data)
        media_count, point_count, annotation_count = self.annotate_media(new_annotation_set, code2label=code2label)
        return media_count, point_count, annotation_count

    def get_label_mappings(self, annotatation_set_data):
        """

        :param annotatation_set_data:
        :return:
        """
        label_scheme_id = annotatation_set_data.get("label_scheme", {}).get("id")
        label_scheme_data = self.sqapi.request("GET", resource="label_scheme/{id}", resource_params={"id":label_scheme_id})
        parent_label_scheme_ids = label_scheme_data.get("parent_label_scheme_ids")
        with open(self.label_map_file) as f:
            label_map_filters = json.load(f)
        if isinstance(label_map_filters, dict):
            code2label = {}
            for l, filts in label_map_filters.items():
                code2label[l] = self.get_label(filts, parent_label_scheme_ids)
        elif isinstance(label_map_filters, list):
            code2label = []
            for filts in label_map_filters:
                code2label.append(self.get_label(filts, parent_label_scheme_ids))
        else:
            raise TypeError("Unknown `label_map_filters` type. Must be a `list` or a `dict`")

        return code2label

    def get_label(self, filts, label_scheme_ids):
        """

        :param filts:
        :param label_scheme_ids:
        :return:
        """
        if isinstance(filts, list):
            q = dict(filters=filts + [dict(name="label_scheme_id", op="in", val=label_scheme_ids)], single=True)
            return self.sqapi.request("GET", resource="label", querystring_params={"q": q})
        return None

    def create_supplemental_annotation_set(self, parent_data, child_name=None, child_description=None):
        """

        :param parent_data:
        :param child_name:
        :param child_description:
        :return:
        """
        child_data = dict()
        child_data["user_id"] = self.sqapi.current_user.get("id")
        child_data['media_collection_id'] = parent_data.get("media_collection",{}).get("id")
        child_data['label_scheme_id'] = parent_data.get('label_scheme',{}).get('id')
        child_data['parent_id'] = parent_data.get('id')
        child_data['description'] = child_description or "Suggested annotations by '{}' for the '{}' annotation_set." \
                              "This is not a standalone annotation set.".format(self.annotator_name, parent_data.get("name"))
        child_data['name'] = child_name or self.annotator_name

        return self.sqapi.request("POST", resource="annotation_set", data_json=child_data)

    def annotate_media(self, annotation_set_data, code2label=None, page=1, results_per_page=500):
        """

        :param annotation_set_data:
        :param code2label:
        :param page:
        :param results_per_page:
        :return:
        """
        annotation_set_id = annotation_set_data.get("id")
        media_collection_id = annotation_set_data.get("media_collection",{}).get("id")
        media_list = self.get_media_collection_media(media_collection_id, page=page, results_per_page=results_per_page)
        num_results = media_list.get('num_results')
        media_count = 0
        point_count = 0
        annotation_count = 0
        for m in media_list.get("objects"):
            media_count += 1
            print(f"\nProcessing: media item {media_count + (page-1)*results_per_page} / {num_results}")
            media_url = m.get('path_best')
            media_type = m.get("media_type", {}).get("name")
            mediaobj = SQMediaObject(media_url, media_type=media_type, media_id=m.get('id'))

            # get media annotations. If this frame has not been observed, it will generat the annotations through the request
            base_annotation_set_id = annotation_set_data.get("parent_id") or annotation_set_data.get("id")
            media_annotations = self.sqapi.request("GET", resource="media/{media_id}/annotations/{annotation_set_id}",
                resource_params=dict(media_id=m.get("id"), annotation_set_id=base_annotation_set_id))
            points = media_annotations.get('annotations')
            point_count += len(points)

            # run point predictions
            annotations = []
            for p in points:
                point_id = p.get("id")
                x = p.get('x')
                y = p.get('y')
                t = p.get('t')

                # decide whether a point label or a frame label
                if x is not None and y is not None:
                    code, probability = self.classify_point(mediaobj, x=x, y=y, t=t)
                else:
                    code, probability = self.classify_frame(mediaobj, t=t)

                # print(f"code: {code}, prob: {probability}")
                # time.sleep(1)

                # lookup label_id from classifier code. If dict, use get, otherwise assume list index code
                if code is not None:
                    label = code2label[code]   # if dict, use key, otherwise treat as index to list
                    label_id = label.get("id") if label is not None else None

                    # build up annotations list
                    if label_id is not None:
                        annotations.append(dict(
                            user_id=self.sqapi.current_user.get("id"),
                            label_id=label_id,
                            annotation_set_id=annotation_set_id,
                            point_id=point_id,
                            likelihood=float(probability)
                            # data=dict(probability=float(probability))
                        ))

            # Submit and save any new annotations
            for a in annotations:
                if a.get('likelihood', 0) >= self.prob_thresh:
                    self.sqapi.request("POST", resource="annotation", data_json=a)
                    annotation_count += 1

        # continue until all images are processed
        if media_list.get("page") < media_list.get("total_pages"):
            _mc, _pc, _ac = self.annotate_media(
                annotation_set_data, code2label=code2label, page=page+1, results_per_page=results_per_page)
            media_count += _mc
            point_count += _pc
            annotation_count += _ac

        return media_count, point_count, annotation_count

    def get_media_collection_media(self, media_collection_id, page=1, results_per_page=500):
        """

        :param media_collection_id:
        :param page:
        :param results_per_page:
        :return:
        """
        q = dict(
            filters=[dict(name="media_collections", op="any", val=dict(name="id", op="eq", val=media_collection_id))],
            order_by=[dict(field="timestamp_start", direction="asc")]
        )
        return self.sqapi.request("GET", resource="media", querystring_params=dict(q=q, page=page, results_per_page=results_per_page))

    def email_annotation_set_user(self, a, media_count, point_count, annotation_count):
        user_ids = [a.get('user', {}).get('id')]
        annotation_set_url = "{}/geodata/annotation_set/{}".format(self.sqapi.base_url, a.get("id"))
        message = f'Hi {a.get("user",{}).get("first_name")}, <br><br>\n' \
                  f'Your annotation set "{a.get("media_collection",{}).get("name")} >> {a.get("name")}" has new ' \
                  f'suggested annotations!<br>\n' \
                  f'{self.annotator_name} processed: {media_count} media items with {point_count} points and ' \
                  f'submitted {annotation_count} annotations which will appear as "Magical Suggestions".<br><br>\n' \
                  f'To see results, click: <a href="{annotation_set_url}">{annotation_set_url}</a>'
        self.sqapi.send_user_email("SQ+ BOT: your Annotation Set has been processed", message, user_ids=user_ids)

    def classify_point(self, mediaobj, x, y, t):
        """

        :param mediaobj:
        :param x:
        :param y:
        :param t:
        :return:
        """
        print(f"media_url: {mediaobj.url}\nx: {x}\ny: {y}\nt: {t}")
        return None, 0.0

    def classify_frame(self, mediaobj, t):
        """

        :param mediaobj:
        :param t:
        :return:
        """
        print(f"media_url: {mediaobj.url}\nt: {t}")
        return None, 0.0

    def detect_points(self, mediaobj):
        """

        :param mediaobj:
        :return:
        """
        return []





