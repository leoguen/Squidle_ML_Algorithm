import ipywidgets as widgets
from IPython.display import display, Markdown as MD, HTML
from .request import query_filter as f
import json
# import ipyplot


def JSON(data, indent=2, **kw):
    return MD(f"```json\n{json.dumps(data, indent=indent, **kw)}\n```")


class APIConnector:
    def __init__(self, sqapi):
        self.sqapi = sqapi
        # Get instance parameters
        self.host_input = widgets.Text(value=self.sqapi.host, description='Host', layout={"width":"450px"})
        self.api_key_input = widgets.Password(placeholder=f'Enter your api key from host', description='API key', layout={"width":"450px"})
        self.connect_btn = widgets.Button(description="Connect to API!")
        self.output = widgets.Output()
        self.connect_btn.on_click(self.connect_to_api)

    def connect_to_api(self, *args, **kw):
        self.output.clear_output()
        with self.output:
            if not self.api_key_input.value.strip():
                display(MD(f"> **ERROR:** you have not set your `{self.api_key_input.description}`!"))
            else:
                self.sqapi.connect(host=self.host_input.value, api_key=self.api_key_input.value)
                user_info = f"{self.sqapi.current_user.get('first_name')} {self.sqapi.current_user.get('last_name')} from {self.sqapi.current_user.get('affiliations_names')}"
                display(MD(f"> **SUCCESS - You're connected!** Welcome *{user_info}*."))
                
    def widgets(self):
        return widgets.VBox([widgets.HBox([widgets.VBox([self.host_input,self.api_key_input]), self.connect_btn]),self.output])


class QueryControls():
    def __init__(self, page=1, page_sizes=None, action="UPDATE!", header=None, on_get=None, filts=None):
        """
        """
        page_sizes = page_sizes or [50, 100, 500]
        layout = widgets.Layout(width="150px")
        self.page = widgets.BoundedIntText(min=1, value=1,description='Page:', layout=layout)
        self.results_per_page = widgets.Dropdown(options=page_sizes,description='#Results:',layout=layout)
        self.output = widgets.Output(layout={'border': '1px solid #cccccc'})
        self.header = MD(header if header is not None else "")
        self.button = widgets.Button(description=action)
        self.filts = filts or {}
        self.controls = widgets.HBox([*self.filts.values(), self.page, self.results_per_page, self.button])
        self._on_get = on_get
        
        self.button.on_click(self.get_content)
        
        display(self.header, self.controls, self.output)
        self.set_content(MD(f"> NO DATA YET: click '{action}' to load..."))
            
    def set_content(self, content):
        if not isinstance(content, list):
            content = [content]            
        self.output.clear_output()
        with self.output:
            display(*content)
        return self
                
    def get_content(self, *args, **kwargs):
        if callable(self._on_get):
            self.set_content(MD("> LOADING..."))
            content = self._on_get(
                page=self.page.value, 
                results_per_page=self.results_per_page.value, 
                **{k:v.value for k,v in self.filts.items()}
            )
            self.set_content(content)
        else:
            self.set_content(MD("..."))
        #return self
        
    def on_get(self, get_content):
        self._on_get = get_content
        # self.get_content()
        return self
    
class SelectControl(QueryControls):
    def __init__(self, q, key_format="{id} | {name}", value_field="id", multiselect=True, null_option=None, rows=None, *args, **kwargs):
        self._q = q
        self._key_format = key_format
        self._value_field = value_field
        self._null_option = null_option
        self._multiselect = multiselect
        if multiselect:
            self.select = widgets.SelectMultiple(options=[], layout={"width":"auto"}, rows=rows or 12)
        else:
            self.select = widgets.Select(options=[], layout={"width":"auto"}, rows=rows or 1)
        super().__init__(on_get=self.get_items, *args, **kwargs)
        
    def get_items(self, page, results_per_page, **filts):
        q = self._q() if callable(self._q) else self._q
        r = q.page(page).results_per_page(results_per_page).execute().json()
        opts = {self._null_option:None} if self._null_option is not None else {}
        for i in r.get('objects'): opts[self._key_format.format(**i)] = i[self._value_field]
        self.select.options = opts
        info = f"#### Showing {len(r['objects'])} / {r['num_results']} results [page: {r['page']} / {r['total_pages']}]. "
        if self._multiselect: info += "Select multiple using `CTL/CMD` or `SHIFT`"
        return [MD(info), self.select]
    
    def value(self):
        return self.select.value
    
    
class HTMLControl(QueryControls):
    def __init__(self, q, template=None, page_sizes=None, *args, **kwargs):
        self._q = q
        self._template = template
        super().__init__(on_get=self.get_html, page_sizes=page_sizes or [10, 20, 100], *args, **kwargs)
    
    def get_html(self, page, results_per_page, **filts):
        q = self._q() if callable(self._q) else self._q
        if self._template is not None:
            q = q.template(self._template)
        r = q.page(page).results_per_page(results_per_page).execute()
        return HTML(r.text)
    

# class ImageGrid(QueryControls):
#     def __init__(self, q, get_image, img_width=150, page_sizes=None, *args, **kwargs):
#         self._q = q
#         self._get_image = get_image
#         self._img_width=img_width
#         self._info = None
#         super().__init__(on_get=self.get_grid, page_sizes=page_sizes or [12, 24, 36, 48], *args, **kwargs)
    
#     def get_grid(self, page, results_per_page, **filts):
#         super().set_content(MD("> LOADING..."))
#         q = self._q() if callable(self._q) else self._q
#         r = q.page(page).results_per_page(results_per_page).execute().json()
#         images, labels = [], []
#         self._info = f"Showing {len(r['objects'])} / {r['num_results']} results [page: {r['page']} / {r['total_pages']}]"
#         for i in r.get("objects"):
#             images.append(self._get_image(i))
#             labels.append(i.get("label").get("name"))
            
#         return images, labels
    
#     def get_content(self, *args, **kwargs):
#         images, labels = self.get_grid(page=self.page.value, results_per_page=self.results_per_page.value)
#         self.output.clear_output()
#         with self.output:
#             display(MD(f"#### {self._info}"))
#             ipyplot.plot_images(images, labels=labels, img_width=self._img_width) 
#         return self


    
    
    
    
    
    
    
    
    
    
    
# # OLD STUFF - too specific, try use more general functions above    
    
# class AnnotationSetSelect(QueryControls):
#     def __init__(self, api=None, filts=None, *args, **kwargs):
#         self._api = api
#         self.select = widgets.SelectMultiple(options=[], layout={"width":"auto"}, rows=12)
#         user_filt_options = ["is_owner","can_edit","can_view"]
#         self.user_level = widgets.Dropdown(options=user_filt_options, description='current_user:')
#         filts = filts or {}
#         filts.update({'user_level': self.user_level})
#         super().__init__(on_get=self.get_annotation_sets, filts=filts, *args, **kwargs)
        
#     def get_annotation_sets(self, page, results_per_page, user_level=None, user_group_ids=None, **filts):
#         q = self._api.get("/api/annotation_set", results_per_page=results_per_page, page=page).order_by("created_at", "desc")
#         if user_level is not None:
#             q.filter(f"current_user_{user_level}", "eq", True)
#         if user_group_ids is not None:
#             pass # TODO: add usergroup filter here
#         r = q.execute().json()
#         key = "{id} | {created_at} | {annotation_count} annotations | {media_collection[name]} >> {name}"
#         self.select.options = {key.format(**i) : i['id'] for i in r.get("objects")}
#         info = f"#### Showing {len(r['objects'])} / {r['num_results']} results [page: {r['page']} / {r['total_pages']}]"
#         return [MD(info), self.select]
        

# class LabelTally(QueryControls):
#     def __init__(self, api=None, page_sizes=None, *args, **kwargs):
#         self._api = api
#         super().__init__(on_get=self.get_label_tally, page_sizes=page_sizes or [10, 20, 100], *args, **kwargs)
    
#     def get_label_tally(self, page, results_per_page, annotation_set_ids=None, **filts):
#         ids = list(annotation_set_ids or [])
#         if len(ids) > 0:
#             r = self._api.get('/api/annotation/tally/label', results_per_page=results_per_page, page=page)\
#                 .filter('annotation_set_id', 'in', ids)\
#                 .template("models/annotation_set/plot_tally.html").execute()
#             return HTML(r.text)
#         return MD("> Select one or more `annotation_sets` from the search above!")
    
    
# class MediaList(QueryControls):
#     def __init__(self, api=None, *args, **kwargs):
#         self._api = api
#         super().__init__(on_get=self.get_media, *args, **kwargs)
    
#     # Build custom widget to display thumbnails matching image search
#     def get_media(self, page, results_per_page, label_search=None, annotation_set_ids=None, **filters):
#         ids = list(annotation_set_ids or [])
#         search_terms = (label_search or "").strip()
#         q = self._api.get('/api/media', results_per_page=results_per_page, page=page, qsparams={"include_link":"true"})\
#             .order_by("timestamp_start")\
#             .template("models/media/list_thumbnails.html")
#         info = "Showing `media` items"
#         if len(search_terms) > 0:
#             or_filt = [
#                 f("annotations", "any", f("annotations", "any", f("label", "has", f("name", "ilike", f"%25{s.strip()}%25")))) 
#                      for s in search_terms.split(",")]
#             q.filters_or(or_filt)
#             info += f", with one or more `labels` matching '{search_terms}'"
#         if len(ids) > 0:
#             q.filter("annotations", "any", f("annotation_set_id", "in", ids))
#             info += f"from `annotation_sets` {ids}"

#         r = q.execute()
#         return [MD(f"{info}\n\n**Click thumbnails for more details on each `media` item**"), HTML(r.text)]
    