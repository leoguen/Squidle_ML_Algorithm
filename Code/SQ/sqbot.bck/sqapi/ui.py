import inspect
import time
from html import escape

from pick import Picker  # pick
import json
from pygments.lexers.data import JsonLexer
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer


_ui_history_stack = []


class UIComponents(object):

    @staticmethod
    def _push_history(args=None, kwargs=None):
        global _ui_history_stack
        if kwargs is None: kwargs = {}
        if args is None: args = []
        #print("Add to stack")
        method = inspect.stack()[1][3]   # method name of caller
        _ui_history_stack.append(dict(method=method, args=args, kwargs=kwargs))

    @staticmethod
    def _go_back():
        global _ui_history_stack
        #print("Attempt to go back")
        if len(_ui_history_stack) > 0:
            state = _ui_history_stack.pop(-1)
            return getattr(UIComponents, state.get("method"))(_push_state=False, *state.get("args",[]),**state.get("kwargs",{}))
        else:
            print(_ui_history_stack)
            print("Empty state history. Exiting...")
            return None

    @staticmethod
    def select_list(title, optionlist, multi_select=False, min_selection_count=1):
        """

        :param title:
        :param optionlist:
        :param multi_select:
        :param min_selection_count:
        :return:
        """
        picker = Picker(optionlist, title, multi_select=multi_select, min_selection_count=min_selection_count)
        picker.register_custom_handler(ord('\x1b'), UIComponents._cancel)
        if not multi_select:
            option, index = picker.start()
            # option, index = pick(optionlist, title, multi_select=False, min_selection_count=min_selection_count)
            return option, index
        else:
            options = picker.start()
            # options = pick(optionlist, title, multi_select=True, min_selection_count=min_selection_count)
            return [(o[0], o[1]) for o in options]  # list of [(opt, ind)]

    @staticmethod
    def _cancel(picker):
        return None, -1

    @staticmethod
    def input_json(title, json_value=u""):
        """

        :param title:
        :param json_value:
        :return:
        """
        print ("\n* SET VALUE FOR: {}. \n  Paste JSON below. Hit ESC+ENTER or CTRL+o when done.\n".format(title))
        contents = prompt(u"> ", default=json_value, multiline=True, mouse_support=True, lexer=PygmentsLexer(JsonLexer))
        # contents = []
        # while True:
        #     try:
        #         line = raw_input("")
        #     except EOFError:
        #         break
        #     contents.append(line)
        print (contents)
        return json.loads(contents) if contents else None

    @staticmethod
    def input_multi(title, fields, exclude=None):
        """

        :param title:
        :param fields:
        :param exclude:
        :return:
        """
        if exclude is None:
            exclude = []
        data = {}
        if isinstance(fields, list):
            fields = {f: None for f in fields}  # convert to dict
        for f in fields.keys():
            if f not in exclude:
                if isinstance(fields[f], (str,)) or fields[f] is None:
                    data[f] = prompt(u"\n* SET VALUE FOR: '{}'\n  > ".format(f), default=fields[f] or u"") or fields[f]
                elif isinstance(fields[f], (dict, list)):
                    data[f] = UIComponents.input_json(f, json.dumps(fields[f], indent=2))
                    #fields[f] = prompt(u"\n* SET VALUE FOR: '{}'\n  > ".format(f), default=json.dumps(fields[f]), multiline=True, mouse_support=True)  #, lexer=JsonLexer)
                else:
                    raise Exception(u"Unsupported field type for '{}'. Don't know how to handle type: {}.".format(f, type(fields[f])))
        return data

    @staticmethod
    def select_object_list(objects, title="Choose an item:", list_format="{name}", actions=None, _push_state=True, *args, **kwargs):
        """

        :param _push_state:
        :param objects:
        :param title:
        :param list_format:
        :param actions:
        :param args:
        :param kwargs:
        :return:
        """

        # If empty list, print empty message with a 1.5s delay and then, go back
        if len(objects) == 0:
            print(f"\n\n{'*'*20}\n* EMPTY LIST!\n{'*'*20}\n\n")
            time.sleep(1.5)
            return UIComponents._go_back()

        select_list = [list_format.format(**o) for o in objects]
        option, index = UIComponents.select_list(title, select_list)
        o = objects[index]

        if index < 0:
            return UIComponents._go_back()
        else:
            if _push_state is True:
                UIComponents._push_history(args=[objects], kwargs=dict(title=title, list_format=list_format, actions=actions))
            if actions is not None:
                o = UIComponents.select_action(actions, obj=o, title="Choose an action for '{}':".format(list_format.format(**o)))
            return o

    @staticmethod
    def select_action(actions, obj=None, title="Choose and action:", _push_state=True):
        actions_list = [a.get("name") for a in actions]
        _, action_index = UIComponents.select_list(title, actions_list)
        callback = actions[action_index].get("callback")

        if action_index < 0:
            return UIComponents._go_back()
        else:
            # Add history
            if _push_state is True:
                UIComponents._push_history(args=[actions], kwargs=dict(obj=obj, title=title))
            if callable(callback):
                return callback(obj) if obj is not None else callback()
            else:
                raise TypeError("callback is not callable!")


def lol2html(lol, style="", escape_html=False):
    """
    Convert list of lists to HTML table.
    Each element is converted to a string, and can optionally be HTML-escaped.

    :param lol: list of lists
    :param style: optional css string to insert into table 'style' attribute
    :param escape_html: whether or not to escape the HTML in each cell. Default `False` to allow for HTML formatting.
    :return: STRING containing HTML-formatted tables
    """
    rows = ""
    for r in lol:
        r = map(str, r)
        if escape_html:
            r = map(escape, r)
        rows += "<tr><td valign='top'>{}</td></tr>\n".format("</td><td valign='top'>".join(r))
    return "<table style='{}'>{}</table>".format(style, rows)
