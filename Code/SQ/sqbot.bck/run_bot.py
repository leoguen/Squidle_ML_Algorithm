from sqapi.api import SQAPIargparser
from sqapi.annotate import get_annotator_keys, get_annotator_plugin
import argparse
from bots import *  # register all bots



# Get standalone `--annotator` argument first to set things up
argparser = argparse.ArgumentParser(add_help=False)  # allow help to passed to SQAPIargparser
argparser.add_argument("--annotator", type=str, default=None)
args, args_unknown = argparser.parse_known_args()   # parse only known args to prevent unknown argument error
assert args.annotator in get_annotator_keys(), "Argument required: --annotator. Can be one of {}".format(get_annotator_keys())

# Get class and api/module arguments
BotClass = get_annotator_plugin(annotator=args.annotator)
apiargs = SQAPIargparser.parse_args()


bot = BotClass(**vars(apiargs))


# run Annotator
bot.run(email_results=False)
