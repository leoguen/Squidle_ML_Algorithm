from sqapi.api import SQAPIargparser
from sqapi.annotate import get_annotator_keys, get_annotator_plugin
import argparse
from bots import *  # register all bots



# Get standalone `--annotator` argument first to set things up
argparser = argparse.ArgumentParser(add_help=False)  # allow help to passed to SQAPIargparser
argparser.add_argument("--annotator", type=str, default='RandoBOT')
args, args_unknown = argparser.parse_known_args()   # parse only known args to prevent unknown argument error
assert args.annotator in get_annotator_keys(), "Argument required: --annotator. Can be one of {}".format(get_annotator_keys())

# Get class and api/module arguments
#BotClass = get_annotator_plugin(annotator=args.annotator)
#apiargs = SQAPIargparser.parse_args()

#For testing 
BotClass = get_annotator_plugin(annotator='RandoBOT')
#apiargs = ''
# Set the host, API key, and label map file for the bot
#open text file in read mode
text_file = open("bots/API_KEY.txt", "r")
#read whole file to a string
api_key = text_file.read()
#close file
text_file.close()
api_token = api_key
url='https://staging.squidle.org'  
user_group_id = '176 '
label_map_file= '/home/leo/Documents/IMAS/Code/SQ/sqbot/models/demo/random_demo-code_label_map.json'

#!!! Check how to get variables into **vars format
#bot = BotClass(**vars(apiargs))
bot = BotClass(api_token=api_token, url=url,user_group_id= user_group_id, annotator_name='RandoBOT', label_map_file=label_map_file)
#bot= BotClass([api_token, url, user_group_id])

# run Annotator
bot.run(email_results=False)
