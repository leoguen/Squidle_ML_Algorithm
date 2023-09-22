import random
from sqapi.annotate import Annotator
from sqapi.request import query_filter as qf
from sqapi.helpers import cli_init, create_parser
from sqapi.media import SQMediaObject


class RandoBOT(Annotator):
    def __init__(self, **annotator_args):
        """

        Short description
        An example of an automated labelling bot that selects random class labels to assign to points.
        It provides terrible suggestions, however it provides a simple boiler-plate example of how to integrate a
        Machine Learning algorithm for label suggestions.

        """
        super().__init__(**annotator_args)
        self.possible_codes = ["ECK", "ASC", "SUB"]

    def classify_point(self, mediaobj: SQMediaObject, x, y, t):
        """
        Overridden method: predict label for x-y point
        """
        # image_data = mediaobj.data()            # cv2 image object containing media data
        # media_path = mediaobj.url               # path to media item
        print(f"CLASSIFYING: {mediaobj.url} | x: {x},  y: {y},  t: {t}")
        classifier_code = random.sample(self.possible_codes, 1)[0]
        prob = round(random.random(), 2)
        return classifier_code, prob


if __name__ == '__main__':
    # Running `bot = cli_init(RandoBOT)` would normally do all the steps below and initialise the class,
    # but in this instance we cant to add some extra commandline arguments to decide what annotation_sets to process

    # Get the cli arguments from the Class __init__ function signatures
    parser = create_parser(RandoBOT)

    # Add some additional custom cli args not related to the model
    parser.add_argument('--annotation_set_id', help="Process specific annotation_set", type=int, default=8266)
    parser.add_argument('--user_group_id', help="Process all annotation_sets contained in a specific user_group", type=int)
    parser.add_argument('--after_date', help="Process all annotation_sets after a date YYYY-MM-DD", type=str)

    args = parser.parse_args()
    args.host = 'https://staging.squidle.org'
    # Set the host, API key, and label map file for the bot
    #open text file in read mode
    text_file = open("bots/API_KEY.txt", "r")
    #read whole file to a string
    api_key = text_file.read()
    #close file
    text_file.close()
    api_token = api_key
    args.label_map_file = '/home/ubuntu/Documents/IMAS/Code/Squidle/bots/rando_bot_label_map.json'
    bot = RandoBOT(**vars(args))

    # Initialise annotation_set request using sqapi instance in Annotator class
    r = bot.sqapi.get("/api/annotation_set")

    # Filter annotation sets based on ID
    if args.annotation_set_id:
        r.filter("id", "eq", args.annotation_set_id)

    # Constrain date ranges to annotation_sets ceated after a specific date
    if args.after_date:
        r.filter("created_at", "gt", args.after_date)

    # Filter annotation_sets based on a user group
    if args.user_group_id:
        r.filter(name="usergroups", op="any", val=dict(name="id", op="eq", val=args.user_group_id))

    # Only return annotation_sets that do not already have suggestions from this user
    r.filter_not(qf("children", "any", val=qf("user_id", "eq", bot.sqapi.current_user.get("id"))))

    bot.start(r)