# Examples

## BOTS

### [demo_randobot.py](bots/demo_randobot.py)
An example of a random label bot that can be run as a service from the command line. Running with `--help` will 
display cli options and parameters.

```shell
> python demo_randobot.py --help

usage: demo_randobot.py [-h] [--host HOST] [--api_key API_KEY] [--annotator_name ANNOTATOR_NAME] [--prob_thresh PROB_THRESH] [--poll_delay POLL_DELAY] [--label_map_file LABEL_MAP_FILE] [--verbosity VERBOSITY] [--email_results] [--annotation_set_id ANNOTATION_SET_ID]
                        [--user_group_id USER_GROUP_ID] [--after_date AFTER_DATE]

Short description
An example of an automated labelling bot that selects random class labels to assign to points.
It provides terrible suggestions, however it provides a simple boiler-plate example of how to integrate a
Machine Learning algorithm for label suggestions.

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           the Squidle+ instance hostname
                        type: str  | default: https://squidle.org
  --api_key API_KEY     the API key for the user on that `host`. If omitted, you'll be asked to log in.
                        type: str  | default: None
  --annotator_name ANNOTATOR_NAME
                        used for the name of the annotation_set, defaults to ClassName
                        type: str  | default: None
  --prob_thresh PROB_THRESH
                        probability threshold for submitted labels, only submitted if p > prob_thresh
                        type: float  | default: 0.5
  --poll_delay POLL_DELAY
                        the poll delay for running the loop. To run once, set poll_delay = -1
                        type: int  | default: 5
  --label_map_file LABEL_MAP_FILE
                        path to a local file that contains the label mappings
                        type: str  | default: None
  --verbosity VERBOSITY
                        the verbosity of the output (0,1,2,3)
                        type: int  | default: 2
  --email_results       flag to optionally send an email upon completion
                        type: bool  | default: None
  --annotation_set_id ANNOTATION_SET_ID
                        Process specific annotation_set
  --user_group_id USER_GROUP_ID
                        Process all annotation_sets contained in a specific user_group
  --after_date AFTER_DATE
                        Process all annotation_sets after a date YYYY-MM-DD
```


### [keras_cnn_bot.py](bots/keras_cnn_bot.py)
a real classifier that uses a tensor_flow/keras model for 
classifying points in images. The class received a number of additional parameters that can all be passed in as 
command line arguments. Run `python keras_cnn_bot.py --help` for more information. This shows an example of how to 
include additional model-specific arguments along with other parameters, as well as how to process the images and 
for a real classifier.

```shell
> python examples/bots/keras_cnn_bot.py --help 

usage: keras_cnn_bot.py [-h] [--model_path MODEL_PATH] [--patch_width PATCH_WIDTH] [--patch_height PATCH_HEIGHT] [--network NETWORK] [--patch_path PATCH_PATH] [--host HOST] [--api_key API_KEY] [--annotator_name ANNOTATOR_NAME] [--prob_thresh PROB_THRESH]
                        [--poll_delay POLL_DELAY] [--label_map_file LABEL_MAP_FILE] [--verbosity VERBOSITY] [--email_results] [--annotation_set_id ANNOTATION_SET_ID] [--user_group_id USER_GROUP_ID] [--after_date AFTER_DATE]

Uses keras to run a tensorflow model

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        the path of the tensorflow model
                        type: str  | default: None
  --patch_width PATCH_WIDTH
                        with of the patches
                        type: int  | default: 299
  --patch_height PATCH_HEIGHT
                        height of the patches
                        type: int  | default: 299
  --network NETWORK     the network to use for the model
                        type: str  | default: keras.applications.inception_v3
  --patch_path PATCH_PATH
                        an optional path to cache the patches (useful if doing multiple runs on the same points)
                        type: str  | default: None
  --host HOST           the Squidle+ instance hostname
                        type: str  | default: https://squidle.org
  --api_key API_KEY     the API key for the user on that `host`. If omitted, you'll be asked to log in.
                        type: str  | default: None
  --annotator_name ANNOTATOR_NAME
                        used for the name of the annotation_set, defaults to ClassName
                        type: str  | default: None
  --prob_thresh PROB_THRESH
                        probability threshold for submitted labels, only submitted if p > prob_thresh
                        type: float  | default: 0.5
  --poll_delay POLL_DELAY
                        the poll delay for running the loop. To run once, set poll_delay = -1
                        type: int  | default: 5
  --label_map_file LABEL_MAP_FILE
                        path to a local file that contains the label mappings
                        type: str  | default: None
  --verbosity VERBOSITY
                        the verbosity of the output (0,1,2,3)
                        type: int  | default: 2
  --email_results       flag to optionally send an email upon completion
                        type: bool  | default: None
  --annotation_set_id ANNOTATION_SET_ID
                        Process specific annotation_set
  --user_group_id USER_GROUP_ID
                        Process all annotation_sets contained in a specific user_group
  --after_date AFTER_DATE
                        Process all annotation_sets after a date YYYY-MM-DD
```
