#!/usr/bin/env bash

## Optionally pass in an API token. If no token is passed it, you will be prompted to log in
#echo -n "Enter your api_token from SQ+: "
#read -s api_token

python examples/bots/torch_bot.py \
  --model_path examples/models/oscar/inception_kelp_nokelp_20ep_9902.model \ #myckpt
  --label_map_file examples/models/oscar/inception_kelp_nokelp_20ep_9902-code_label_map.json \ #mylabelmap
  --user_group_id 55 \ #myusergroup
  --poll_delay 5 \
  --crop_perc 0.18 \
  # --api_token $api_token    # uncomment to use API key