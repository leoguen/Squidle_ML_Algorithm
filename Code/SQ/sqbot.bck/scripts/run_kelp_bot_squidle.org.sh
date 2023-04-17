#!/usr/bin/env bash


echo -n "Enter your api_token from SQ+: "
read -s api_token


python run_bot.py \
    --api_token $api_token \
    --annotator KerasBOT \
    --model_path models/oscar/inception_kelp_nokelp_20ep_9902.model \
    --label_map_file models/oscar/inception_kelp_nokelp_20ep_9902-code_label_map.json \
    --url https://squidle.org \
    --user_group_id 55 \
    --after_date "$(date -v -1d '+%Y-%m-%d')" \
    --email_results
#    --url https://squidle.org \
#    --user_group_id 55             # If supplied will process all visible datasets shared in that group
#    --affiliation_group_id 55      # If supplied will process all visible datasets from all members in that group
#    --annotation_set_id 55         # If supplied will process that annotation_set
