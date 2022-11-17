import csv
import os
from os import path
import pandas as pd
import urllib.request
import cv2
import numpy as np
import json
import requests

class extract_metadata():
    
    '''
    Loads API Token from .txt file
    '''
    def load_token(self, here):
        with open(here + '/API_TOKEN.txt', "r") as file:
            API_TOKEN = file.read().rstrip()
        #print(API_TOKEN)
        return API_TOKEN
    
    '''
    Get Dataset from SQ which 
    will be posted to new mediaset in SQ
    '''
    def get_dataset(self, API_TOKEN, dataset_url, here):
        with requests.Session() as s:
            head = {'auth-token': API_TOKEN}
            download = s.get(dataset_url, headers=head)
            decoded_content = download.content.decode('utf-8')

            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            my_list = list(cr)
            # open the file in the write mode
            with open( here + '/test.csv', 'w') as myfile:
                wr = csv.writer(myfile, 
                                quoting=csv.QUOTE_ALL)
                wr.writerows(my_list)
            return my_list

    def create_media_collection(self, API_TOKEN, url):
        # get user_id
        params={"q": json.dumps({"filters":[{"name":"api_token","op":"eq","val":API_TOKEN}],"single":True})}
        user_id = requests.get(url+"/api/users", params=params).json().get("id")

        # parameters for new media_collection
        payload = {
            "name": "API test 01",
            "description":"Testing API-created media_collection",
            "user_id": user_id,
        }

        headers={
            "auth-token": API_TOKEN,
            "Content-type": "application/json",
            "Accept": "application/json",
        }

        # Make request
        r = requests.post(
            url+"/api/media_collection",
            headers=headers,
            json=payload,
        )

        # Check results
        new_media_collection = r.json()
        media_collection_id = new_media_collection.get("id")
        return media_collection_id, user_id

    def create_annotation_set(self, API_TOKEN, url):
        # Setup initial params
        # get user_id
        params={"q": json.dumps({"filters":[{"name":"api_token","op":"eq","val":API_TOKEN}],"single":True})}
        user_id = requests.get(url+"/api/users", params=params).json().get("id")

        # parameters for new media_collection
        payload = {
            "name": "Annotation test 01",
            "description":"Testing API-created Annotationset",
            "user_id": user_id,
        }

        headers={
            "auth-token": API_TOKEN,
            "Content-type": "application/json",
            "Accept": "application/json",
        }

        # Make request
        r = requests.post(
            url+"/api/annotation_set_file",
            headers=headers,
            json=payload,
        )

        # Check results
        new_annotation_set = r.json()
        annotation_set_id = new_annotation_set.get("id")
        return annotation_set_id, user_id

    def post_media_collection(self, API_TOKEN, data_url, media_collection_id, url, user_id):
        
        headers={
            "auth-token": API_TOKEN,
            "Content-type": "application/json",
            "Accept": "application/json",
        }
        
        data = {
            "user_id": user_id,
            "include_colums" : '["id","key","path_best","pose.timestamp","pose.lat","pose.lon","pose.alt","pose.dep"]',
            "file_url": data_url
        }

        requests.post(url + '/api/media_collection/' + str(media_collection_id) + '/media', 
                    json=data, 
                    headers=headers)
    
    def post_annotation_set(self, API_TOKEN, dataset, media_collection_id, url, user_id):
        headers={
            "auth-token": API_TOKEN,
            "Content-type": "application/json",
            "Accept": "application/json",
        }
        
        data = {
            "user_id": user_id,
            "include_colums" : '["id","key","path_best","pose.timestamp","pose.lat","pose.lon","pose.alt","pose.dep"]',
            "template" : 'dataframe.csv',

        }

        requests.post(url + '/api/media_collection/' + str(media_collection_id) + '/media', 
                    json=data, 
                    headers=headers)

if __name__ == "__main__":
    url = "https://squidle.org"
    annotation_url = 'https://www.squidle.org/api/annotation_set/4058/export?template=dataframe.csv&disposition=attachment&include_columns=[%22label.id%22,%22label.uuid%22,%22label.name%22,%22label.lineage_names%22,%22comment%22,%22needs_review%22,%22tag_names%22,%22updated_at%22,%22point.id%22,%22point.x%22,%22point.y%22,%22point.t%22,%22point.data%22,%22point.media.id%22,%22point.media.key%22,%22point.media.path_best%22,%22point.pose.timestamp%22,%22point.pose.lat%22,%22point.pose.lon%22,%22point.pose.alt%22,%22point.pose.dep%22]&f={%22operations%22:[{%22module%22:%22pandas%22,%22method%22:%22json_normalize%22},{%22method%22:%22sort_index%22,%22kwargs%22:{%22axis%22:1}}]}&q={%22filters%22:[{%22name%22:%22label_id%22,%22op%22:%22is_not_null%22}]}&translate={%22vocab_registry_keys%22:[%22worms%22,+%22caab%22,+%22catami%22]}'
    media_url = 'https://www.squidle.org/api/media_collection/3825/export?template=dataframe.csv&disposition=attachment&include_columns=["id","key","path_best","pose.timestamp","pose.lat","pose.lon","pose.alt","pose.dep"]&f={"operations":[{"module":"pandas","method":"json_normalize"}]}&q={"filters":[]}'
    here = os.path.dirname(os.path.abspath(__file__))
    
    data = extract_metadata()
    API_TOKEN = data.load_token(here)
    
    #dataset = data.get_dataset(API_TOKEN, dataset_url, here)
    
    # Begin for Testing purpose
    with open(here + '/test_media.csv', newline='') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    data_url = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/AUV/auv_viewer_data/images/Tasmania201006/r20100604_230524_lanterns_14_shallow/full_res/PR_20100604_231041_196_LC16.jpg'
    media_collection_id, user_id = data.create_media_collection(API_TOKEN, url)
    print(media_collection_id, user_id)
    #annotation_set_id, user_id = data.create_annotation_set(API_TOKEN, url)
    #print(annotation_set_id, user_id)
    
    data.post_media_collection(API_TOKEN, data_url, media_collection_id, url, user_id)
    #data.post_annotation_set(API_TOKEN, dataset, annotation_set_id, url, user_id)

'''
Notes for Future reference
'''
#search for annotation_set matching id=84 that have the label_id=310 
#/api/annotation_set/84/annotations?include_columns=["label_id","id","color","point","point.media_id","point.id"]&q={"filters":[{"name":"label_id","op":"eq","val":310}]}&results_per_page=100

# for a known database
#'https://www.squidle.org/api/annotation_set/4058/export?template=dataframe.csv&disposition=attachment&include_columns=[%22label.id%22,%22label.uuid%22,%22label.name%22,%22label.lineage_names%22,%22comment%22,%22needs_review%22,%22tag_names%22,%22updated_at%22,%22point.id%22,%22point.x%22,%22point.y%22,%22point.t%22,%22point.data%22,%22point.media.id%22,%22point.media.key%22,%22point.media.path_best%22,%22point.pose.timestamp%22,%22point.pose.lat%22,%22point.pose.lon%22,%22point.pose.alt%22,%22point.pose.dep%22]&f={%22operations%22:[{%22module%22:%22pandas%22,%22method%22:%22json_normalize%22},{%22method%22:%22sort_index%22,%22kwargs%22:{%22axis%22:1}}]}&q={%22filters%22:[{%22name%22:%22label_id%22,%22op%22:%22is_not_null%22}]}&translate={%22vocab_registry_keys%22:[%22worms%22,+%22caab%22,+%22catami%22]}',

#request all data with annotation ecklonia
#/api/media?q={"filters":[{"name":"annotations","op":"any","val":{"name":"annotations","op":"any","val":{"name":"label","op":"has","val":{"name":"name","op":"ilike","val":"%ecklonia%"}}}}]}