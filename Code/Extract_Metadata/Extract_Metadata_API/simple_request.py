import csv
import requests
import os


CSV_URL = 'https://www.squidle.org/api/annotation_set/4058/export?template=dataframe.csv&disposition=attachment&include_columns=[%22label.id%22,%22label.uuid%22,%22label.name%22,%22label.lineage_names%22,%22comment%22,%22needs_review%22,%22tag_names%22,%22updated_at%22,%22point.id%22,%22point.x%22,%22point.y%22,%22point.t%22,%22point.data%22,%22point.media.id%22,%22point.media.key%22,%22point.media.path_best%22,%22point.pose.timestamp%22,%22point.pose.lat%22,%22point.pose.lon%22,%22point.pose.alt%22,%22point.pose.dep%22]&f={%22operations%22:[{%22module%22:%22pandas%22,%22method%22:%22json_normalize%22},{%22method%22:%22sort_index%22,%22kwargs%22:{%22axis%22:1}}]}&q={%22filters%22:[{%22name%22:%22label_id%22,%22op%22:%22is_not_null%22}]}&translate={%22vocab_registry_keys%22:[%22worms%22,+%22caab%22,+%22catami%22]}'
here = os.path.dirname(os.path.abspath(__file__))
with open(here + '/API_TOKEN.txt', "r") as file:
    API_TOKEN = file.read().rstrip()

with requests.Session() as s:
    myToken = 'dd6d7052ff66724bf17aab5b543e2ffb35b945f08eca65f4e573faf4'
    myUrl = 'https://www.squidle.org/api/annotation_set/4058/export?template=dataframe.csv&disposition=attachment&include_columns=[%22label.id%22,%22label.uuid%22,%22label.name%22,%22label.lineage_names%22,%22comment%22,%22needs_review%22,%22tag_names%22,%22updated_at%22,%22point.id%22,%22point.x%22,%22point.y%22,%22point.t%22,%22point.data%22,%22point.media.id%22,%22point.media.key%22,%22point.media.path_best%22,%22point.pose.timestamp%22,%22point.pose.lat%22,%22point.pose.lon%22,%22point.pose.alt%22,%22point.pose.dep%22]&f={%22operations%22:[{%22module%22:%22pandas%22,%22method%22:%22json_normalize%22},{%22method%22:%22sort_index%22,%22kwargs%22:{%22axis%22:1}}]}&q={%22filters%22:[{%22name%22:%22label_id%22,%22op%22:%22is_not_null%22}]}&translate={%22vocab_registry_keys%22:[%22worms%22,+%22caab%22,+%22catami%22]}'
    head = {'auth-token': myToken}
    download = s.get(myUrl, headers=head)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        print(row)