import requests
import io
from io import BytesIO
import csv
import os
import pandas as pd
import json

class create_csv_list():
    
    '''
    Loads API Token from .txt file
    '''
    def load_token(self, HERE):
        with open(HERE + '/API_TOKEN.txt', "r") as file:
            API_TOKEN = file.read().rstrip()
        #print(API_TOKEN)
        return API_TOKEN

    '''
    Get Dataset from SQ which contains all accessible annotation sets
    '''
    def get_annotation_id(self, API_TOKEN, URL, dataset_url, here):
        with requests.Session() as s:
            id_list = []
            head = {'auth-token': API_TOKEN}
            RESULTS_PP = 100
            # Get info about number of page numbers
            parameters = '?results_per_page={}&page=0'.format(RESULTS_PP)
            download = s.get(URL + dataset_url + parameters, headers=head)
            decoded_content = download.content.decode('utf-8')
            # Parse the json string to a python dictionary
            data = json.loads(decoded_content)
            print('No of total pages: {} with maximum of {} entries.'.format(data['total_pages'], RESULTS_PP))
            print('Starting download ...')
            for i in range(data['total_pages']+1):
                parameters = '?results_per_page={}&page={}'.format(RESULTS_PP, i)
                download = s.get(URL + dataset_url + parameters, headers=head)
                decoded_content = download.content.decode('utf-8')
                # Parse the json string to a python dictionary
                data = json.loads(decoded_content)

                for object in data['objects']:
                    id_list.append(object['id'])

            return id_list

    '''
    Create a .csv file with all entries
    '''
    def get_annotation_set(self, API_TOKEN, URL,  id_list, HERE):
        # Create .csv file to append to 
        NAME = '/Annotation_Sets/Full_Annotation_List.csv'
        with open(HERE + NAME, 'w') as creating_new_csv_file: 
            pass
        
        counter = 0
        # Iterate through all annotation ids
        for id in id_list: 
            annotation_url = '/' + str(id) + '/export?template=dataframe.csv&disposition=attachment&include_columns=["label.id","label.uuid","label.name","tag_names","point.id","point.x","point.y","point.t","point.data","point.media.id","point.media.path_best","point.pose.timestamp","point.pose.lat","point.pose.lon","point.pose.alt","point.pose.dep","label.translated.id","label.translated.uuid","label.translated.name","label.translated.lineage_names","label.translated.translation_info"]&f={"operations":[{"module":"pandas","method":"json_normalize"},{"method":"sort_index","kwargs":{"axis":1}}]}&q={"filters":[{"name":"point","op":"has","val":{"name":"has_xy","op":"eq","val":true}},{"name":"label_id","op":"is_not_null"}]}&translate={"vocab_registry_keys":["worms","caab","catami"],"target_label_scheme_id":null}'
            
            

            with requests.Session() as s:
                head = {'auth-token': API_TOKEN}
                #print(URL+annotation_url)
                download = s.get(URL+annotation_url, headers=head)
                decoded_content = download.content.decode('utf-8')
                df = pd.read_csv(io.StringIO(decoded_content))
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                    #print(df.shape[])
                    print(df.head(0))
                
                try:
                    df = df[['label.id', 'label.name', 'label.uuid', 'point.id', 'point.media.path_best', 'point.x', 'point.y', 'tag_names']]
                except:
                    print("Problem with annotationset {}, will be skipped".format(id))
                    continue
                

                # Skips header if this is not the first object
                header = False
                if id == id_list[0]:
                    header = True
                
                #df.drop("Unnamed: 0", axis=1, inplace=True)
                df.to_csv(path_or_buf=HERE+NAME, mode='a', index=False, header=header)
            counter += 1
            print('CSV file number {}/{} saved for id: {}'.format(counter,len(id_list) , id))
        return NAME

if __name__ == "__main__":
    URL = "https://squidle.org/api"
    dataset_url = '/annotation_set'
    HERE = os.path.dirname(os.path.abspath(__file__))
    data = create_csv_list()
    API_TOKEN = data.load_token(HERE)
    
    id_list = data.get_annotation_id(API_TOKEN, URL,  dataset_url, HERE)
    
    
    # For faster testing pruposes a list file is just loaded
    '''
    # empty list to read list from a file
    id_list = []
    
    # open file and read the content in a list
    with open(HERE + '/id_list.txt', 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            id_list.append(x)
    '''

    print('Retrieved a list of {} annotation sets.'.format(len(id_list)))
    annotation_url = URL+dataset_url
    NAME = data.get_annotation_set(API_TOKEN, annotation_url,  id_list, HERE)
    # Name given for testing purpose
    # data.clean_up_dataset(HERE, NAME)
