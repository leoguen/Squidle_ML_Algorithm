import requests
import io
import pandas as pd
import json
from datetime import datetime
import argparse
import os

class create_csv_list():
    
    '''
    Loads API Token from .txt file
    '''
    def load_token(self, api_token_path):
        try:
            with open(api_token_path, "r") as file:
                api_token = file.read().strip()
                return api_token
        except Exception as e:
            print(f"Error reading API token from file: {str(e)}")
            return None

    '''
    Get Dataset from SQ which contains all accessible annotation sets
    '''
    def get_annotation_id(self, API_TOKEN, URL, dataset_url):
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
    def get_annotation_set(self, API_TOKEN, URL,  id_list, bool_translation, export_scheme, anno_name, prob_name):
        # Create .csv file to append to 
        anno_name = anno_name
        prob_name = prob_name
        
        with open(anno_name, 'w') as creating_new_csv_file: 
            pass

        with open(prob_name, 'w') as creating_new_csv_file: 
            pass
        
        counter = 0
        # Iterate through all annotation ids
        for id in id_list: 
            annotation_url = '/' + str(id) + export_scheme
            with requests.Session() as s:
                head = {'auth-token': API_TOKEN}
                #print(URL+annotation_url)
                download = s.get(URL+annotation_url, headers=head)
                decoded_content = download.content.decode('utf-8')
                df = pd.read_csv(io.StringIO(decoded_content))
                
                try:
                    if bool_translation:
                        df = df[['label.id', 'label.name', 'label.uuid', 'point.id','point.media.deployment.campaign.key' , 'point.media.path_best', 'point.x', 'point.y', 'tag_names', 'label.translated.id','label.translated.lineage_names', 'label.translated.name','label.translated.translation_info','label.translated.uuid' ]]
                    else: 
                        df = df[['label.id', 'label.name', 'label.uuid', 'point.id','point.media.deployment.campaign.key' , 'point.media.path_best', 'point.x', 'point.y', 'tag_names']]
                except:
                    print("Problem with annotationset {}, will be skipped".format(id))
                    print(df.head(0))
                    with open(prob_name, 'a') as f:
                        f.write('\n ID: ' + str(id) + '\n')
                        f.write(str(df.head(0)))
                    continue
                
                # Skips header if this is not the first object
                header = False
                if id == id_list[0]:
                    header = True
                
                df.to_csv(path_or_buf=anno_name, mode='a', index=False, header=header)
                
            counter += 1
            print('CSV file number {}/{} saved for id: {}'.format(counter,len(id_list) , id))
        # create pivot table
        self.create_pivot_csv(df, anno_name)
        return anno_name
    
    def create_pivot_csv(self, df, anno_name):
        pivot_df = df.pivot_table(index = ['label.translated.lineage_names'], aggfunc ='size')
        pivot_df = pivot_df.sort_values(ascending=False)
        # Find the index where you want to insert the text (before the file extension)
        index = anno_name.rfind(".csv")

        if index != -1:
            # Insert the text before the file extension
            new_anno_name = anno_name[:index] + "_pivot_table" + anno_name[index:]
        else:
            print("Invalid file path format (must end with '.csv').")
        pivot_df.to_csv(new_anno_name)
    
    def api_parse_args(self):
        parser = argparse.ArgumentParser(description='Enter API token or path to API token saved in .txt')
        parser.add_argument('--api_token', type=str, default="/home/ubuntu/Documents/IMAS/API_TOKEN.txt", help='Enter API token or path to API token saved in .txt')
        args = parser.parse_args()

        # Check if the provided value is a file path
        if os.path.isfile(args.api_token):
            api_token = self.load_token(args.api_token)
        else:
            # Assume the provided value is the API token itself
            api_token = args.api_token
        return api_token

    def parse_args(self):
        current_date = datetime.now().strftime('%Y%m%d')  # This will give you the date in the format YYYYMMDD
        parser = argparse.ArgumentParser(description='Create CSV List for Annotation Sets')
        parser.add_argument('--bool_translation', type=bool, default=True, help='Decide whether to obtain the used labelling scheme or translate it to a different one as well')
        parser.add_argument('--export_scheme', type=str, default='/export?supplementary_annotation_set_id=&template=dataframe.csv&disposition=attachment&include_columns=["label.id","label.uuid","label.name","label.lineage_names","comment","needs_review","tag_names","updated_at","point.id","point.x","point.y","point.t","point.data","point.is_targeted","point.media.id","point.media.key","point.media.path_best","point.pose.timestamp","point.pose.lat","point.pose.lon","point.pose.alt","point.pose.dep","point.media.deployment.key","point.media.deployment.campaign.key","label.translated.id","label.translated.uuid","label.translated.name","label.translated.lineage_names","label.translated.translation_info"]&f={"operations":[{"module":"pandas","method":"json_normalize"},{"method":"sort_index","kwargs":{"axis":1}}]}&q={"filters":[{"name":"label_id","op":"is_not_null"}]}&translate={"vocab_registry_keys":["worms","caab","catami"],"target_label_scheme_id":"1"}', help='Export scheme for annotation sets')
        parser.add_argument('--anno_name', type=str, default=f'./Annotationsets/{current_date}_Full_Annotation_List.csv', help='Annotation set name')
        parser.add_argument('--prob_name', type=str, default=f'./Annotationsets/{current_date}_Problem_Files_Full_Annotation_List.csv', help='Problem file name')
        parser.add_argument('--api_token', type=str, default="/home/ubuntu/Documents/IMAS/API_TOKEN.txt", help='Enter API token or path to API token saved in .txt')

        return parser.parse_args()

if __name__ == "__main__":
    URL = "https://squidle.org/api"
    dataset_url = '/annotation_set'
    data = create_csv_list()
    
    # Read API Token
    #api_token = data.api_parse_args()
    
    # Get List of all Annotationset ID accessible to you
    
    #print('Retrieved a list of {} annotation sets.'.format(len(id_list)))

    args = data.parse_args()

    # Check if the provided value is a file path
    if os.path.isfile(args.api_token):
        api_token = data.load_token(args.api_token)
    else:
        # Assume the provided value is the API token itself
        api_token = args.api_token

    id_list = data.get_annotation_id(api_token, URL,  dataset_url)
    annotation_url = URL+dataset_url

    annotation_name = data.get_annotation_set(api_token, annotation_url,  id_list, args.bool_translation, args.export_scheme, args.anno_name, args.prob_name)
