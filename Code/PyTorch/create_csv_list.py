import requests
import io
import pandas as pd
import json
from datetime import datetime
import argparse
import os
import time

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
    def get_annotation_set(self, API_TOKEN, URL, id_list, bool_translation, export_scheme, anno_name, prob_name):
        # Ensure output files exist
        open(anno_name, 'w').close()
        open(prob_name, 'w').close()

        counter = 0
        for id in id_list:
            df = self._process_single_id(API_TOKEN, URL, id, export_scheme, bool_translation, prob_name)
            if df is None:
                continue

            # Write to CSV (header only once)
            header = (counter == 0)
            df.to_csv(path_or_buf=anno_name, mode='a', index=False, header=header)

            counter += 1
            print(f'CSV file number {counter}/{len(id_list)} saved for id: {id}')

        # Create pivot table
        self.create_pivot_csv(df, anno_name)
        return anno_name


    def _process_single_id(self, API_TOKEN, URL, id, export_scheme, bool_translation, prob_name):
        """Handle one annotation set ID end-to-end"""
        annotation_url = '/' + str(id) + export_scheme
        with requests.Session() as s:
            head = {'auth-token': API_TOKEN}
            download = s.get(URL + annotation_url, headers=head)

            if download.status_code == 401:
                print(f"Access to ID {id} is unauthorized")
                return None

            data = json.loads(download.content.decode('utf-8'))

            if "result_url" in data and "status_url" in data:
                decoded_content = self._poll_and_download(s, data, head, id)
            else:
                decoded_content = download.content.decode("utf-8")

            df = pd.read_csv(io.StringIO(decoded_content))

            return self._filter_columns(df, bool_translation, id, prob_name)


    def _poll_and_download(self, session, data, head, id):
        """Poll until background task is finished, then download result"""
        status_url = "https://squidle.org" + data["status_url"]
        result_url = "https://squidle.org" + data["result_url"]

        while True:
            resp = session.get(status_url, headers=head)
            try:
                status = resp.json()
            except ValueError:
                print(f"Non-JSON response from {status_url}:")
                print(resp.text[:200])
                time.sleep(2)
                continue

            state = status.get("status", "").upper()
            if state == "DONE":
                print(f"Download for Annotation Set {id} finished!")
                break
            elif state == "STARTED":
                print(f"Download for Annotation Set {id} ongoing")
            time.sleep(2)

        result = session.get(result_url, headers=head)
        return result.content.decode("utf-8")


    def _filter_columns(self, df, bool_translation, id, prob_name):
        """Select only desired columns, log problems if mismatch"""
        try:
            if bool_translation:
                df = df[['label.id', 'label.name', 'label.uuid', 'point.id',
                        'point.media.deployment.campaign.key',
                        'point.media.path_best', 'point.x', 'point.y', 'tag_names',
                        'label.translated.id','label.translated.lineage_names',
                        'label.translated.name','label.translated.translation_info',
                        'label.translated.uuid']]
            else:
                df = df[['label.id', 'label.name', 'label.uuid', 'point.id',
                        'point.media.deployment.campaign.key',
                        'point.media.path_best', 'point.x', 'point.y', 'tag_names']]
        except Exception:
            print(f"Problem with annotationset {id}, will be skipped")
            print(df.head(0))
            with open(prob_name, 'a') as f:
                f.write('\n ID: ' + str(id) + '\n')
                f.write(str(df.head(0)))
            return None

        return df

    
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
        current_date = datetime.now().strftime('%Y%m%d_%H%M%S')  # This will give you the date in the format YYYYMMDD
        parser = argparse.ArgumentParser(description='Create CSV List for Annotation Sets')
        parser.add_argument('--bool_translation', type=bool, default=True, help='Decide whether to obtain the used labelling scheme or translate it to a different one as well')
        parser.add_argument('--export_scheme', type=str, default='/export?supplementary_annotation_set_id=&template=dataframe.csv&disposition=attachment&include_columns=["label.id","label.uuid","label.name","label.lineage_names","comment","needs_review","tag_names","updated_at","point.id","point.x","point.y","point.t","point.data","point.is_targeted","point.media.id","point.media.key","point.media.path_best","point.pose.timestamp","point.pose.lat","point.pose.lon","point.pose.alt","point.pose.dep","point.media.deployment.key","point.media.deployment.campaign.key","label.translated.id","label.translated.uuid","label.translated.name","label.translated.lineage_names","label.translated.translation_info"]&f={"operations":[{"module":"pandas","method":"json_normalize"},{"method":"sort_index","kwargs":{"axis":1}}]}&q={"filters":[{"name":"label_id","op":"is_not_null"}]}&translate={"vocab_registry_keys":["worms","caab","catami"],"target_label_scheme_id":"1"}', help='Export scheme for annotation sets')
        parser.add_argument('--anno_name', type=str, default=f'./Annotationsets/{current_date}_Full_Annotation_List.csv', help='Annotation set name')
        #parser.add_argument('--anno_name', type=str, default=f'PyTorch/Annotationsets/{current_date}_Full_Annotation_List.csv', help='Annotation set name')
        parser.add_argument('--prob_name', type=str, default=f'./Annotationsets/{current_date}_Problem_Files_Full_Annotation_List.csv', help='Problem file name')
        #parser.add_argument('--prob_name', type=str, default=f'PyTorch/Annotationsets/{current_date}_Problem_Files_Full_Annotation_List.csv', help='Problem file name')
        
        parser.add_argument('--api_token', type=str, default="PyTorch/API_TOKEN.txt", help='Enter API token or path to API token saved in .txt')

        return parser.parse_args()

if __name__ == "__main__":
    URL = "https://squidle.org/api"
    dataset_url = '/annotation_set'
    data = create_csv_list()
    args = data.parse_args()

    # Check if the provided value is a file path
    if os.path.isfile(args.api_token):
        api_token = data.load_token(args.api_token)
    else:
        # Assume the provided value is the API token itself
        api_token = args.api_token

    # Get List of all Annotationset ID accessible to you
    id_list = data.get_annotation_id(api_token, URL,  dataset_url)
    
    # To save list after created first time
    #with open("ID_List.txt", "w") as f:
    #    for id in id_list:
    #        f.write(str(id) + "\n")

    # To load created list
    #with open("/home/leo/Documents/IMAS/Code/PyTorch/ID_List.txt", "r") as f:
    #    id_list = [line.strip() for line in f]

    annotation_url = URL+dataset_url

    annotation_name = data.get_annotation_set(api_token, annotation_url,  id_list, args.bool_translation, args.export_scheme, args.anno_name, args.prob_name)
