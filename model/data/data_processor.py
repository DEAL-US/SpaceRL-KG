from pathlib import Path
from tqdm import tqdm
import os
from pprint import pprint as pp
import re

class DataProcessor(object):
    '''
    recieves: The names of the datasets that you want to process:
    '''
    def __init__(self, datasets):
        self.sorted_logs = {k:[] for k in datasets}
        self.data_path = Path(__file__).parent.resolve()
        self.log_path = str(Path(self.data_path))+"/logs"
        self.r = re.compile(r"(.*?):(.*?)(,|$)")

        for l in os.listdir(self.log_path):
            for d in datasets:
                if(l.startswith(d)):
                    self.sorted_logs[d].append(l)

    def extract_info_from_logs(self):
        a = tqdm(self.sorted_logs.items())
        data_dict = dict()
        i = 0
        for l in a:
            for logfile in l[1]:
                a.set_description(f" processing log data for: {logfile}")
                # Open file.
                with open(f"{self.log_path}/{logfile}", "r") as f:
                    allinfo = f.read()
                    #separate each instance of training in log.
                    infoarr = allinfo.split(" - START LOG FOR DATASET ")
                    #remove first timestamp
                    infoarr.pop(0)
                    
                    #iterate over each log.
                    for data in infoarr:
                        split_info, episodes = data.split("\n\n")
                        split_info_arr = split_info.split("\n")
                        dataset = split_info_arr.pop(0)
                        dataset = dataset.strip().replace("\"","")
                        split_info_dict = {"DATASET":dataset}

                        for d in split_info_arr:
                            k,v = d.split(": ")
                            split_info_dict[k] = v

                        episodes_dict = {}
                        episodes_arr = episodes.split("] - ")
                        episodes_arr.pop(0) #remove first timestamp
                        for e in episodes_arr:
                            e = e[:e.rfind('\n')]
                            e = e.replace("\n", ", ")
                            options, path = e.split(", Path: ")
                            
                            options_arr = self.r.findall(options)
                            ep_num = options_arr.pop(0)
                            ep_num = int(ep_num[1].strip())
                            ep_data = {}
                            ep_data["Path"] = path
                            for o in options_arr:
                                k = o[0].strip()
                                v = o[1].strip()

                                if k=="Target" or k=="Destination":
                                    ep_data[k] = v

                                elif k == "Arrival":
                                    ep_data[k] = v=="True"
                                
                                else:
                                    ep_data[k] = float(v)

                            episodes_dict[ep_num] = ep_data

                        data_dict[i] = (split_info_dict, episodes_dict)
                        i+=1
                        

processor = DataProcessor(["COUNTRIES", "FB15K-237", "KINSHIP","NELL-995","UMLS", "WN18RR"])
processor.extract_info_from_logs()
