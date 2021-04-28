# build real dataset with activation
import sys
import os
import time
import threading

sys.path.append("../")
from config import *
sys.path.append("../__utility__/")
from prepocessing_utils import *

to_file = '../' + DATASET_DIR + 'dataset_x{}.csv'.format(SAMPLE_REUSE_RATE)

class __(threading.Thread):
    def __init__(self, threadID, NUM_THREAD, list_task, threadLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.num_thread = NUM_THREAD
        self.list_task = list_task
        self.threadLock = threadLock
    
    def run(self):
        for index, (appliance, house_id) in enumerate(self.list_task):
            if index % self.num_thread == self.threadID:
                t = time.time()
                df_app = load_origin_df(house_id, APP_CONFIG[appliance]['nick'][house_id-1])
                df_agg = load_origin_df(house_id, 'aggregate')
                data = {
                    'app_power': [],
                    'agg_power': [],
                    'start_time':[],
                    'end_time': []
                }
                for _, (start_time, end_time) in load_activations(house_id, appliance)[['start_time', 'end_time']].iterrows():
                    # house 1 fridge is too much, so we have to limit it.
                    if house_id == 1 and appliance == 'fridge' and _%10 != 0:
                        continue
                    l_app, l_agg, l_st, l_ed = get_samples(appliance, df_app, df_agg, start_time, end_time, SAMPLE_REUSE_RATE)

                    data['app_power'] += l_app
                    data['agg_power'] += l_agg
                    data['start_time'] += l_st
                    data['end_time'] += l_ed
                df = pd.DataFrame(data)
                df['house_id'] = house_id
                df['appliance'] = appliance
                df['contain_activation'] = 1
                df['is_synthetic'] = 0
                
                self.threadLock.acquire()
                df.to_csv(to_file, mode='a', index=False, header=False)
                print('{}: {}-{} | time: {:.2f}'.format(
                    self.threadID, appliance, house_id, time.time()-t))
                self.threadLock.release()
                
                


def __build_dataset__(NUM_THREAD = 4):
    if not os.path.exists('../' + DATASET_DIR):
        os.makedirs('../' + DATASET_DIR)   
    list_task = [ (appliance, house_id) for index, (appliance, house_id) in load_all_activations()[['appliance', 'house_id']].drop_duplicates().iterrows()]
    if not os.path.exists(to_file):
        pd.DataFrame({
            'app_power': [],
            'agg_power': [],
            'start_time':[],
            'end_time': [],
            'house_id': [],
            'appliance': [],
            'contain_activation': [],
            'is_synthetic': []
        }).to_csv(to_file, index=False)
    
    start_time = time.time()
    threadLock = threading.Lock()
    threads = []
    for i in range(NUM_THREAD):
        thread = __(i, NUM_THREAD, list_task, threadLock)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    print(time.time() - start_time)

if __name__ == "__main__":
    __build_dataset__()