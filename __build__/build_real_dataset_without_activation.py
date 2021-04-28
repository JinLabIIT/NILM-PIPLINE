# build real dataset without activation
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
        for index, (appliance, house_id, number_samples) in enumerate(self.list_task):
            if index % self.num_thread == self.threadID:
                t = time.time()
                data = {
                    'app_power': [],
                    'agg_power': [],
                    'start_time':[],
                    'end_time': []
                }
                if house_id == 1 and appliance == 'fridge':
                    number_samples //= 10
                l_app, l_agg, l_st, l_et = random_samples(house_id, appliance, number_samples)
                
                data['app_power'] += l_app
                data['agg_power'] += l_agg
                data['start_time'] += l_st
                data['end_time'] += l_et
                
                df = pd.DataFrame(data)
                df['house_id'] = house_id
                df['appliance'] = appliance
                df['contain_activation'] = 0
                df['is_synthetic'] = 0

                self.threadLock.acquire()
                df.to_csv(to_file, mode='a', index=False, header=False)
                print('{}: {}-{} | time: {:.2f}'.format(
                    self.threadID, appliance, house_id, time.time()-t))
                self.threadLock.release()  
                
def __build_dataset__(NUM_THREAD = 4):
    if not os.path.exists('../' + DATASET_DIR):
        os.makedirs('../' + DATASET_DIR)   
    
    list_task = []
    df = load_all_activations()
    df_done = pd.read_csv(to_file)
    for appliance, house_id in [ (appliance, i) for appliance in APP_CONFIG.keys() for i in range(1,6)]:
        size = df[(df['appliance'] == appliance) & (df['house_id'] == house_id)].shape[0] - df_done[(df_done['appliance'] == appliance) & (df_done['house_id'] == house_id)].shape[0]
        if size > 0:
            list_task.append((appliance, house_id, size*SAMPLE_REUSE_RATE))
    
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
    ____build_dataset__()