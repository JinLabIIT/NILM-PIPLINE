# build activation
import sys
import os
import time
import threading

sys.path.append("../")
from config import *
sys.path.append("../__utility__/")
from prepocessing_utils import *

to_file = '../' + DATA_PREPROCESS_DIR + 'activation.csv'

class __(threading.Thread):
    def __init__(self, threadID, NUM_THREAD, list_task, threadLock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.num_thread = NUM_THREAD
        self.list_task = list_task
        self.threadLock = threadLock
    
    def run(self):
        for index, task in enumerate(self.list_task):
            if index % self.num_thread == self.threadID:
                appliance, house_id = task
                on_power_threshold = APP_CONFIG[appliance]['on_power_threshold']
                min_on_duration = APP_CONFIG[appliance]['min_on_duration']
                max_off_duration = APP_CONFIG[appliance]['max_off_duration']
                
                df = get_activations(house_id, appliance)
                df['appliance'] = appliance
                df['house_id'] = house_id
                
                self.threadLock.acquire()
                df.to_csv(to_file,mode='a', index=False, header=False)
                print('{}: {}-{}'.format(self.threadID, appliance, house_id))
                self.threadLock.release()
                

def __build_activation_set__(NUM_THREAD = 4):
    if not os.path.exists('../' + DATA_PREPROCESS_DIR):
        os.makedirs('../' + DATA_PREPROCESS_DIR)   
    
    list_task = []
    for appliance in APP_CONFIG.keys():
        for index, nick in enumerate(APP_CONFIG[appliance]['nick']):
            house_id = index+1
            channel_id = get_channel(index+1, nick)
            if channel_id is not None:
                list_task.append((appliance, house_id))
    
    pd.DataFrame({
        'start_time': [],
        'end_time': [],
        'appliance': [],
        'house_id': []
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
    __build_activation_set__()