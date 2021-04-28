# file path
DATA_DIR = 'data/'
DATA_ORIGIN_DIR = DATA_DIR + 'origin/'
DATA_PREPROCESS_DIR = DATA_DIR + 'preprocess/'
DATASET_DIR = DATA_DIR + 'dataset/'

# forward fill within 180s, otherwise fill 0
FORWARD_FILLING_WINDOW = 180

# sampling gap size
DEFAULT_STEP_SIZE = 6

# one activation to {SAMPLE_REUSE_RATE} samples
SAMPLE_REUSE_RATE = 5

APP_CONFIG = {
    'kettle':{
        'nick': ['kettle', 'kettle', 'kettle', 'kettle', 'kettle'],
        'max_power': 3100,
        'on_power_threshold': 2000,
        'min_on_duration': 12,
        'max_off_duration': 12
    },
    'fridge':{
        'nick': ['fridge', 'fridge', 'fridge', 'freezer', 'fridge'],
        'max_power': 300,
        'on_power_threshold': 50,
        'min_on_duration': 60,
        'max_off_duration': 12
    },
    'microwave':{
        'nick': ['microwave', 'microwave', 'microwave', 'microwave', 'microwave'],
        'max_power': 3000,
        'on_power_threshold': 200,
        'min_on_duration': 12,
        'max_off_duration': 30
    },
    'dishwasher':{
        'nick': ['dishwasher', 'dish_washer', 'dishwasher', 'dishwasher', 'dishwasher'],
        'max_power': 2500,
        'on_power_threshold': 10,
        'min_on_duration': 1800,
        'max_off_duration': 1800
    },
    'washing_machine':{
        'nick': ['washing_machine', 'washing_machine', 'washing_machine', 'washing_machine', 'washer_dryer'],
        'max_power': 2500,
        'on_power_threshold': 20,
        'min_on_duration': 1800,
        'max_off_duration': 160
    }
}

JACK_KELLY_STEP_SIZE = 6
JACK_KELLY_INPUT_CONFIG = {
    'kettle': 128*JACK_KELLY_STEP_SIZE,
    'fridge': 512*JACK_KELLY_STEP_SIZE, 
    'microwave': 288*JACK_KELLY_STEP_SIZE,
    # 'dishwasher': 1024 + 512*JACK_KELLY_STEP_SIZE, 
    'dishwasher': 2048 + 1024*JACK_KELLY_STEP_SIZE,
    'washing_machine': 1024*JACK_KELLY_STEP_SIZE
}