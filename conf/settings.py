from datetime import datetime

#bgr
MEAN = (0.934340233408533, 0.933164368805413, 0.9290009326656783)

#bgr
STD = (0.20586904031872816, 0.20420952068390197, 0.20682781911842638)

CHECKPOINT_PATH = 'checkpoints'

TIME_NOW = datetime.now().isoformat()

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

DATA_PATH = '/home/baiyu/Downloads/web_crawler'

IMAGE_SIZE = 512