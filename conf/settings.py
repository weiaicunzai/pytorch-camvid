from datetime import datetime

#voc2012 bgr
#MEAN = (0.40450239443559777, 0.4373051208637668, 0.45734658153594476)
#STD = (0.2846743681700796, 0.27163815793569834, 0.2747289066704502)

# camvid bgr
MEAN = (0.42019099703461577, 0.41323568513979647, 0.4010048431259079)
STD = (0.30598050258519743, 0.3089986932156864, 0.3054061869915674)

CHECKPOINT_FOLDER = 'checkpoints'
LOG_FOLDER ='runs'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

DATA_PATH = '/content/drive/My Drive/dataset/camvid'

#IMAGE_SIZE = (480, 360)
IMAGE_SIZE = 473

MILESTONES = [100, 150]

IGNORE_LABEL = 255
