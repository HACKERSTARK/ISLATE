WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

labels = [
    {'name': 'A', 'id': 1},
    {'name': 'B', 'id': 2},
    {'name': 'D', 'id': 3},
    {'name': 'L', 'id': 4},
    {'name': 'O', 'id': 5},
    {'name': 'U', 'id': 6},
    {'name': 'W', 'id': 7},
    {'name': 'Space', 'id': 8},
    {'name': 'Enter', 'id': 9},
]

# Writing label_map.pbtxt
with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item {\n')
        f.write('\tname: "{}"\n'.format(label['name']))
        f.write('\tid: {}\n'.format(label['id']))
        f.write('}\n')

# Generating TFRecord files (run in subprocess)
import os
os.system(f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {IMAGE_PATH}/train -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/train.record")
os.system(f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {IMAGE_PATH}/test -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/test.record")
