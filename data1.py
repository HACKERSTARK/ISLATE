import os
import shutil
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
IMAGE_PATH = os.path.join(WORKSPACE_PATH, 'images')
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'models')
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'pre-trained-models')
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'pipeline.config')
CHECKPOINT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

src_config = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config')
dst_config = CONFIG_PATH
shutil.copy(src_config, dst_config)

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 9
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'checkpoint', 'ckpt-31')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text.encode('utf-8'))
