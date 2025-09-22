import os
import cv2
import numpy as np
import tensorflow as tf
import threading
import time
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from gtts import gTTS
from playsound import playsound
from object_detection.utils import label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# === CONFIG ===
CONFIG_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = 'Tensorflow/workspace/models/my_ssd_mobnet/'
LABEL_MAP_PATH = os.path.join('Tensorflow/workspace/annotations', 'label_map.pbtxt')

# === LOAD MODEL ===
def load_model():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-31')).expect_partial()
    return model

@tf.function
def detect_fn(image, model):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    return detections

# === INIT MODEL & LABEL MAP ===
detection_model = load_model()
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

# === TEXT-TO-SPEECH (English only) ===
def speak_text(text):
    try:
        tts = gTTS(text=text, lang="en")
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        messagebox.showerror("Speech Error", str(e))

# === APP CLASS ===
class HandDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Detection + Voice")
        self.root.geometry("1000x750")

        self.video_label = Label(root)
        self.video_label.pack()

        self.string_display = Label(root, text="", font=("Helvetica", 20))
        self.string_display.pack(pady=10)

        # Buttons
        btn_frame = Frame(root)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Speak", command=self.speak_string, width=12).grid(row=0, column=0, padx=10)
        Button(btn_frame, text="Clear", command=self.clear_string, width=12).grid(row=0, column=1, padx=10)
        Button(btn_frame, text="Exit", command=self.exit_app, width=12).grid(row=0, column=2, padx=10)

        self.cap = cv2.VideoCapture(0)
        self.final_string = ""
        self.last_character = ""
        self.enter_detected_time = None
        self.running = True
        self.update_video()

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, detection_model)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'],
            detections['detection_classes'] + 1,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=0.5,
            agnostic_mode=False
        )

        # Detection Logic
        if len(detections['detection_scores']) > 0 and detections['detection_scores'][0] > 0.8:
            class_id = detections['detection_classes'][0]
            label = category_index[class_id + 1]['name'].lower()
            current_time = time.time()

            if label == 'enter':
                print("Enter Detected")

            elif label == 'space':
                if self.last_character != ' ':
                    self.final_string += ' '
                    self.last_character = ' '
                    self.string_display.config(text=self.final_string)

            elif label != self.last_character:
                self.final_string += label.upper()
                self.last_character = label
                self.string_display.config(text=self.final_string)

        # Display video
        im = Image.fromarray(image_np)
        imgtk = ImageTk.PhotoImage(image=im)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def speak_string(self):
        if self.final_string.strip():
            threading.Thread(target=speak_text, args=(self.final_string,)).start()

    def clear_string(self):
        self.final_string = ""
        self.last_character = ""
        self.string_display.config(text="")

    def exit_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

# === RUN ===
if __name__ == '__main__':
    root = Tk()
    app = HandDetectorApp(root)
    root.mainloop()
