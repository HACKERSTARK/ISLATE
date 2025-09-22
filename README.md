# ISLATE: Indian Sign Language Translator & Enhancer  

A **real-time Indian Sign Language (ISL) gesture recognition system** built with **TensorFlow**, **SSD MobileNet**, and **Tkinter**.  
The project translates ISL gestures into text and provides **multilingual voice output** in 11 Indian languages.  

---

## 🚀 Features  
- **Real-time gesture recognition** using **SSD MobileNet v2** with ~90% accuracy.  
- **Tkinter-based GUI** for gesture-to-text conversion from webcam input.  
- **Multilingual speech output** powered by **Google Translate API** (11 Indian languages).  
- **Contextual sentence generation** from 50+ ISL gestures for smoother communication.  

---

## 📂 Project Structure  

├── app.py # Tkinter GUI for ISL translation 
├── collectdata.py # Script to collect ISL gesture images
├── data.py # Preprocessing & dataset handling
├── data1.py # Additional helper script
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
├── workspace
│ ├── annotations # Label maps, TFRecords
│ ├── images # Training/test/collected gesture images
│ ├── models # Custom trained models
│ └── pre-trained-models # Pre-trained SSD MobileNet model
└── Tensorflow
├── labelImg # Labeling tool
├── models # TensorFlow Object Detection API
└── scripts # Helper scripts
---

## ⚙️ Installation  

### 1. Clone the repository  
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Create and activate virtual environment
python -m venv tf38_env
source tf38_env/bin/activate   # On Linux/Mac
tf38_env\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt
4. Setup TensorFlow Object Detection API

Make sure you have the TensorFlow models repo cloned inside Tensorflow/models/.
Follow official setup instructions
________________________________________________________________________________________________________________
▶️ Usage
1. Collect ISL Gesture Data
python collectdata.py

2. Train the Model

Update pipeline.config inside workspace/models/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/ and then run:

python Tensorflow/models/research/object_detection/model_main_tf2.py \
  --pipeline_config_path=workspace/models/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config \
  --model_dir=workspace/models/ \
  --alsologtostderr

3. Run Real-Time ISL Translator
python app.py
_________________________________________________________________________________________________________________________

🎤 Example Workflow

Show an ISL gesture in front of your webcam.

The system detects and converts it into text.

Text is translated into the selected Indian language.

Audio output is played using Google Translate API.
_________________________________________________________________________________________________________________________

📊 Dataset & Training

~50+ ISL gestures.

Custom collected dataset using collectdata.py.

Augmented for better accuracy.

Trained with SSD MobileNet v2 (320x320) for real-time performance.
_________________________________________________________________________________________________________________________

🛠️ Tech Stack

TensorFlow 2.x – Model training & inference

OpenCV – Real-time webcam feed processing

Tkinter – GUI development

Google Translate API – Multilingual speech output

SSD MobileNet v2 – Lightweight real-time detection model
_________________________________________________________________________________________________________________________

📌 Future Improvements

Expand gesture dataset for full ISL vocabulary.

Improve sentence formation using NLP.

Deploy as a mobile app for accessibility.
_________________________________________________________________________________________________________________________

🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License.


---
