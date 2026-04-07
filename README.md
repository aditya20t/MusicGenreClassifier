---
title: Music Genre Classifier
emoji: 🎵
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
python_version: '3.10'
---

# 🎵 Music Genre Classifier

This project uses a fine-tuned **DistilHuBERT** model to classify music genres from the [GTZAN dataset](https://huggingface.co/datasets/marsyas/gtzan). It features a real-time web interface built with Gradio.

## 🚀 Features
- **Model:** Fine-tuned `ntu-spml/distilhubert` on 10 genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).
- **Interface:** Upload or record audio directly via the browser.
- **Top-K Predictions:** Displays the top 5 most likely genres with confidence scores.

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd MusicGenreClassifier
   ```

2. **Create the environment:**
   ```bash
   conda create -n audio python=3.10 -y
   conda activate audio
   conda install -c conda-forge pyarrow datasets -y
   pip install transformers accelerate torch torchaudio librosa gradio evaluate
   ```

## 💻 Usage

### Training/Fine-tuning
To train the model on your local machine:
```bash
python main.py
```

### Running the App
To launch the Gradio web interface:
```bash
python app.py
```

## 📊 Dataset
The model is trained on the **GTZAN** dataset, which consists of 1,000 audio tracks each 30 seconds long. It is the most used public dataset for evaluation in machine listening.

## 🔗 Model Link
The fine-tuned weights are available here: [aditya20t/distilhubert-musicClassifier](https://huggingface.co/aditya20t/distilhubert-musicClassifier)