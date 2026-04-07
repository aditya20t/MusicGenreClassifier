import gradio as gr
from transformers import pipeline

# Point this to your new Hub model ID
model_id = "aditya20t/distilhubert-musicClassifier"
classifier = pipeline("audio-classification", model=model_id)

def predict(audio):
    preds = classifier(audio)
    return {p["label"]: p["score"] for p in preds}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(num_top_classes=5),
    title="Music Genre Classifier"
)

demo.launch()