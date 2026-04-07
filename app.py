import gradio as gr
from transformers import pipeline

# Point this to your new Hub model ID
model_id = "aditya20t/distilhubert-musicClassifier"
# Explicitly set the device (0 for GPU, -1 for CPU)
classifier = pipeline("audio-classification", model=model_id)

def predict(audio):
    if audio is None:
        return None
    preds = classifier(audio)
    return {p["label"]: p["score"] for p in preds}

# Custom CSS for a cleaner look
custom_css = """
#title {text-align: center;}
#description {text-align: center; margin-bottom: 20px;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎵 Music Genre Classifier", elem_id="title")
    gr.Markdown("Upload an audio clip (up to 30s) to identify its music genre.", elem_id="description")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath", 
                label="Upload Audio or Record",
                sources=["upload", "microphone"]
            )
            submit_btn = gr.Button("Analyze Genre", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Predictions")

    # Add Examples (ensure these files exist in your directory or use URLs)
    gr.Examples(
        examples=[], # Add paths to local .wav files here if available
        inputs=audio_input
    )

    submit_btn.click(
        fn=predict,
        inputs=audio_input,
        outputs=label_output,
    )

if __name__ == "__main__":
    demo.launch()