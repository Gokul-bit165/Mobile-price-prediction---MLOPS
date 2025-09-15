import gradio as gr
import pickle  # if you have a trained model saved
import numpy as np

# --- Load your trained model ---
# Replace 'model.pkl' with your trained model file
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# --- Dummy regression function ---
def predict_price(battery, ram, memory, screen, camera):
    """
    Dummy regression logic: Replace this with your trained ML model
    """
    features = np.array([[battery, ram, memory, screen, camera]])
    # return model.predict(features)[0]  # Uncomment when using your model
    price = battery*0.3 + ram*0.5 + memory*1.2 + screen*20 + camera*10
    return round(price, 2)  # numeric price

# --- Gradio inputs ---
inputs = [
    gr.Slider(500, 6000, step=50, label="Battery Power (mAh)", 
              info="Battery capacity of the mobile in mAh"),
    gr.Slider(256, 8192, step=128, label="RAM (MB)", 
              info="RAM size in MB"),
    gr.Slider(4, 512, step=4, label="Internal Memory (GB)", 
              info="Internal storage in GB"),
    gr.Slider(3.0, 7.5, step=0.1, label="Screen Size (inches)", 
              info="Screen size in inches"),
    gr.Slider(2, 108, step=1, label="Primary Camera Resolution (MP)", 
              info="Camera resolution in MP")
]

# --- Gradio output ---
outputs = gr.Number(label="Predicted Price (in $)")  # Numeric output for regression

# --- Gradio interface ---
interface = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs=outputs,
    title="📱 Mobile Price Predictor (Regression)",
    description="Enter mobile specifications to predict the numeric price."
)

if __name__ == "__main__":
    interface.launch()
