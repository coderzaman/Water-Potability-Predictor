import gradio as gr
import pandas as pd
import pickle

# --- 1. Load the Saved Pipeline ---
try:
    with open('water_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    print("Pipeline loaded successfully!")
except Exception as e:
    print(f"Error loading pipeline: {e}")

# --- 2. Define the Prediction Function ---
def predict_water_quality(ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity):
    
    # A. Feature Engineering: Classify pH
    def classify_ph(ph_val):
        if ph_val < 6.5: return 'Acidic'
        elif ph_val > 8.5: return 'Alkaline'
        else: return 'Neutral'
    
    ph_group = classify_ph(ph)
    
    # B. Manual One-Hot Encoding
    ph_Acidic = 1 if ph_group == 'Acidic' else 0
    ph_Alkaline = 1 if ph_group == 'Alkaline' else 0
    ph_Neutral = 1 if ph_group == 'Neutral' else 0
    
    # C. Create a DataFrame from input with EXACT column order as training
    # Note: 'ph_Alkaline' must come before 'ph_Neutral' alphabetically
    input_data = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity],
        'ph_Acidic': [ph_Acidic],
        'ph_Alkaline': [ph_Alkaline],
        'ph_Neutral': [ph_Neutral]
    })
    
    # D. Predict using the Pipeline
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1] # Probability of Class 1
    
    # E. Format the Output
    if prediction == 1:
        return f"Potable (Safe to Drink) \nProbability of Safety: {probability:.2%}"
    else:
        return f"Not Potable (Unsafe) \nProbability of Safety: {probability:.2%}"

# --- 3. Create Gradio Interface ---
iface = gr.Interface(
    fn=predict_water_quality,
    inputs=[
        gr.Number(label="pH Value (0-14)", value=7.0),
        gr.Number(label="Hardness (mg/L)", value=200.0),
        gr.Number(label="Solids (ppm)", value=20000.0),
        gr.Number(label="Chloramines (ppm)", value=7.0),
        gr.Number(label="Sulfate (mg/L)", value=300.0),
        gr.Number(label="Conductivity (μS/cm)", value=400.0),
        gr.Number(label="Organic Carbon (ppm)", value=15.0),
        gr.Number(label="Trihalomethanes (μg/L)", value=60.0),
        gr.Number(label="Turbidity (NTU)", value=4.0)
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=2),
    title="Water Potability Predictor",
    description="Enter the water quality parameters below to check if the water is safe for human consumption."
)

# --- 4. Launch the App ---
if __name__ == "__main__":
    iface.launch()