from transformers import pipeline
import pandas as pd
from google.colab import files

# ✅ Read CSV without manual intervention
locations_df = pd.read_csv("India pincode list.csv")

# ✅ Load Hugging Face AI Model (Optimized for Colab T4 GPU)
science_generator = pipeline(
    "text-generation",
    model="facebook/opt-2.7b",  # ✅ More Powerful, Works Well on Colab
    device=0  # ✅ Uses GPU in Google Colab
)

def generate_science_fact(city, state, lat, lon):
    prompt = (
        f"{city}, {state} (Lat: {lat}, Lon: {lon}) has a unique climate and geography. "
        f"Explain a scientific fact about this region, focusing on climate, geography, or environmental impact."
    )
    response = science_generator(
        prompt,
        max_length=250,  # ✅ Increased Length for Detailed Answer
        num_return_sequences=1,
        truncation=True,
        do_sample=True,  # ✅ Adds Variation to Responses
        temperature=0.6   # ✅ Controls Creativity (Lower = More Factual)
    )
    return response[0]["generated_text"]

# ✅ Ask user for a pincode input
input_pincode = input("Enter a pincode: ").strip()

# ✅ Lookup location details by pincode
location = locations_df[locations_df['Pincode'].astype(str) == input_pincode]

if location.empty:
    print(f"No location found for pincode: {input_pincode}")
else:
    row = location.iloc[0]  # Get the first matching row
    # 👉 Accessing columns using correct names (e.g., 'OfficeName', 'StateName')
    city = row["OfficeName"]
    state = row["StateName"]  # Corrected column name to 'StateName'
    latitude = row["Latitude"]
    longitude = row["Longitude"]


    science_fact = generate_science_fact(city, state, latitude, longitude)

    # ✅ Display Results
    print("\n✅ Science Behind Your Location:")
    print(f"📍 Pincode: {input_pincode}")
    print(f"🏙 City: {city}, {state}")
    print(f"🌍 Latitude: {latitude}, Longitude: {longitude}")
    print(f"\n🔬 Science Fact: {science_fact}\n")