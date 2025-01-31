from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('data/DiseaseAndSymptoms.csv')

# Combine all symptoms into a single text column
df['combined_symptoms'] = df.apply(lambda row: ' '.join(row[1:].dropna().astype(str)), axis=1)

# Separate features (symptoms) and target (disease)
X = df['combined_symptoms']
y = df['Disease']

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Transform data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Initialize and train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_tfidf, y)

# Save model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Extract unique symptoms
symptoms_list = set()
for row in df.iloc[:, 1:].values.flatten():
    if pd.notna(row):
        symptoms_list.add(row)
symptoms_list = sorted(symptoms_list)

# Define disease precautions and medicines
# Creating a comprehensive dictionary with precautions and medicines for each disease
disease_info = {
    "Fungal infection": {
        "precautions": ["Keep skin dry and clean", "Use antifungal creams", "Wear breathable clothing", "Avoid sharing personal items"],
        "medicines": ["Clotrimazole", "Ketoconazole", "Fluconazole"]
    },
    "Allergy": {
        "precautions": ["Avoid allergens", "Use antihistamines", "Wear protective masks", "Keep windows closed during pollen season"],
        "medicines": ["Loratadine", "Cetirizine", "Diphenhydramine"]
    },
    "GERD": {
        "precautions": ["Avoid spicy food", "Eat smaller meals", "Do not lie down immediately after eating", "Maintain a healthy weight"],
        "medicines": ["Omeprazole", "Ranitidine", "Esomeprazole"]
    },
    "Chronic cholestasis": {
        "precautions": ["Follow a low-fat diet", "Avoid alcohol", "Take vitamin supplements", "Exercise regularly"],
        "medicines": ["Ursodeoxycholic acid", "Cholestyramine", "Vitamin K supplements"]
    },
    "Drug Reaction": {
        "precautions": ["Discontinue the suspected drug", "Consult a doctor immediately", "Use antihistamines if allergic", "Monitor symptoms closely"],
        "medicines": ["Antihistamines", "Corticosteroids", "Epinephrine (in severe cases)"]
    },
    "Peptic ulcer disease": {
        "precautions": ["Avoid spicy and acidic foods", "Reduce stress", "Do not smoke", "Limit alcohol consumption"],
        "medicines": ["Omeprazole", "Ranitidine", "Antacids"]
    },
    "AIDS": {
        "precautions": ["Practice safe sex", "Avoid sharing needles", "Get regular medical check-ups", "Maintain a healthy diet"],
        "medicines": ["Antiretroviral therapy (ART)", "Tenofovir", "Lamivudine"]
    },
    "Diabetes": {
        "precautions": ["Maintain a healthy diet", "Exercise regularly", "Monitor blood sugar levels", "Avoid excessive sugar intake"],
        "medicines": ["Metformin", "Insulin", "Glipizide"]
    },
    "Gastroenteritis": {
        "precautions": ["Stay hydrated", "Eat light meals", "Avoid dairy products", "Wash hands regularly"],
        "medicines": ["ORS (Oral Rehydration Salts)", "Loperamide", "Probiotics"]
    },
    "Bronchial Asthma": {
        "precautions": ["Avoid dust and smoke", "Use inhalers as prescribed", "Keep an emergency inhaler", "Stay away from allergens"],
        "medicines": ["Salbutamol", "Fluticasone", "Montelukast"]
    },
    "Hypertension": {
        "precautions": ["Reduce salt intake", "Exercise regularly", "Avoid stress", "Monitor blood pressure regularly"],
        "medicines": ["Amlodipine", "Losartan", "Hydrochlorothiazide"]
    },
    "Migraine": {
        "precautions": ["Avoid bright lights and loud noises", "Maintain a regular sleep schedule", "Stay hydrated", "Avoid caffeine and alcohol"],
        "medicines": ["Sumatriptan", "Ibuprofen", "Paracetamol"]
    },
    "Cervical spondylosis": {
        "precautions": ["Do neck exercises", "Maintain good posture", "Use ergonomic furniture", "Apply heat or cold packs"],
        "medicines": ["Ibuprofen", "Paracetamol", "Muscle relaxants"]
    },
    "Paralysis (brain hemorrhage)": {
        "precautions": ["Monitor blood pressure", "Maintain a healthy diet", "Avoid smoking", "Do regular physiotherapy"],
        "medicines": ["Blood thinners", "Rehabilitation therapy", "Physiotherapy"]
    },
    "Jaundice": {
        "precautions": ["Stay hydrated", "Avoid alcohol", "Follow a healthy diet", "Get enough rest"],
        "medicines": ["Liver supplements", "Vitamin B-complex", "Cholestyramine"]
    },
    "Malaria": {
        "precautions": ["Use mosquito repellent", "Sleep under mosquito nets", "Avoid stagnant water", "Wear full-sleeve clothing"],
        "medicines": ["Chloroquine", "Artemether", "Primaquine"]
    },
    "Chickenpox": {
        "precautions": ["Avoid scratching", "Use calamine lotion", "Take lukewarm baths", "Stay isolated"],
        "medicines": ["Acyclovir", "Antihistamines", "Paracetamol"]
    },
    "Dengue": {
        "precautions": ["Stay hydrated", "Use mosquito repellents", "Avoid aspirin", "Get plenty of rest"],
        "medicines": ["Paracetamol", "Electrolytes", "Pain relievers"]
    },
    "Typhoid": {
        "precautions": ["Drink boiled water", "Avoid street food", "Eat home-cooked meals", "Get vaccinated"],
        "medicines": ["Azithromycin", "Ciprofloxacin", "Cefixime"]
    },
    "Hepatitis A": {
        "precautions": ["Maintain hygiene", "Avoid raw food", "Get vaccinated", "Stay hydrated"],
        "medicines": ["Supportive care", "Liver supplements", "Pain relievers"]
    },
    "Tuberculosis": {
        "precautions": ["Complete the full course of medication", "Cover mouth while coughing", "Maintain proper ventilation", "Eat a protein-rich diet"],
        "medicines": ["Isoniazid", "Rifampin", "Pyrazinamide"]
    },
    "Common Cold": {
        "precautions": ["Drink warm fluids", "Rest well", "Avoid cold weather", "Use steam inhalation"],
        "medicines": ["Paracetamol", "Vitamin C", "Antihistamines"]
    },
    "Pneumonia": {
        "precautions": ["Get vaccinated", "Avoid smoking", "Practice good hygiene", "Stay hydrated"],
        "medicines": ["Antibiotics", "Cough syrups", "Paracetamol"]
    },
    "Arthritis": {
        "precautions": ["Do low-impact exercises", "Maintain a healthy weight", "Use heating pads", "Take omega-3 supplements"],
        "medicines": ["Ibuprofen", "Diclofenac", "Glucosamine"]
    },
    "Urinary tract infection": {
        "precautions": ["Drink plenty of water", "Maintain hygiene", "Urinate frequently", "Avoid caffeine and alcohol"],
        "medicines": ["Ciprofloxacin", "Nitrofurantoin", "Trimethoprim"]
    },
    "Psoriasis": {
        "precautions": ["Keep skin moisturized", "Avoid smoking", "Use medicated shampoos", "Reduce stress"],
        "medicines": ["Methotrexate", "Cyclosporine", "Topical corticosteroids"]
    },
    "Impetigo": {
        "precautions": ["Wash hands frequently", "Avoid scratching", "Keep the affected area clean", "Use antiseptic creams"],
        "medicines": ["Mupirocin", "Cephalexin", "Amoxicillin"]
    }
}


# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    symptoms_text = ' '.join(selected_symptoms)

    if not symptoms_text:
        return jsonify({'error': 'No symptoms selected'}), 400

    # Transform input symptoms
    symptoms_tfidf = tfidf_vectorizer.transform([symptoms_text])

    # Predict disease
    prediction = classifier.predict(symptoms_tfidf)[0]

    # Fetch precautions & medicines
    precautions = disease_info.get(prediction, {}).get("precautions", ["No data available"])
    medicines = disease_info.get(prediction, {}).get("medicines", ["No data available"])

    return jsonify({
        'predicted_disease': prediction,
        'precautions': precautions,
        'medicines': medicines
    })

if __name__ == '__main__':
    app.run(debug=True)
