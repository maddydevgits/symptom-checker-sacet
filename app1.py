from flask import Flask, render_template, request, redirect, session, jsonify
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_session import Session
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "supersecretkey"

# MongoDB Configuration (Replace with your MongoDB Atlas connection string)
app.config["MONGO_URI"] = "mongodb://localhost:27017/sacet-symptom"
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
app.config["SESSION_TYPE"] = "filesystem"  # Use filesystem-based sessions
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

# Load dataset and train model
df = pd.read_csv('data/DiseaseAndSymptoms.csv')
df['combined_symptoms'] = df.apply(lambda row: ' '.join(row[1:].dropna().astype(str)), axis=1)
X = df['combined_symptoms']
y = df['Disease']

# Train TF-IDF Vectorizer & Random Forest Classifier
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_tfidf, y)

# Save Model & Vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Load Model & Vectorizer
with open('model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define Disease Info (Previously Created Dictionary)
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

# Extract unique symptoms
symptoms_list = sorted(set(str(symptom) for symptom in df.iloc[:, 1:].values.flatten() if pd.notna(symptom)))


### USER AUTHENTICATION ###
@app.route('/')
def home():
    if "user" in session:
        return redirect('/symptoms')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        if mongo.db.users.find_one({"email": email}):
            return "Email already registered. Try logging in.", 400

        mongo.db.users.insert_one({"name": name, "email": email, "password": password})
        return redirect('/')

    return render_template('signup.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    user = mongo.db.users.find_one({"email": email})

    if user and bcrypt.check_password_hash(user['password'], password):
        session["user"] = user["email"]
        return redirect('/symptoms')
    else:
        return "Invalid credentials", 400

@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect('/')

### SYMPTOM CHECKER ###
@app.route('/symptoms')
def symptoms():
    if "user" not in session:
        return redirect('/')
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    if "user" not in session:
        return jsonify({'error': 'User not logged in'}), 403

    selected_symptoms = request.form.getlist('symptoms')
    symptoms_text = ' '.join(selected_symptoms)

    if not symptoms_text:
        return jsonify({'error': 'No symptoms selected'}), 400

    symptoms_tfidf = tfidf_vectorizer.transform([symptoms_text])
    prediction = classifier.predict(symptoms_tfidf)[0]
    
    precautions = disease_info.get(prediction, {}).get("precautions", ["No data available"])
    medicines = disease_info.get(prediction, {}).get("medicines", ["No data available"])

    # Store History in MongoDB
    mongo.db.history.insert_one({
        "user": session["user"],
        "symptoms": selected_symptoms,
        "predicted_disease": prediction,
        "precautions": precautions,
        "medicines": medicines
    })

    return jsonify({
        'predicted_disease': prediction,
        'precautions': precautions,
        'medicines': medicines
    })

@app.route('/history')
def history():
    if "user" not in session:
        return redirect('/')
    
    user_history = list(mongo.db.history.find({"user": session["user"]}, {"_id": 0, "user": 0}))
    return render_template('history.html', history=user_history)

if __name__ == '__main__':
    app.run(debug=True)
