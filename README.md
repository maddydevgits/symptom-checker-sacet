# Symptom Checker with User Authentication & History Tracking

A **Flask-based Symptom Checker** that allows users to **sign up, log in, and predict diseases** based on symptoms. The system also **stores user history in MongoDB Atlas** and provides **precautions and medicine recommendations**.

---

## 🚀 Features

✅ **User Authentication**
   - Signup & Login with hashed passwords (stored in MongoDB Atlas)
   - Session-based authentication

✅ **Symptom Checker**
   - Select symptoms and predict disease using **ML classification**
   - Provides **precautions & medicine recommendations**

✅ **User History Tracking**
   - Stores each disease prediction under the **logged-in user’s account**
   - Users can view their past symptom checks in **History**

✅ **Modern UI (Glassmorphism)**
   - Fully styled **Login, Signup, Symptom Checker & History Pages**
   - Responsive & Mobile-friendly

---

## 🛠️ Technologies Used

- **Python & Flask** (Backend API)
- **MongoDB Atlas** (User Data & History Storage)
- **Machine Learning (Scikit-learn, RandomForest)**
- **Flask-Bcrypt** (Password Hashing)
- **Flask-Session** (Session-based Authentication)
- **HTML, CSS, JavaScript** (Frontend UI)
- **Flask-PyMongo** (MongoDB Integration)

---

## 🏠 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/symptom-checker.git
cd symptom-checker
```

### 2️⃣ Install Dependencies
```bash
pip install flask flask-pymongo flask-bcrypt flask-session scikit-learn pandas
```

### 3️⃣ Set Up MongoDB Atlas
1. Create a **MongoDB Atlas** account
2. Create a new **database** and **collection** (`users` & `history`)
3. **Get your MongoDB connection string** and update `app.py`:
   ```python
   app.config["MONGO_URI"] = "your_mongodb_atlas_connection_string"
   ```

### 4️⃣ Run the Application
```bash
python app.py
```
**Now visit:** `http://127.0.0.1:5000/`

---

## 🎯 How to Use

### **1️⃣ Signup**
- Visit `/signup`
- Create an account (Data is stored in MongoDB Atlas)

### **2️⃣ Login**
- Visit `/`
- Enter your credentials
- After login, you’ll be redirected to the **Symptom Checker**

### **3️⃣ Check Symptoms**
- Select **symptoms** and click **"Predict Disease"**
- The system will return:
  - **Predicted Disease**
  - **Precautions**
  - **Suggested Medicines**

### **4️⃣ View History**
- Click **"History"** to view past symptom checks
- Each record contains:
  - Symptoms selected
  - Predicted disease
  - Precautions & Medicines

---

## 📂 Project Structure

```
symptom-checker/
️│—— templates/
️│   ├— index.html       # Symptom Checker UI
️│   ├— signup.html      # Signup Page
️│   ├— login.html       # Login Page
️│   └— history.html     # User's History Page
️│—— static/              # (For any additional CSS or JS files)
️│—— data/
️│   └— DiseaseAndSymptoms.csv  # Dataset used for ML model
️│—— app.py               # Main Flask Application
️│—— model.pkl            # Trained Random Forest Model
️│—— vectorizer.pkl       # TF-IDF Vectorizer
️│—— README.md            # This file
️│—— requirements.txt      # List of dependencies
```

---

## 🌟 **Future Enhancements**
💥 **JWT-based Authentication** for enhanced security  
💥 **Admin Dashboard** for managing users  
💥 **Chatbot Feature** for symptom consultation  

---

## 🤝 Contributing
Feel free to contribute! Open a pull request or raise an issue. 🚀

---

## 📜 License
This project is **open-source** and available under the **MIT License**.

