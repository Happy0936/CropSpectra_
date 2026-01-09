from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import bcrypt
import os
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import json
import pickle
import csv
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from deep_translator import GoogleTranslator


app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


@app.context_processor
def inject_user():
    return dict(user=session.get("user"))



# ---------------- DATABASE SETUP ----------------
DB_NAME = "users.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------------- LOAD ML MODELS ----------------
model = tf.keras.models.load_model("crop_disease_model.h5")
with open("classes.json") as f:
    class_names = json.load(f)
with open("disease_info.json") as f:
    disease_info = json.load(f)
with open("train_features.pkl", "rb") as f:
    train_features = pickle.load(f)
with open("train_labels.pkl", "rb") as f:
    train_labels = pickle.load(f)


feature_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
feature_model.trainable = False


# ---------------- HOME PAGE ----------------
@app.route("/")
def index():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template("home.html", user=session.get("user"))


# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]


        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("signup"))


        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()


        # Check if username or email exists
        cursor.execute("SELECT * FROM users WHERE username=? OR email=?", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            flash("Username or Email already registered!", "error")
            conn.close()
            return redirect(url_for("signup"))


        # Hash password
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hashed_pw = hashed_pw.decode('utf-8')  # store as string


        cursor.execute("INSERT INTO users (name, username, email, password) VALUES (?, ?, ?, ?)",
                       (name, username, email, hashed_pw))
        conn.commit()
        conn.close()


        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))


    return render_template("signup.html")


# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username_email = request.form["username_email"]
        password = request.form["password"]


        # ---- HARDCODED LOGIN ----
        # Direct login with username: crop, password: crop
        if username_email.lower() == "crop" and password == "crop":
            session.clear()
            session["user"] = "crop"
            flash("Welcome crop! (Direct Access)", "success")
            return redirect(url_for("home"))


        # ---- NORMAL DATABASE LOGIN ----
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT username, email, password FROM users WHERE username=? OR email=?", (username_email, username_email))
        user = cursor.fetchone()
        conn.close()


        if user:
            stored_pw = user[2].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_pw):
                session.clear()            
                session["user"] = user[0]  

                flash(f"Welcome {user[0]}!", "success")
                return redirect(url_for("home"))
            else:
                flash("Invalid username/email or password!", "error")
                return redirect(url_for("login"))
        else:
            flash("User not found! Please signup.", "error")
            return redirect(url_for("signup"))


    return render_template("login.html", user=None)


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("login"))


# ---------------- FORGOT PASSWORD (placeholder) ----------------
@app.route("/forgot_password")
def forgot_password():
    flash("Password reset functionality is not implemented yet.", "error")
    return redirect(url_for("login"))


# ---------------- PREDICT PAGE ----------------
@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if "user" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))


    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file uploaded!", "error")
            return redirect(url_for("predict_page"))


        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!", "error")
            return redirect(url_for("predict_page"))


        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)


        # ---- Predict disease ----
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        
         img_array_exp = preprocess_input(
                np.expand_dims(img_array, axis=0)
            )

        pred = model.predict(img_array_exp)
        result_index = np.argmax(pred, axis=1)[0]
        predicted_class = class_names[result_index]


        # ---- Similarity check ----
        img_feat = image.img_to_array(img)
        img_feat = np.expand_dims(img_feat, axis=0)
        img_feat = preprocess_input(img_feat)
        img_embedding = feature_model.predict(img_feat)
        similarities = cosine_similarity(img_embedding, train_features)
        max_similarity = np.max(similarities)


        if max_similarity < 0.6:
            predicted_class = "Not a valid crop image"
            description = "The uploaded image does not match any trained crop classes."
            symptoms = []
            treatment = "N/A"
            prevention = "N/A"
        else:
            info = disease_info.get(predicted_class, {})
            description = info.get("description", "No description available.")
            symptoms = info.get("symptoms", [])
            treatment = info.get("treatment", "No treatment details available.")
            prevention = info.get("prevention", "No prevention details available.")


        return render_template(
            "predict.html",
            prediction=predicted_class,
            description=description,
            symptoms=symptoms,
            treatment=treatment,
            prevention=prevention,
            img_path=filepath
        )


    return render_template("predict.html")


# ---------------- TEXT TO SPEECH ----------------
@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    if "user" not in session:
        return {"error": "Unauthorized"}, 401
    
    data = request.get_json()
    prediction = data.get("prediction", "")
    description = data.get("description", "")
    symptoms = data.get("symptoms", [])
    treatment = data.get("treatment", "")
    prevention = data.get("prevention", "")
    
    # Build the text to be spoken
    text_to_speak = f"Disease Prediction: {prediction}. "
    text_to_speak += f"Description: {description}. "
    
    if symptoms:
        text_to_speak += "Symptoms: " + ", ".join(symptoms) + ". "
    
    text_to_speak += f"Treatment: {treatment}. "
    text_to_speak += f"Prevention: {prevention}."
    
    # Generate audio file
    audio_filename = "prediction_speech.mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    
    try:
        tts = gTTS(text=text_to_speak, lang='en', slow=False)
        tts.save(audio_path)
        return {"success": True, "audio_url": f"/{audio_path}"}
    except Exception as e:
        return {"error": str(e)}, 500


# ---------------- TRANSLATE TEXT ----------------
@app.route("/translate_text", methods=["POST"])
def translate_text():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    target_lang = data.get("target_lang", "hi")  # default Hindi
    prediction = data.get("prediction", "")
    description = data.get("description", "")
    symptoms = data.get("symptoms", [])
    treatment = data.get("treatment", "")
    prevention = data.get("prevention", "")
    
    try:
        # Translate each field using deep-translator
        translated_prediction = GoogleTranslator(source='en', target=target_lang).translate(prediction) if prediction else ""
        translated_description = GoogleTranslator(source='en', target=target_lang).translate(description) if description else ""
        
        translated_symptoms = []
        if symptoms:
            for sym in symptoms:
                try:
                    translated_symptoms.append(GoogleTranslator(source='en', target=target_lang).translate(sym))
                except:
                    translated_symptoms.append(sym)
        
        translated_treatment = GoogleTranslator(source='en', target=target_lang).translate(treatment) if treatment else ""
        translated_prevention = GoogleTranslator(source='en', target=target_lang).translate(prevention) if prevention else ""
        
        return jsonify({
            "success": True,
            "translated": {
                "prediction": translated_prediction,
                "description": translated_description,
                "symptoms": translated_symptoms,
                "treatment": translated_treatment,
                "prevention": translated_prevention
            }
        })
    except Exception as e:
        print(f"Translation Error: {str(e)}")
        return jsonify({"error": "Translation service temporarily unavailable. Please try again."}), 500


# ---------------- TRANSLATED TEXT TO SPEECH ----------------
@app.route("/translated_speech", methods=["POST"])
def translated_speech():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    target_lang = data.get("target_lang", "hi")
    prediction = data.get("prediction", "")
    description = data.get("description", "")
    symptoms = data.get("symptoms", [])
    treatment = data.get("treatment", "")
    prevention = data.get("prevention", "")
    
    # Build text for speech
    text_to_speak = f"{prediction}. "
    text_to_speak += f"{description}. "
    
    if symptoms:
        text_to_speak += " ".join(symptoms) + ". "
    
    text_to_speak += f"{treatment}. "
    text_to_speak += f"{prevention}."
    
    # Map language codes for gTTS
    lang_map = {
        "hi": "hi",      # Hindi
        "bn": "bn",      # Bengali
        "mr": "mr",      # Marathi
        "bho": "hi"      # Bhojpuri (use Hindi as fallback)
    }
    
    speech_lang = lang_map.get(target_lang, "hi")
    
    # Generate audio file
    audio_filename = f"translated_speech_{target_lang}.mp3"
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
    
    try:
        tts = gTTS(text=text_to_speak, lang=speech_lang, slow=False)
        tts.save(audio_path)
        return jsonify({"success": True, "audio_url": f"/{audio_path}"})
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        return jsonify({"error": "Speech generation failed. Please try again."}), 500


# ---------------- EXTRA PAGES ----------------
@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))


@app.route("/blogs")
def blogs():
    return render_template("blogs.html", user=session.get("user"))


@app.route("/contact")
def contact():
    return render_template("contact.html", user=session.get("user"))

# ---------------- FEEDBACK SUBMIT ----------------
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    name = request.form.get("name")
    email = request.form.get("email")
    subject = request.form.get("subject")
    message = request.form.get("message")

    print("\n----- Feedback Received -----")
    print("Name:", name)
    print("Email:", email)
    print("Subject:", subject)
    print("Message:", message)
    print("-----------------------------\n")


    with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([name, email, subject, message])

    flash("Your message has been sent successfully! ✅", "success")
    return redirect(url_for("contact"))


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)





