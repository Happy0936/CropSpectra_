from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import bcrypt
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import json
import pickle
import csv
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


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
        img_array_exp = np.expand_dims(img_array, axis=0) / 255.0
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
    app.run(debug=True)






