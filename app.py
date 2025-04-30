import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from database import init_db, register_user, authenticate_user

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the trained model
model = tf.keras.models.load_model("nail_disease_cnn_model.h5")

# Define image size (same as training)
IMG_SIZE = 128  

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Get class labels
train_path = "dataset/nail_images/nail/train"
labels = sorted([label for label in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, label))])

# Define symptom details (deficiencies, causes, food recommendations)
symptom_info = {
    "beau-s-line": {
        "deficiencies": ["Zinc deficiency", "Protein deficiency"],
        "reason": "Caused by severe illness, malnutrition, or trauma.",
        "foods": ["Lean meats", "Eggs", "Dairy", "Nuts", "Whole grains"]
    },
    "black line": {
        "deficiencies": ["Iron deficiency", "Vitamin B12 deficiency"],
        "reason": "Possible trauma, endocarditis, or vitamin deficiency.",
        "foods": ["Spinach", "Red meat", "Lentils", "Fish"]
    },
    "clubbing": {
        "deficiencies": ["Vitamin C deficiency"],
        "reason": "Linked to chronic lung or heart disease.",
        "foods": ["Citrus fruits", "Bell peppers", "Strawberries", "Leafy greens"]
    },
    "normal": {
        "deficiencies": ["No deficiency detected"],
        "reason": "Healthy nails, no concerns.",
        "foods": ["Balanced diet with vitamins and minerals"]
    }
}

# Function to predict disease
def predict_nail_disease(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown", 0.0, ["Unknown"], "Unknown", ["Unknown"]

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    confidence = np.max(prediction) * 100
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    details = symptom_info.get(predicted_label, {"deficiencies": ["Unknown"], "reason": "Unknown", "foods": ["Unknown"]})
    return predicted_label, confidence, details["deficiencies"], details["reason"], details["foods"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected!")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        prediction, confidence, deficiencies, reason, recommended_foods = predict_nail_disease(filepath)

        return render_template("results.html", prediction=prediction, confidence=confidence, deficiencies=deficiencies,
                               reason=reason, recommended_foods=recommended_foods, image_path=filepath)
    return render_template("index.html")

# User Authentication Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if authenticate_user(username, password):
            session["user"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        register_user(username, password)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
