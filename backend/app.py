from flask import Flask, render_template, redirect, url_for, request, send_file, jsonify
import os
from PIL import Image, ImageChops
from io import BytesIO
import requests
import numpy as np
import sys
import os
import cv2
import time
from datetime import datetime
import base64


def show_fullscreen(image_pil):
    image_np = np.array(image_pil.convert("RGB"))  # Convert PIL to numpy array
    image_cv = cv2.cvtColor(
        image_np, cv2.COLOR_RGB2BGR
    )  # Convert RGB to BGR for OpenCV

    cv2.namedWindow("Sketch", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Sketch", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Sketch", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def is_black_image(image):
    # Compare with black image of the same size
    black = Image.new("RGB", image.size)
    diff = ImageChops.difference(image, black)
    return not diff.getbbox()  # Returns None if the image is completely black


# This gets the directory where app.py is located
base_dir = os.path.abspath(os.path.dirname(__file__))

# Add the root directory to the Python path
sys.path.append(base_dir)
model_path = os.path.join(
    base_dir, "model", "pytorch_CycleGAN_and_pix2pix", "pytorch_CycleGAN_and_pix2pix"
)
sys.path.append(model_path)

from model.pytorch_CycleGAN_and_pix2pix.pixx.script import img_to_sketch

# from mysqldb import mysql

app = Flask(
    __name__,
    template_folder=os.path.join(os.pardir, "frontend"),
    static_folder=os.path.join(os.pardir, "frontend", "static"),
)

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "fruity"
app.config["MYSQL_DB"] = "tracefy"

# mysql.init_app(app)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


REMOTE_BACKEND_URL = "https://470e-219-122-229-5.ngrok-free.app/infer"


# Home route
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        return render_template("prompt.html")

    return render_template("index.html")

    #     cursor = mysql.connection.cursor()
    #     query = "SELECT * FROM accounts WHERE username=%s AND password_=%s"
    #     cursor.execute(query, (username, password))
    #     user = cursor.fetchone()
    #     cursor.close()

    #     if user:
    #         return render_template("prompt.html")
    #     else:
    #         return "Invalid username or password."
    # return render_template("index.html")


@app.route("/prompt", methods=["GET", "POST"])
def prompt():
    return render_template("prompt.html")


@app.route("/save_prompt", methods=["POST"])
def save_prompt():
    img = request.files.get("imageUpload")
    prompt_text = request.form.get("promptText")
    model_type = request.form.get("modelType", "pix2pix + flux")  # Default to 'flux'

    image_pil = Image.open(img.stream).convert("RGB")
    sketch = img_to_sketch(image_pil)
    sketch.show()

    if model_type == "pix2pix + flux":
        # Use pix2pix model to generate a sketch
        sketch = img_to_sketch(image_pil)

        # result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.png")
        # sketch.save(result_image_path)

        buffer = BytesIO()
        sketch.save(buffer, format="PNG")
        buffer.seek(0)
        control_image = buffer
    else:
        # Use the original image as the control image for flux
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")
        buffer.seek(0)
        control_image = buffer

    payload = {
        "prompt": prompt_text,
        "controlStrength": 0.7,  # Example value, adjust as needed
        "controlGuidanceEnd": 0.4,  # Example value, adjust as needed
        "seed": 42,  # Example value, adjust as needed
        "randomizeSeed": "true",
    }
    files = {"controlImage": ("controlImage.png", control_image, "image/png")}

    # Send request to the remote app.py backend
    response = requests.post(REMOTE_BACKEND_URL, data=payload, files=files)

    if response.status_code == 200:
        # Parse the JSON response with two images
        json_data = response.json()
        result_image_data = json_data.get("result_image")
        control_image_data = json_data.get("control_image")

        # Decode and save result image
        result_image_bytes = result_image_data.encode("latin1")
        result_image = Image.open(BytesIO(result_image_bytes))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Create unique filename with timestamp
        filename = f"output_{timestamp}.png"
        # filename = "output_20250520_102603.png"
        result_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        result_image.save(result_image_path)    
        with open(result_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        result_image_data_uri = f"data:image/png;base64,{encoded_string}"

    # result_image_url = url_for('static', filename='output.png')

    # Decode and save control image
    # control_image_bytes = control_image_data.encode("latin1")
    # control_image = Image.open(BytesIO(control_image_bytes))
    # control_image_path = os.path.join(app.config['UPLOAD_FOLDER'], "control_output.png")
    # control_image.save(control_image_path)

    # Only pass output.png to dashboard.html
    return render_template("dashboard.html", result_image_url=result_image_data_uri)
    # else:
    #     return jsonify({"error": "Failed to generate image"}), 500


# Dashboard route (protected)
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# Register route
@app.route("/register")
def register():
    return "Register Page"


if __name__ == "__main__":
    app.run(debug=True)

    # if response.status_code == 200:
    # Save the generated image from the Colab backend
    # print("Response headers:", response.headers)
    # print("Response content-type:", response.headers.get("Content-Type"))
    # print("Response length:", len(response.content))
    # try:
    #     json_data = response.json()
    #     print("Colab JSON response:", json_data)
    # except Exception as e:
    #     print("Failed to parse JSON:", e)
