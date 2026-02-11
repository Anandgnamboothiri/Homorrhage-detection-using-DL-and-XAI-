"""
Brain Hemorrhage Detection - Flask Web App
Binary classification: Hemorrhagic vs NORMAL (JPG images)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, url_for
import keras
import cv2

from magicsauce import MODEL_PATH, INPUT_SHAPE, preprocess_jpg
from gradcam import generate_gradcam

app = Flask(__name__, static_url_path='/static')

# Load model if it exists
model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    try:
        model._make_predict_function()  # TF1 compatibility
    except AttributeError:
        pass  # TF2+ doesn't need this
    print(f"Loaded model from {MODEL_PATH}")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}. Run 'python train.py' first.")


@app.route("/")
def file_front_page():
    return render_template('base.html')


@app.route("/handleUpload", methods=['POST'])
def handle_file_upload():
    if model is None:
        return render_template('index.html', error="Model not loaded. Run 'python train.py' to train and save the model first.")

    if 'brainscan' not in request.files:
        return render_template('index.html', error="No file selected.")

    file = request.files['brainscan']
    if file.filename == '' or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return render_template('index.html', error="Please upload a JPG or PNG image.")

    try:
        # Read image from uploaded file
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_array is None:
            return render_template('index.html', error="Could not decode image.")

        # Directory for saving input and display images
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        images_dir = os.path.join(static_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Base name for files derived from upload filename
        safe_base = os.path.basename(file.filename).replace(' ', '_').rsplit('.', 1)[0]

        # Save a copy of the original uploaded image (BGR) for Grad-CAM
        original_input_path = os.path.join(images_dir, f'input_{safe_base}.jpg')
        cv2.imwrite(original_input_path, img_array)

        # Preprocess and predict
        tensor = preprocess_jpg(img_array)
        prediction = model.predict(tensor)[0][0]  # Binary: probability of Hemorrhagic

        # Save uploaded image for display (convert BGR to RGB for display)
        display_filename = f'upload_{safe_base}.png'
        display_path = os.path.join(images_dir, display_filename)
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(display_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Generate Grad-CAM heatmap overlay
        gradcam_output_path = os.path.join(static_dir, 'gradcam_output.jpg')
        gradcam_url = None
        try:
            print(f"Grad-CAM: Starting generation for {original_input_path}")
            print(f"Grad-CAM: Output will be saved to {gradcam_output_path}")
            result_path = generate_gradcam(model, original_input_path, gradcam_output_path)
            if result_path is not None:
                gradcam_url = url_for('static', filename='gradcam_output.jpg')
                print(f"Grad-CAM: Successfully generated heatmap at {result_path}")
            else:
                print("Grad-CAM: generate_gradcam returned None (check logs above)")
        except Exception as e:
            # Log error server-side but do not break existing inference pipeline
            print(f"Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()

        return render_template(
            'index.html',
            prediction=prediction,
            url=url_for('static', filename=f'images/{display_filename}'),
            filename=display_filename,
            gradcam_url=gradcam_url
        )
    except Exception as e:
        return render_template('index.html', error=f"Error processing image: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, port=9999)
