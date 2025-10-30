###-----IMPORTS-----###
from flask import Flask, request, jsonify, render_template_string, send_from_directory, url_for
from ultralytics import YOLO
import os
import cv2
import numpy as np
from grouping import process_image_and_group
import json
from datetime import datetime

###-----FLASK APP SETUP-----###
app = Flask(__name__)

###-----BASE DIRECTORY SETUP-----###
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'ModelTraining/runs/detect/train/weights/best.pt')
model = YOLO(model_path)

upload_folder = os.path.join(base_dir, 'uploads')
results_folder = os.path.join(base_dir, 'Results')
json_folder = os.path.join(results_folder, 'JSON')
images_folder = os.path.join(results_folder, 'Images')

for folder in [upload_folder, results_folder, json_folder, images_folder]:
    os.makedirs(folder, exist_ok=True)

###-----HTML TEMPLATE-----###
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Smart Shelf Detection</title>
    <style>
        body { font-family: Arial; text-align: center; background: #f4f4f4; }
        .container { margin-top: 50px; }
        .btn { padding: 10px 20px; font-size: 18px; margin-top: 15px; cursor: pointer; }
        img { margin-top: 20px; border: 3px solid #444; max-width: 90%; }
        .json-link { margin-top: 20px; display: block; font-size: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛒 Smart Shelf AI System</h1>

        <form method="POST" action="/upload" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <br>
            <button type="submit" class="btn">Analyze Shelf</button>
        </form>

        {% if image_url %}
        <h2>Results</h2>
        <img src="{{ image_url }}" alt="Processed Image">
        <a class="json-link" href="{{ json_url }}" download>Download JSON Results</a>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    image_path = os.path.join(upload_folder, filename)
    file.save(image_path)

    results = model(image_path)

    detections = []
    cropped_images = []

    image = cv2.imread(image_path)

    for i, result in enumerate(results[0].boxes.xyxy):
        x_min, y_min, x_max, y_max = map(int, result)
        detections.append({'xmin': float(x_min), 'ymin': float(y_min), 'xmax': float(x_max), 'ymax': float(y_max)})

        cropped_img = image[y_min:y_max, x_min:x_max]
        crop_path = os.path.join(upload_folder, f'crop_{timestamp}_{i}.jpg')
        cv2.imwrite(crop_path, cropped_img)
        cropped_images.append(crop_path)

    group_ids = process_image_and_group(cropped_images)
    group_ids_list = group_ids.tolist()

    for i, det in enumerate(detections):
        det['group_id'] = int(group_ids_list[i])

    response_data = {"detections": detections, "groups": group_ids_list}

    json_filename = f"results_{timestamp}.json"
    json_path = os.path.join(json_folder, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(response_data, f, indent=2)

    ### ✅ SAME COLOR FOR SAME GROUP ###
    unique_groups = sorted(set(group_ids_list))
    group_colors = {gid: np.random.randint(0, 255, size=3).tolist() for gid in unique_groups}

    for det in detections:
        x_min, y_min, x_max, y_max = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        group_id = det['group_id']
        color = group_colors[group_id]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(image, f"Group {group_id}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    vis_filename = f"visualization_{timestamp}.jpg"
    vis_path = os.path.join(images_folder, vis_filename)
    cv2.imwrite(vis_path, image)

    image_url = url_for('serve_image', filename=vis_filename)
    json_url = "/Results/JSON/" + json_filename

    return render_template_string(HTML_TEMPLATE, image_url=image_url, json_url=json_url)

@app.route('/Results/Images/<filename>')
def serve_image(filename):
    return send_from_directory(images_folder, filename)

@app.route('/Results/JSON/<filename>')
def serve_json(filename):
    return send_from_directory(json_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
