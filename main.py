from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = load_model('./static/models/leaves_model.keras')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)

            img = np.array(Image.open(image).resize((32, 32)).convert('RGB'))
            img = img / 255

            img = img[np.newaxis, :]

            start = time.time()

            pred = model.predict(img)

            # Prediction Time
            runtimes = round(time.time() - start, 4)

            if pred <= 0.5:
                result = "Kemangi"
            else:
                result = "Seledri"


            return render_template('/index.html', label=result,
                                   run_time=runtimes, img=img, uploaded_image=image.filename)

    return render_template('/index.html')


@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2000)