import os
import cv2
import time
import pickle
import numpy as np
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
from flask import request, redirect
from keras.models import load_model
from keras.preprocessing import image
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/pictures'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
db = SQLAlchemy(app)


class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(512), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow())

    def __repr__(self) -> str:
        return '<Task %r>' % self.id


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')


@app.route('/houseprice', methods=['GET', 'POST'])
def houseprice():
    return render_template('houseprice.html')


@app.route('/detectemotion', methods=['GET', 'POST'])
def detectemotion():
    return render_template('detectemotion.html')


@app.route('/upcoming')
def upcoming():
    return render_template('upcoming.html')


@app.route('/todo', methods=['GET', 'POST'])
def todo():
    if request.method == 'POST':
        task_content = request.form['content']
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/todo')
        except:
            return 'Unfortunately your operation was unsuccessful.'
    else:
        tasks = Todo.query.order_by(Todo.date).all()
        return render_template('/todo.html', tasks=tasks)


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/todo')
    except:
        return 'There was a problem deleting that task'


@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        try:
            db.session.commit()
            return redirect('/todo')
        except:
            return 'There was an issue updating your task'

    else:
        return render_template('update.html', task=task)


def get_processed_data(arr):
    lst = list()
    lst.append(int(arr[0]))
    lst.append(int(arr[1]))
    lst.append(int(arr[2]))
    lst.append(1) if arr[3] == "C (all)" else lst.append(0)
    lst.append(1) if arr[3] == "FV" else lst.append(0)
    lst.append(1) if arr[3] == "RH" else lst.append(0)
    lst.append(1) if arr[3] == "RL" else lst.append(0)
    lst.append(1) if arr[3] == "RM" else lst.append(0)
    lst.append(1) if arr[4] == "Grvl" else lst.append(0)
    lst.append(1) if arr[4] == "Pave" else lst.append(0)
    lst.append(1) if arr[5] == "IR1" else lst.append(0)
    lst.append(1) if arr[5] == "IR2" else lst.append(0)
    lst.append(1) if arr[5] == "IR3" else lst.append(0)
    lst.append(1) if arr[5] == "Reg" else lst.append(0)
    lst.append(1) if arr[6] == "Bnk" else lst.append(0)
    lst.append(1) if arr[6] == "HLS" else lst.append(0)
    lst.append(1) if arr[6] == "Low" else lst.append(0)
    lst.append(1) if arr[6] == "Lvl" else lst.append(0)
    lst.append(1) if arr[7] == "AllPub" else lst.append(0)
    lst.append(1) if arr[7] == "NoSeWa" else lst.append(0)
    lst.append(1) if arr[8] == "Abnorml" else lst.append(0)
    lst.append(1) if arr[8] == "AdjLand" else lst.append(0)
    lst.append(1) if arr[8] == "Alloca" else lst.append(0)
    lst.append(1) if arr[8] == "Family" else lst.append(0)
    lst.append(1) if arr[8] == "Normal" else lst.append(0)
    lst.append(1) if arr[8] == "Partial" else lst.append(0)
    return lst


@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    model = pickle.load(open('./models/model1.pkl', 'rb'))
    scale = pickle.load(open('./models/scale.pkl', 'rb'))
    features = [x for x in request.form.values()]
    order = [2, 3, 0, 1, 4, 6, 7, 5, 8]
    features = [features[i] for i in order]
    x = features
    features = get_processed_data(features)
    final_features = np.array([features])
    scalled_X = final_features  # scale.transform(final_features)
    print(scalled_X)
    prediction = model.predict(scalled_X)
    print(prediction)
    return render_template('price.html', Predicted_price="{:.2f} INR".format(prediction[0][0]))


@app.route('/detect_emtion', methods=['POST'])
def detect_emtion():
    # Save image from user
    file1 = request.files['img-up']
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
    file1.save(path)

    face_classifier = cv2.CascadeClassifier('./models/haarcascade.xml')
    classifier = load_model('./models/model.h5')
    class_labels = ['Angry', 'Disgust', 'Fear',
                    'Happy', 'Neutral', 'Sad', 'Surprise']
    cap = cv2.imread('./static/pictures/img.jpg')

    # Grab a single frame of video
    frame = cap
    labels = []
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    label = None
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            print(label)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    milliseconds = int(round(time.time() * 1000))
    img_path = "./static/pictures/img{}.jpg".format(milliseconds)
    cv2.imwrite(img_path, cap)
    #
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    resized_image = cv2.resize(
        image, (3*width, 3*height), interpolation=cv2.INTER_CUBIC)
    print(label)
    return render_template('detected.html', image=img_path)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html', title='404'), 404


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
