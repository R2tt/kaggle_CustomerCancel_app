# from flask import Flask, render_template
# app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     return render_template('index.html')
# 
# if __name__ == "__main__":
#     app.run()

import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import pickle

classes = ["0","1"]

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)#学習済みモデルをロード

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取ったファイルを読み込み、np形式に変換
            # img = image.load_img(filepath, grayscale=True, target_size=(image_size,image_size))
            # img = image.img_to_array(img)
            # data = np.array([img])
            data = pd.read_csv(filepath)
            #変換したデータをモデルに渡して予測する
            data["product_prd_1"] = [1 if s=="prd_1" else 0 for s in data["product"]]
            data["product_prd_2"] = [1 if s=="prd_2" else 0 for s in data["product"]]
            result = model.predict_proba(data[["age","usage_period","product_prd_1","product_prd_2"]])
            #predicted = result.argmax()
            #pred_answer = classes[predicted]
            return render_template("index.html",answer=result[:,1])

    return render_template("index.html",answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
