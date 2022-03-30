from crypt import methods
from flask import Flask, request, render_template, jsonify
import io
from flask_cors import CORS
from predict import *

app = Flask(__name__)
CORS(app)

@app.route("/predict_img", methods=["GET","POST"])
def predict_label():
    if request.method == "GET":
        return render_template("home.html",value = "Image")
    if render_template.method == "POST":
        if "file" not in request.files:
            return "Image not uploaded"
        
        file = request.file["file"].read()
        
        try:
            img = Image.open(io.BytesIO(file))
        except IOError:
            return jsonify(predictions = "Not an Image, please upload file again!")
        
        img = img.convert("RGB")
        
        label = predict(img)
        
        return jsonify(predictions = label)
    
if __name == "__main__":
    app.run(debug = True)
    
    
