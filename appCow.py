from flask import Flask, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)


@app.route("/detect", methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify(error="No image received.")

    file = request.files.get('file')
    img_bytes = file.read()
    img_path = "./upload_images/test.jpg"

    with open(img_path, "wb") as img:
        img.write(img_bytes)

    #processar a imagem
    model = YOLO("./best.pt")
    results = model(img_path)
    result = results[0]
    resp = []

    for box in result.boxes:
        if round(box.conf[0].item(), 2) > 0.5:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            resp.append([class_id, cords])

    return jsonify(predict=resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
