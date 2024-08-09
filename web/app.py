from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
from model import process_image_with_yolo, process_image_with_mask_rcnn, process_video_with_yolo, process_video_with_mask_rcnn

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'static', 'uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        yolo_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_yolo_{filename}')
        maskrcnn_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_maskrcnn_{filename}')

        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            yolo_result_path, yolo_inference_time = process_video_with_yolo(file_path, yolo_output_path)
            maskrcnn_result_path, maskrcnn_inference_time = process_video_with_mask_rcnn(file_path, maskrcnn_output_path)
        else:
            yolo_result_path, yolo_inference_time = process_image_with_yolo(file_path)
            maskrcnn_result_path, maskrcnn_inference_time = process_image_with_mask_rcnn(file_path)

        if yolo_inference_time and maskrcnn_inference_time:
            yolo_ratio = yolo_inference_time / maskrcnn_inference_time
            maskrcnn_ratio = maskrcnn_inference_time / yolo_inference_time
        else:
            yolo_ratio = None
            maskrcnn_ratio = None

        print(yolo_result_path, "<-- yolo result path")
        print(file_path, "<-- file path")
        print(maskrcnn_result_path, "<-- maskrcnn")
        return render_template('result.html', original_image_path=os.path.relpath(file_path, 'static'),
                               yolo_result_path=yolo_result_path,
                               maskrcnn_result_path=maskrcnn_result_path,
                               yolo_inference_time=yolo_inference_time,
                               maskrcnn_inference_time=maskrcnn_inference_time,
                               yolo_ratio = yolo_ratio,
                               maskrcnn_ratio = maskrcnn_ratio)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))


@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(video_path, mimetype='video/mp4', as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True)
