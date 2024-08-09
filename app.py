from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import os
import whisper
import threading
import time
import queue
from faster_whisper import WhisperModel

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
SRT_FOLDER = 'srt'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SRT_FOLDER'] = SRT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(SRT_FOLDER):
    os.makedirs(SRT_FOLDER)

task_queue = queue.Queue()
task_in_progress = False
task_doing = None

def generate_srt(file_path, srt_path):
    model_size = "large-v3"
    model = WhisperModel(model_size_or_path=model_size, device="cuda", compute_type="float16")
    print("Transcribing audio file:", file_path)
    segments, info = model.transcribe(file_path, beam_size=5,word_timestamps=True,language="zh",vad_filter=True)

    segments = list(filter(lambda x: x.text.strip() != "", segments))

    # delete existing SRT file
    if os.path.exists(srt_path):
        os.remove(srt_path)
    # create new SRT file
    with open(srt_path, "w", encoding='utf-8') as srt_file:
        cnt = 0
        for _, segment in enumerate(segments):
            # print(segment) 
            cnt += 1
            
            text = segment.text.strip()
            if text == "":  # skip empty segments
                continue
            start_time = segment.words[0].start
            end_time = segment.words[-1].end

            # write to SRT file
            srt_file.write(f"{cnt}\n")
            srt_file.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            srt_file.write(f"{text}\n\n")
            
    print("SRT file generated successfully:", srt_path)

def format_srt_time(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    milliseconds = int((t - int(t)) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def process_queue():
    global task_in_progress
    global task_doing
    while True:
        if not task_queue.empty() and not task_in_progress:
            print("Processing task from queue...")
            task_in_progress = True
            mp3_file_path, srt_file_path = task_queue.get()
            print("set task_doing", mp3_file_path)
            task_doing = mp3_file_path
            generate_srt(mp3_file_path, srt_file_path)
            print("remove task_doing")
            task_doing = None
            task_in_progress = False
            task_queue.task_done()
        time.sleep(1)

@app.route('/')
def index():
    srt_files = os.listdir(app.config['SRT_FOLDER'])
    mp3_files = os.listdir(app.config['UPLOAD_FOLDER'])
    queue_items = list(task_queue.queue).copy()
    queue_items = [item[0] for item in queue_items]
    if task_doing is not None:
        queue_items.append(task_doing)
    else:
        print("no task_doing")
    print(queue_items)
    return render_template('index.html', srt_files=srt_files, mp3_files=mp3_files, queue_items=queue_items)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename =  file.filename[:-3].replace(" ", "_") + time.strftime("%H_%M_%S") + ".mp3"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("Saving file to:", file_path)
        file.save(file_path)
        return jsonify({'message': 'File successfully uploaded'})

@app.route('/generate_srt', methods=['POST'])
def generate_srt_route():
    mp3_filename = request.form['mp3_filename']
    mp3_file_path = os.path.join(app.config['UPLOAD_FOLDER'], mp3_filename)
    srt_filename = mp3_filename[:-3] + 'srt'
    srt_file_path = os.path.join(app.config['SRT_FOLDER'], srt_filename)
    task_queue.put((mp3_file_path, srt_file_path))
    return jsonify({'message': 'SRT generation task added to queue', 'srt_filename': srt_filename})

@app.route('/download/<filename>')
def download_file(filename):
    if filename.endswith(".mp3"):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    elif filename.endswith(".srt"):
        return send_from_directory(app.config['SRT_FOLDER'], filename, as_attachment=True)
    else:
        return jsonify({"error": "Invalid file type"})

if __name__ == '__main__':
    threading.Thread(target=process_queue, daemon=True).start()
    app.run(port=9000, host='0.0.0.0')
