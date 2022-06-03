import concurrent.futures
import logging
from threading import Thread
from time import time

from flask import Flask, flash, request, redirect, jsonify
from comprehend_clasifier import ComprehendDetect, LanguageEnum

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

comprehend_classifier = ComprehendDetect()

MAX_THREADS = 10


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect(file_content):
    languages = comprehend_classifier.detect_languages(file_content)
    lang_code = languages[0]['LanguageCode']

    functions = [
        comprehend_classifier.detect_entities,
        comprehend_classifier.detect_key_phrases,
        comprehend_classifier.detect_sentiment,
        comprehend_classifier.detect_syntax,
    ]
    demo_size = 5
    results = []
    for i in functions:
        language = getattr(LanguageEnum, lang_code).value
        thread = Thread(
            target=i,
            args=(file_content, language, results, demo_size)
        )
        thread.start()
        thread.join()
    return results


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        file_content = file.read().decode('utf-8')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):

            results = detect(file_content)

            return jsonify(results)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def async_detect(file_text: str) -> list:
    return detect(file_text)


def thread_pools_task(file_list: list) -> list:
    max_threads = min(len(file_list), MAX_THREADS)
    logger.info(f'Max threads: {max_threads} for {len(file_list)} files')

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for f in file_list:
            file_content = f.read().decode('utf-8')
            result = executor.submit(async_detect, file_content)
            futures.append(result)
        for future in concurrent.futures.as_completed(futures):
            results = [i for i in future.result()]

    logger.info(f'Threads finished with {len(results)} results')
    return results


@app.route("/data", methods=['GET', 'POST'])
async def get_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'error': 'No file part'})

        file_list = request.files.getlist("file")
        init_time = time()
        results = thread_pools_task(file_list)
        results.append({'total_time': round(time() - init_time, 4)})
        return jsonify(results)

    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h4>Upload Single File</h4>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        
        <h4>Upload Multiple Files</h4>
          <form action = "http://localhost:5001/data" method = "POST" 
             enctype = "multipart/form-data">
             <input type = "file" name = "file" multiple/>
             <input type = "submit"/>
          </form>
        '''

if __name__ == '__main__':
    app.env = 'development'
    app.secret_key = 'badasses123'
    app.run(debug=True, port=5001, threaded=True)
