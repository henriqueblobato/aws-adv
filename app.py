import concurrent.futures
import logging
import os
from threading import Thread
from time import time

from flask import Flask, flash, request, redirect, jsonify
from comprehend_clasifier import ComprehendDetect, LanguageEnum

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
MAX_THREADS = 10

logger = logging.getLogger(__name__)
app = Flask(__name__)
comprehend_classifier = ComprehendDetect()


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
        comprehend_classifier.detect_pii,
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
def main():
    return redirect('/data')


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
        <h5>Upload Multiple Files</h5>
          <form action = "/data" method = "POST" 
             enctype = "multipart/form-data">
             <input type = "file" name = "file" multiple/>
             <input type = "submit"/>
          </form>
          <footer
            style="position: absolute; bottom: 0; width: 100%; text-align: center; font-size: 12px;"> 
          Incredible frontend designed by a really experienced front end developer 
        </html>
        '''

