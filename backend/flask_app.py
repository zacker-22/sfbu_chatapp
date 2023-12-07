from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

from script import ChatGpt
import threading
import json
chat = ChatGpt()
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def home():
    if chat.state == "LOADING":
        return render_template('./loading.html')

    args = request.args
    
        
    
    if 'query' in args:
        query = args['query']
        result = chat.answer_question(question=query)
        return render_template('./index.html', query = query, result = result)
    return render_template('./index.html')



@app.route('/api')
@cross_origin()
def api():
    args = request.args
    print(args)
    print("here")

    # return jsonify({'result': 'Hello'})
    chat_history = []
    if 'chat_history' in args:
        chat_history = json.loads(args['chat_history'])
    
    print((chat_history))
    if 'query' in args:
        query = args['query']
        result = chat.answer_question(question=query, chat_history=chat_history)
        print(jsonify({'query': query, 'result': result}))
        print(result)
        return jsonify({'query': query, 'result': result})
        
    
    return jsonify({'query': '', 'result': ''})


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.run(debug=True, port=5000)