from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'Flask backend is running.'

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()

    if user_message == 'hi':
        response = 'hello'
    else:
        response = "I didn't understand that."

    return jsonify({'reply': response})

if __name__ == '__main__':
    app.run(debug=True)
