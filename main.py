from flask import Flask, jsonify, render_template
from generate_name import name_samples

app = Flask(__name__)

@app.route('/generate-names', methods=['GET'])
def generate_names():
    name_list = name_samples()
    return jsonify(names=name_list)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)