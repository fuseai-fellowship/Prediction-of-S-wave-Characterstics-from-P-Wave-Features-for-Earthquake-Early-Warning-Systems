from flask import Flask, send_from_directory, render_template
app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:p>')
def static_file(p):
    return send_from_directory('static', p)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
