from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('prev.html')


@app.route('/process_drawing', methods=['POST'])
def process_drawing():
    drawing_data = request.form.get('drawing')

    # Process the drawing data as needed (e.g., save to a file, analyze, etc.)
    # Example: save the image to a file
    with open('drawing.png', 'wb') as f:
        f.write(drawing_data.decode('base64'))

    return "Drawing processed successfully"


if __name__ == '__main__':
    app.run(debug=True)
