from flask import Flask, render_template, redirect, url_for, request
import requests
from forms import SearchForm
import json
import unicodedata

app = Flask(__name__)
app.config['SECRET_KEY'] = '3141592653589793238462643383279502884197169399'


@app.route('/', methods=['GET', 'POST'])
def home():
    form = SearchForm()
    print(request.method)
    if request.method == 'POST' and form.is_submitted():
        form_data = {
            'genre': form.genre.data,
            'type': form.type.data,
            'episode': form.episode.data,
            'members': form.members.data
        }
        data = json.dumps(form_data)
        print("data ", data)
        return redirect(url_for('predict', message=data))
    return render_template('form.html', form=form)


@app.route('/predict')
def predict():
    print(json.loads(request.args['message']))
    data = json.loads(request.args['message'])
    info = requests.get('http://python-api:7777/project1/api/v1.0/predict', params=data)
    print(info.url)
    info = unicodedata.normalize('NFKD', info.text).encode('ascii','ignore')
    info = json.loads(info)
    print(info)
    return render_template('result.html', info=info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
