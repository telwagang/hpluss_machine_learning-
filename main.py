from json import dumps

from flask import Flask, request
from flask_restful import Api
from flask_restful import reqparse
from AI import train, predict, ROOT_DIR, write_csv
from Data import day,specity,location,time


app = Flask(__name__)
api = Api(app)


@app.route('/predict/<string:days>/<string:place>/<string:specialty>/<string:times>', methods=['GET'])
def predictfn(days,place,times,specialty):

    print(days)
    x = day(days)
    y = location(place)
    z = time(times)
    s = specity(specialty)

    test = [s, x, z, 21.0, y]

    model = train('load')
    data = predict(model, test)
    ans = str(data[0])
    print(ans)
    return dumps({'data': ans})


@app.route('/train', methods=['GET'])
def trianfn():
    data = request.data
    write_csv(data)
    train('train')
    return dumps({'data': 'done'})


if __name__ == '__main__':
    app.run(port=5400, debug='true')
