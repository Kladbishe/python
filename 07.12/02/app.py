import json
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'
app.config['SECRET_KEY'] = "random string"


db = SQLAlchemy(app)

# model
class students(db.Model):
    id = db.Column('student_id', db.Integer, primary_key = True)
    name = db.Column(db.String(100))
    city = db.Column(db.String(50))
    addr = db.Column(db.String(200))
    pin = db.Column(db.String(10))


    def __init__(self, name, city, addr,pin):
        self.name = name
        self.city = city
        self.addr = addr
        self.pin = pin
# model

# views
@app.route('/')
def show_all():
    res=[]
    for student in students.query.all():
        res.append({"addr":student.addr,"city":student.city,"id":student.id,"name":student.name,"pin":student.pin})
    return  (json.dumps(res))
   
@app.before_request
def create_tables():
    db.create_all()

@app.route('/new', methods = ['GET', 'POST'])
def new():
    request_data = request.get_json()
    # print(request_data['city'])
    city = request_data['city']
    name= request_data["name"]
    addr= request_data["addr"]
    pin= request_data["pin"]


    newStudent= students(name,city,addr,pin)
    db.session.add (newStudent)
    db.session.commit()
    return "a new rcord was create"


if __name__ == '__main__':
    with app.app_context():
            db.create_all()
    app.run(debug = True)

