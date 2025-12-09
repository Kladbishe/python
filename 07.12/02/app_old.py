from flask_cors import CORS
from flask import Flask, jsonify, request
from flask_mysqldb import MySQL

# entry point
app = Flask(__name__)
CORS(app)

# MySQL config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'password'
app.config['MYSQL_DB'] = 'shop'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # so we get dicts instead of tuples

mysql = MySQL(app)

# routes
@app.route("/products", methods=["GET"])
def get_products():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM products")
    products = cur.fetchall()
    cur.close()
    return jsonify(products)  # list of dicts -> JSON

