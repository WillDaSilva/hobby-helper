from flask import Flask, request
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello_world():
  if request.method == "POST":
    print(request.values)
    #call_the_nn(request.values)
    return "Hi."
  else:
    return "You should be posting here!"
