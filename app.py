from automatminer_web.index import app

server = app.server

if __name__ == '__main__':
    print("starting...")
    app.run_server(debug=True)
