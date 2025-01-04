from dashboard import index

if __name__ == "__main__":
    index.app.run_server(debug=True, host="0.0.0.0", port=8050)
