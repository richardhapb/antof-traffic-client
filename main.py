from dashboard import index

# This file is only for using in development, in production
# the server variable from `dashboard/index.py` directly
if __name__ == "__main__":
    index.app.run_server(debug=True, host="0.0.0.0", port=8050)
