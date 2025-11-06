"""
Lightweight compatibility entrypoint.

This root-level `app.py` delegates to `src.app.init_app()` so there's a
single canonical Flask application implementation (with fallback model
logic and proper template/static dirs). This keeps behavior consistent
whether the app is started via this file or the `src` package.
"""
from src.app import init_app
from src.config import API_CONFIG


app = init_app()


if __name__ == "__main__":
    app.run(host=API_CONFIG['host'], port=API_CONFIG['port'], debug=API_CONFIG['debug'])