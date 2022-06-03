import os

from app import app

if __name__ == "__main__":
    app.run(
        port=os.environ.get('PORT', 5000),
        debug=os.environ.get('DEBUG', False),
    )
