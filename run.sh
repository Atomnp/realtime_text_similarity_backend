#! /bin/sh
export FLASK_APP=app
export FLASK_ENV=development
export BUILD=prod #dev, prod, staging
flask run --no-debugger
