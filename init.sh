#!/bin/bash
set -e

MARKER_FILE=/app/.initialized

if [ ! -f "$MARKER_FILE" ]; then

    echo "Waiting for MLflow host in port 8080"
    while ! nc -z mlflow 8080; do
        sleep 2
    done

    echo "MLflow listening in port 8080, training the model..."
    python dashboard/train.py

    echo "Training ok, ready for init the server"

    touch "$MARKER_FILE"
else
    echo "Model trained previously, skiping training."
fi

exec "$@"

