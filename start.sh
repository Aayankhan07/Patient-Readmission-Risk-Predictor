#!/bin/bash

# Start the FastAPI backend on port 8000 in the background
echo "Starting FastAPI backend on port 8000..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend on the port assigned by Render (defaults to 8501 if unset)
PORT=${PORT:-8501}
echo "Starting Streamlit dashboard on port $PORT..."
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
