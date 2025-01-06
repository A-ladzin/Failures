FROM python:3.10-slim
WORKDIR /app
VOLUME /app/data
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3","/app/make_submission.py"]

COPY . /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libglib2.0-0 -y && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    chmod +x /app/entrypoint.sh /app/baseline.py /app/make_submission.py
