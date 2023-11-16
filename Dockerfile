FROM python:3.9

# Install dependencies
RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install python-multipart


# Set working directory to previously added app directory

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


# Copy files
COPY detection/config ./detection/config/
COPY detection/utils ./detection/utils/
COPY deploy.py .
COPY detection/detect_images.py ./detection/detect_images.py
COPY detection/detect_video.py ./detection/detect_video.py
COPY detection/ssd/ ./detection/ssd/

RUN mkdir -p data


EXPOSE 8000
CMD ["uvicorn", "deploy:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]