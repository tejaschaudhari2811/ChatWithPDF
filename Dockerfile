FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip -r /app/requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
