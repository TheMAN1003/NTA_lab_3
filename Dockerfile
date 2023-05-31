FROM python:3.9-slim-buster

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir sympy
RUN pip install --no-cache-dir numpy

CMD ["python", "./main.py"]