
FROM python:3.12-slim

WORKDIR /app


COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y gcc g++
RUN apt-get install -y libgdal-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt install netcat-traditional

COPY . /app
RUN mkdir data

ENTRYPOINT [ "bash", "/app/init.sh" ]
EXPOSE 8050
CMD ["gunicorn", "--workers", "2", "--chdir", "/app/waze", "--bind", "0.0.0.0:8050", "index:server"]
