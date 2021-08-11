FROM python:3.7.5-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
      gcc \
      g++

ADD requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--port=8888", "--ip=0.0.0.0"]
