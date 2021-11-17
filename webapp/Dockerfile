FROM node:14.18.0

WORKDIR /app

RUN apt-get update && apt-get -y install git && rm -rf /var/lib/apt/lists/*

RUN npm install

# COPY client/package.json .
COPY client/package.json .
COPY client/package-lock.json .
RUN npm install


FROM python:3.7.11

RUN apt-get update -y && \
    apt-get install -y python3-pip python-dev

COPY Base-RCNN-FPN.yaml /app/
COPY reqs.txt .

RUN pip install --upgrade pip
RUN pip install -r reqs.txt

RUN apt update; apt install -y libgl1

RUN pip3 install torch torchvision torchaudio
RUN apt-get update && apt-get install -y gcc libglib2.0-0
RUN python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html

ADD server /app/server/


# ENTRYPOINT [ "python" ]

#EXPOSE 5000
CMD [ "python", "app/server/run.py" ]