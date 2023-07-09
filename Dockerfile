FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt update || true

# RUN #apt-get install python3.9 -y
RUN apt-get install python3.9-venv -y
#RUN python3.9 -m venv venv
RUN apt-get install python3-pip -y
#RUN source venv/bin/activate
#RUN pip --version

#RUN #apt install -y \
#        python3.9 \
#        python3.9-distutils \
#        python3.9-venv && \
#    python3.9 -m ensurepip #&& \
#    python3.9 --version && \
#    pip3.9 --version

#RUN alias pip=pip3
#RUN alias python=python3

WORKDIR /hlcv/
COPY requirements.txt /hlcv/requirements.txt
COPY download_thumbnails.py /hlcv/download_thumbnails.py

RUN python3.9 -m pip install -r /hlcv/requirements.txt

WORKDIR /hlcv/
ENTRYPOINT ["python3.9", "download_thumbnails.py"]