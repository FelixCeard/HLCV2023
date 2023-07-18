FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt update || true

RUN apt-get install python3.9-venv -y
RUN apt-get install python3-pip -y

WORKDIR /home/hlcv_team017/HLCV2023/

COPY ./striped_lens /home/hlcv_team017/HLCV2023/striped_lens

COPY requirements.txt /home/hlcv_team017/HLCV2023/requirements.txt
COPY test_striped_lens.py /home/hlcv_team017/HLCV2023/test_striped_lens.py
COPY test_stripped_lens_long.py /home/hlcv_team017/HLCV2023/test_stripped_lens_long.py

RUN mkdir /home/hlcv_team017/HLCV2023/data
RUN mkdir /home/hlcv_team017/HLCV2023/data/thumbnails

COPY run.sh /home/hlcv_team017/HLCV2023/run.sh

RUN useradd -u 8877 hlcv_team017

RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/run.sh
RUN chmod 777 /home/hlcv_team017/HLCV2023/run.sh
RUN chmod ugo+rwx /home/hlcv_team017
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/striped_lens
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/data/thumbnails
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/data
# RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/LENS

USER hlcv_team017

ENTRYPOINT ["/bin/sh", "/home/hlcv_team017/HLCV2023/run.sh"]
#ENTRYPOINT ["python3.9", "/home/hlcv_team017/HLCV2023/LENS/test_lens.py"]
