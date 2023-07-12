FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt update || true

RUN apt-get install python3.9-venv -y
RUN apt-get install python3-pip -y

WORKDIR /home/hlcv_team017/HLCV2023/hlcv/
COPY requirements.txt /home/hlcv_team017/HLCV2023/hlcv/requirements.txt
COPY test_lens.py /home/hlcv_team017/HLCV2023/hlcv/test_lens.py

RUN python3.9 -m pip install -r /home/hlcv_team017/HLCV2023/hlcv/requirements.txt

RUN useradd -u 8877 hlcv_team017

RUN chmod ugo+rwx /home/hlcv_team017
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/hlcv

USER hlcv_team017

ENTRYPOINT ["python3.9", "/home/hlcv_team017/HLCV2023/hlcv/test_lens.py"]
