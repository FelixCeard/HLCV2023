FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt update || true

RUN apt-get install python3.9-venv -y
RUN apt-get install python3-pip -y

WORKDIR /home/hlcv_team017/HLCV2023/LENS/
COPY requirements.txt /home/hlcv_team017/HLCV2023/LENS/requirements.txt
COPY test_lens.py /home/hlcv_team017/HLCV2023/LENS/test_lens.py
COPY run.sh /home/hlcv_team017/HLCV2023/LENS/run.sh

RUN useradd -u 8877 hlcv_team017

RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/LENS/run.sh
RUN chmod ugo+rwx /home/hlcv_team017
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023
RUN chmod ugo+rwx /home/hlcv_team017/HLCV2023/LENS

USER hlcv_team017

ENTRYPOINT ["/home/hlcv_team017/HLCV2023/LENS/run.sh"]
#ENTRYPOINT ["python3.9", "/home/hlcv_team017/HLCV2023/LENS/test_lens.py"]
