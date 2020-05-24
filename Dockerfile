FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt update \
    && apt upgrade -y \
    && apt install -y tmux htop wget gdb
    
RUN apt update && apt install -y sox libsox-dev libsox-fmt-all

COPY requirements.txt /root/asr-python/requirements.txt

RUN apt install -y python-pip python3-pip \
    && pip3 install -r /root/asr-python/requirements.txt \
    && pip3 install torchaudio -f https://download.pytorch.org/whl/torch_stable.html

COPY src /root/asr-python/src
COPY montreal-forced-aligner /root/asr-python/montreal-forced-aligner
COPY dict /root/asr-python/dict

# fix montreal forced aligner issue
RUN apt install -y libgfortran3:amd64
