FROM nvidia/cuda:11.4.0-base-ubuntu20.04

ARG USER_ID
ARG GROUP_ID

RUN apt -y update
RUN apt install python3 python3-pip nano git -y
RUN apt install python-is-python3 -y

WORKDIR /workspace

COPY requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user