FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get -y install tmux python3 python3-pip

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip3 install -r requirements.txt

CMD bash
