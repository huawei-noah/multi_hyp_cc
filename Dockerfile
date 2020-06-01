FROM pytorch/pytorch

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get -y install python-opencv tmux

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt

CMD bash
