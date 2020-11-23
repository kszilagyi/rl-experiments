FROM us.gcr.io/rl-experiments-296208/base:2020-Nov-21-03
ADD . /src
WORKDIR src
RUN pip install -r requirements.txt # todo copy this first and install so don't need to rebuild base always