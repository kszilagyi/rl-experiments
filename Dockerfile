FROM us.gcr.io/rl-experiments-296208/base:2020-Nov-21-03
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD ./src /rl-experiments/src
ADD ./jobs /rl-experiments/jobs
WORKDIR rl-experiments
