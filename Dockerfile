ARG BASE_IMAGE=rayproject/ray:2.9.2-py39

FROM $BASE_IMAGE

WORKDIR /text_ml

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && sudo chown -R ray:users /text_ml \
    && sudo chmod 755 /text_ml

# Download nltk dataset locally
RUN python -m nltk.downloader punkt

# Copy model, dataset, tokenizer into docker image
COPY t5-small ./t5-small
COPY sql-create-context ./sql-create-context
COPY t5-small-tokenizer ./t5-small-tokenizer

RUN sudo chown -R ray:users /text_ml/t5-small \
    && sudo chmod 755 /text_ml/t5-small
RUN sudo chown -R ray:users /text_ml/sql-create-context \
    && sudo chmod 755 /text_ml/sql-create-context

# Copy evaluate script
COPY evaluate ./evaluate
RUN sudo chown -R ray:users /text_ml/evaluate \
    && sudo chmod 755 /text_ml/evaluate

# Install nightly build of Ray to get latest fixes
# https://github.com/ray-project/ray/pull/42953
RUN pip uninstall -y ray
RUN pip install -U "ray[default] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"

COPY *.py ./

USER ray
