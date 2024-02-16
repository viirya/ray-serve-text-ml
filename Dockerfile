ARG BASE_IMAGE=rayproject/ray:2.9.0-py39

FROM $BASE_IMAGE

WORKDIR /text_ml

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && sudo chown -R ray:users /text_ml \
    && sudo chmod 755 /text_ml

COPY *.py ./
COPY t5-small ./t5-small

USER ray
