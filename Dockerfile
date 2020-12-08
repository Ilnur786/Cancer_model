FROM ubuntu:latest

WORKDIR /opt

COPY requirements.txt ./

RUN apt update && \
    apt install -y python3-dev python3-pip && \
    apt clean && \
    pip3 install -r requirements.txt

COPY dist/cancer_model-0.1-py3-none-any.whl ./

RUN pip3 install cancer_model-0.1-py3-none-any.whl

CMD ["python3", "-m", "cancer_model"]


