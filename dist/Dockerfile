FROM continuumio/miniconda3

#keep user privileges minimal 
RUN adduser --disabled-login safeuser

#add jupyter lab
RUN conda install jupyterlab

#add python packages
WORKDIR /src
COPY . /src
RUN pip install -r requirements.txt

WORKDIR /usr/src/safeuser


RUN chown -R safeuser:safeuser ./
USER safeuser


EXPOSE 1234
