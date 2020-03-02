FROM continuumio/miniconda3

#### Installing packages ======================================= 
## currently have some problems and need more investigation.
## install python pacakges based on conda environment.yml file
# COPY environment.yml /.
# RUN conda env create -f environment.yml --name myenv
# RUN conda activate myenv

## install python packages based on conda command directly
RUN pip install mlflow==1.6.0
RUN pip install azure-storage-blob==2.1.0
RUN conda install scikit-learn==0.22.1
RUN conda install -c conda-forge pyarrow==0.15.1
RUN conda install pandas==1.0.1

ENV AZURE_STORAGE_CONNECTION_STRING='xxx'