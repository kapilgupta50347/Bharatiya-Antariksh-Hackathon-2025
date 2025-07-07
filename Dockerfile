FROM python:3.12-slim

# Install system-level dependencies for NetCDF and HDF5
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pm-conc-map

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]