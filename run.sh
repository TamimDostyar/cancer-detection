docker build -t cancer-detection .
docker rm cancer_detection && docker run -it --name cancer_detection cancer_detection /bin/bash
docker exec cancer_detection jupyter notebook --allow-root --port 8001 --no-browser --NotebookApp.token='' --ip=0.0.0.0