docker build -t cancer-detection .
 docker run -it --rm \
  -p 8888:8888 \
  -v /Users/tamimdostyar/Documents/coding_stuff/cancer_detection/datasets:/root/.cache/kagglehub/datasets \
  -v /Users/tamimdostyar/Documents/coding_stuff/cancer_detection/datasets:/app/datasets \
  -v "$(pwd)/notebooks":/notebooks \
  cancer-detection \
  jupyter notebook --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root