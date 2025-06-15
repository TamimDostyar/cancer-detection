import os
import tensorflow as tf
import subprocess
import time

# Configuration
MODEL_FOLDER = 'models'
NOTEBOOK_PATH = '/app/notebooks/train.ipynb'  # This is correct based on the container structure
DOCKER_MODEL_PATH = '/app/my_cancer_cnn_model.h5'
LOCAL_MODEL_PATH = os.path.join(MODEL_FOLDER, 'cancer_detection_model')

def run_notebook():
    """Run the training notebook to generate the model"""
    try:
        print("Running training notebook...")
        print(f"Notebook path: {NOTEBOOK_PATH}")
        
        # First, verify the notebook exists
        check_cmd = f"docker exec xenodochial_kalam ls -l {NOTEBOOK_PATH}"
        print(f"Checking notebook existence: {check_cmd}")
        subprocess.run(check_cmd, shell=True, check=True)
        
        # Execute the notebook using papermill
        cmd = f"docker exec xenodochial_kalam papermill {NOTEBOOK_PATH} {NOTEBOOK_PATH} --no-input"
        print(f"Executing notebook: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print("Notebook execution completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running notebook: {str(e)}")
        return False

def save_model_from_docker(docker_path, local_path):
    """Save model from Docker container to local storage"""
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        # Copy the .h5 model file from Docker to local
        docker_cp_cmd = f"docker cp xenodochial_kalam:{DOCKER_MODEL_PATH} {MODEL_FOLDER}/"
        print(f"Copying model from Docker: {docker_cp_cmd}")
        os.system(docker_cp_cmd)
        
        # Load the .h5 model
        h5_model_path = os.path.join(MODEL_FOLDER, 'my_cancer_cnn_model.h5')
        print(f"Loading model from: {h5_model_path}")
        model = tf.keras.models.load_model(h5_model_path)
        
        # Save as SavedModel format
        print(f"Saving model to: {LOCAL_MODEL_PATH}")
        model.save(LOCAL_MODEL_PATH, save_format='tf')
        
        # Clean up the .h5 file
        os.remove(h5_model_path)
        
        print(f"Model successfully saved to {LOCAL_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

if __name__ == "__main__":
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("No local model found. Starting model generation process...")
        
        # First, run the notebook to generate the model
        if not run_notebook():
            print("Failed to run training notebook. Please check:")
            print("1. Is the Docker container running?")
            print("2. Is papermill installed in the container?")
            print("3. Are there any errors in the notebook?")
            exit(1)
            
        # Wait a bit for the model to be saved
        print("Waiting for model to be saved...")
        time.sleep(5)
        
        # Then try to save the model
        print("Attempting to save model from Docker to local storage...")
        if not save_model_from_docker(DOCKER_MODEL_PATH, LOCAL_MODEL_PATH):
            print("\nFailed to save model. Please check:")
            print("1. Is the Docker container running?")
            print("2. Is the model path correct in the Docker container?")
            print("3. Do you have write permissions in the local 'models' directory?")
    else:
        print(f"Model already exists at {LOCAL_MODEL_PATH}") 