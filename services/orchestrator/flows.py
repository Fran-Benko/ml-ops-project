from prefect import flow, task
import subprocess
import os

@task
def generate_data():
    print("Generating synthetic data...")
    result = subprocess.run(["python", "-m", "src.sintetic_gen"], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Data generation failed: {result.stderr}")
    return result.stdout

@task
def train_model(dataset_name):
    print(f"Training model with dataset: {dataset_name}")
    result = subprocess.run(["python", "-m", "src.train_pipeline", "--data", dataset_name], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")
    return result.stdout

@flow(name="MLOps Pipeline")
def mlops_pipeline(dataset_name: str = None):
    if not dataset_name:
        generate_data()
    else:
        train_model(dataset_name)

if __name__ == "__main__":
    mlops_pipeline.serve(name="mlops-deployment", port=4200)
