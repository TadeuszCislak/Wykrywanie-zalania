import kagglehub

# Download latest version
path = kagglehub.dataset_download("robertomarinoformica/sen1floods11-dataset")

print("Path to dataset files:", path)