# AI on Containers - Demo

# About

This repository contains the source code and instructions for how to leverage Google Kubernetes Engine (GKE) for distributed AI/ML training and Cloud Run for model serving. GPUs can be attached to both services for high-demand workloads. 

## Prerequisites

- A GCP account and project
- An IDE of choice (e.g. VSCode)
- Docker

Enable the following APIs on GCP:
- Artifact Registry
- Cloud Storage
- Cloud Run
- GKE
- Cloud Build

## Directory Structure
```
gke-distributed-training-demo/ 
├── trainer/ 
│ ├── train.py
│ ├── data.py 
│ └── utils.py (Optional) 
├── serving/ 
│ ├── app.py 
│ └── requirements.txt 
├── Dockerfile.train 
├── Dockerfile.serving 
├── k8s/ 
│ └── job.yaml 
├── .gitlab-ci.yml 
├── requirements.txt 
└── README.md
```

## Setup

### Setting up GCP project

1/ Create the GCP project with the name you want: `aioc-demo`

2/ Define the following environment variables
```
export REGION=us-west1
export PROJECT=aioc-demo
export GAR_REPO=aioc-docker-repo
export BUCKET=aioc-demo-bucket
```

3/ Ensure you are logged in and set in the right project
```
gcloud auth login

gcloud config set project ${PROJECT}
```

4/ Enable the following APIs
```
gcloud services enable \
	artifactregistry.googleapis.com \
	storage-api.googleapis.com \
	run.googleapis.com \
	container.googleapis.com
```

5/ Create a service account with necessary permissions
```
gcloud iam service-accounts create aioc-training-sa \ 
--display-name="AIoC Training Service Account"

gcloud projects add-iam-policy-binding ${PROJECT} \ 
--member="serviceAccount:aioc-training-sa@${PROJECT}.iam.gserviceaccount.com" \ 
--role="roles/storage.objectCreator"
```

### GKE Cluster Setup

6/ Create a GKE Autopilot
### Set up remaining infrastructure

7/ Create a repository in Artifact Registry
```
gcloud artifacts repositories create ${GAR_REPO} \
    --repository-format=docker \
    --location=${REGION}
```
