#!/bin/bash
set -e

# Deployment script for Diabetes 30-Day Readmission Risk API on Google Cloud Run.
# Make sure project ID is set or provided as an argument.

PROJECT_ID=$1

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: ./deploy.sh [PROJECT_ID]"
    echo "Example: ./deploy.sh my-gcp-project-123"
    exit 1
fi

echo "Using Project ID: $PROJECT_ID"
echo "Submitting docker build to Google Container Registry..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/diabetes-api --project $PROJECT_ID

echo "Deploying to Google Cloud Run in asia-south1..."
gcloud run deploy diabetes-api \
  --image gcr.io/$PROJECT_ID/diabetes-api \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --project $PROJECT_ID

echo "Deployment complete!"
