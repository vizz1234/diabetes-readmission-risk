# Cloud Run Deployment Guide

This guide describes how to deploy the Diabetes 30-Day Readmission Risk System prediction service to Google Cloud Run.

## Prerequisites
1. Installed Google Cloud SDK (`gcloud`).
2. GCP Project created and billing enabled.
3. Enabled APIs: Cloud Build, Cloud Run, and Google Container Registry.
   ```bash
   gcloud services enable billingbudgets.googleapis.com run.googleapis.com builds.googleapis.com containerregistry.googleapis.com
   ```

## Deploying
Execute the deploy script, passing your GCP Project ID:
```bash
chmod +x infra/cloudrun/deploy.sh
./infra/cloudrun/deploy.sh YOUR_PROJECT_ID
```

## Rollback Procedure
If a release causes issues and you need to immediately roll back to a previous revision:
1. Identify the revision name you wish to roll back to:
   ```bash
   gcloud run revisions list --service=diabetes-api --region=asia-south1
   ```
2. Direct 100% of traffic to the target previous revision:
   ```bash
   gcloud run services update-traffic diabetes-api \
     --to-revisions=PREVIOUS_REVISION_NAME=100 \
     --region=asia-south1
   ```
   *Replace `PREVIOUS_REVISION_NAME` with the actual revision identifier (e.g., `diabetes-api-00002-vob`).*

## Cost Control & Budget Alerts
Cloud Run is highly cost-effective because it scales to zero when idle. To prevent unexpected charges:
1. Note that you can utilize the GCP $300/90-day free trial credits.
2. **Set up a Budget Alert**:
   - Go to the GCP Console -> **Billing** -> **Budgets & alerts**.
   - Create a budget of $10 (or a value of your choice).
   - Set threshold rules at 50%, 90%, and 100% of budget consumption to trigger Email/PubSub notifications.
