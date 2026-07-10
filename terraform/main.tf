provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  type        = string
  description = "The GCP project ID to deploy resources to."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Target deployment region."
}

variable "api_image" {
  type        = string
  description = "Container image URI for the FastAPI server."
}

variable "dashboard_image" {
  type        = string
  description = "Container image URI for the Streamlit dashboard."
}

# 1. GCS Bucket for MLflow artifacts store
resource "google_storage_bucket" "mlflow_artifacts" {
  name          = "${var.project_id}-mlflow-artifacts"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true
}

# 2. Artifact Registry for Container Storage
resource "google_artifact_registry_repository" "prrp_registry" {
  location      = var.region
  repository_id = "prrp-repository"
  description   = "Docker repository for Patient Readmission Risk Predictor services"
  format        = "DOCKER"
}

# 3. Cloud Run Service for FastAPI API
resource "google_cloud_run_v2_service" "api_service" {
  name     = "prrp-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = var.api_image
      ports {
        container_port = 8000
      }
      env {
        name  = "MLFLOW_TRACKING_URI"
        value = "http://localhost:5000" # Update to remote MLflow URL in production
      }
      env {
        name  = "ALLOWED_ORIGINS"
        value = "*" # Restrict this to dashboard URL in production
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
}

# 4. Cloud Run Service for Streamlit Dashboard
resource "google_cloud_run_v2_service" "dashboard_service" {
  name     = "prrp-dashboard"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = var.dashboard_image
      ports {
        container_port = 8501
      }
      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api_service.uri
      }
      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
    }
  }
}

# 5. Allow Public Access to Dashboard Service
resource "google_cloud_run_v2_service_iam_member" "public_dashboard" {
  name     = google_cloud_run_v2_service.dashboard_service.name
  location = google_cloud_run_v2_service.dashboard_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# 6. Allow Public Access to API Service (Required if browser makes direct AJAX calls)
resource "google_cloud_run_v2_service_iam_member" "public_api" {
  name     = google_cloud_run_v2_service.api_service.name
  location = google_cloud_run_v2_service.api_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "api_uri" {
  value       = google_cloud_run_v2_service.api_service.uri
  description = "The URI of the Cloud Run API service."
}

output "dashboard_uri" {
  value       = google_cloud_run_v2_service.dashboard_service.uri
  description = "The URL to access the Streamlit Dashboard."
}
