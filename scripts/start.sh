#!/bin/bash
set -e
cd /home/ubuntu/crypto-bot

# Load image coordinates written by CodeBuild
source build.env

echo "[start] Pulling image $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG..."
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Tell compose to use the pre-built ECR image instead of building locally
export ECR_IMAGE="$ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG"

mkdir -p data
docker compose up -d
echo "[start] Done."
