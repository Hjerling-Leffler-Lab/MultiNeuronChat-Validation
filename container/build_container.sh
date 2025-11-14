#!/bin/bash

# Set Dockerfile path (default to Dockerfile in current directory)
CONTAINER_REGISTRY="ghcr.io/siedler"

PROJECT_NAME="multi-neuron-chat"
CONTAINER_NAME="validation-pipeline"
FULL_CONTAINER_NAME="${PROJECT_NAME}-${CONTAINER_NAME}"

# Extract version from LABEL in Dockerfile
VERSION=$(grep -m 1 'LABEL version=' Dockerfile | cut -d '=' -f 2)

if [ -z "$VERSION" ]; then
  echo "❌ Version not found in Dockerfile LABEL. Please ensure it has a line like: LABEL version=1.0.0"
  exit 1
fi

echo "📦 Building Docker image: ${FULL_CONTAINER_NAME}:${VERSION}"

# Build the Docker image
docker buildx rm multiarch-builder || true
docker buildx create --use --name multiarch-builder

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

docker buildx build \
  --platform linux/amd64 \
  -f "${DOCKERFILE_PATH}" \
  -t "${CONTAINER_REGISTRY}/${FULL_CONTAINER_NAME}:${VERSION}" \
  -t "${CONTAINER_REGISTRY}/${FULL_CONTAINER_NAME}:latest" \
  --push \
  --pull \
  "${PROJECT_ROOT}"

if [ $? -eq 0 ]; then
  echo "✅ Successfully built ${FULL_CONTAINER_NAME}:${VERSION} and pushed to ${CONTAINER_REGISTRY}"
else
  echo "❌ Docker build failed."
  exit 1
fi