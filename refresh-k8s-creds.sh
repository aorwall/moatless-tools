#!/bin/bash

# Refresh AWS EKS credentials
echo "Refreshing AWS EKS credentials..."
aws eks update-kubeconfig --region us-west-2 --name swe-search --alias aws-swe-search

# Refresh Azure AKS credentials
echo "Refreshing Azure AKS credentials..."
az aks get-credentials --resource-group screening --name tm-core --overwrite-existing --context azure-tm-core

# Refresh Moatless credentials (assuming it's an AKS cluster)
echo "Refreshing Moatless credentials..."
az aks get-credentials --resource-group rg-moatless --name moatless-k8s --overwrite-existing --context moatless-k8s

echo "All credentials refreshed!"

# Show current contexts
echo "Available contexts:"
kubectl config get-contexts 