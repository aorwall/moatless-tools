#!/bin/bash
# Default to local environment if not specified
ENV=${1:-local}
python -m scripts.run_api --dev --env $ENV