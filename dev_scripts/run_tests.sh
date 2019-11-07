#!/bin/bash

coverage run setup.py test
coverage xml
if [[ ! -z "$CODACY_PROJECT_TOKEN" ]]; then
  python-codacy-coverage -r coverage.xml
else
  echo "No Codacy Project Token Defined, skipping..."
fi