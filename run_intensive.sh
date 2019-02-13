#!/bin/bash


# Submit and run an intensive test to circleci.

curl \
            --user ${CIRCLE_CI_TOKEN}: \
            --header "Content-Type: application/json" \
            --data "{\"build_parameters\": {\"CIRCLE_JOB\": \"py372\", \"SKIP_INTENSIVE\": \"0\"}}" \
            --request POST "https://circleci.com/api/v1.1/project/github/hackingmaterials/automatminer/tree/master"