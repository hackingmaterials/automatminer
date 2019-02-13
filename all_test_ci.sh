#!/bin/bash

curl \
            --user ${CIRCLE_CI_TOKEN}: \
            --header "Content-Type: application/json" \
            --data "{\"build_parameters\": {\"CIRCLE_JOB\": \"py372\", \"FULL_TESTS\": \"true\"}}" \
            --request POST "https://circleci.com/api/v1.1/project/github/hackingmaterials/automatminer/tree/master"