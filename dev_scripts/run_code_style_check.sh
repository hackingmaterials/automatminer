# Check PEP8 compliance with Flake8

formatting_errors=$(flake8 automatminer)
# https://unix.stackexchange.com/questions/146942/how-can-i-test-if-a-variable-is-empty-or-contains-only-spaces
if [ -z "${formatting_errors// }" ]
  then
    echo "Code is well-formatted."
    exit 0
  else
    >&2 echo "Code misformatted!"
    >&2 echo "$formatting_errors"
    exit 1
  fi
