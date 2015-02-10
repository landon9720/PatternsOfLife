#!/bin/bash

set -e

push() {
  aws --region=us-east-1 s3 cp "$1" "s3://yatchmenoff.com/$1"
}

make buddies.html

push buddies.html
push buddies.js
push buddies.data
