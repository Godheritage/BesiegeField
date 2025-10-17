#!/usr/bin/env bash
set -e

DOWNLOAD_URL="https://drive.usercontent.google.com/download?id=1-5twWawVCuOU233F6sT0oB2ahFLUsIrF&export=download&authuser=0&confirm=t&uuid=965c740f-23e1-48e4-94ed-b214b29ee9c0&at=AKSUxGPsMtLtKqtt9gGgX-aE28ZT:1760617141860 "

wget -O cold_start_dataset.zip --no-check-certificate "$CONFIRM_URL"
unzip -q cold_start_dataset.zip -d extracted
