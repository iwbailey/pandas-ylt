#!/bin/bash

pip install --upgrade build

python -m build

pip install --upgrade twine

python -m twine upload dist/*
