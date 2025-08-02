#!/bin/bash

pip install --upgrade build

python -m build

pip install --upgrade twine

# Note that the API must be stored in your $HOME/.pypirc for this to work
python -m twine upload dist/* --repository pandas-ylt
