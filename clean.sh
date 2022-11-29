#bin/bash
find . | grep -E "(.vscode|__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
