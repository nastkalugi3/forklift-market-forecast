#!/usr/bin/env bash
# Совместимость: то же, что python update.py (см. python update.py --help)
cd "$(dirname "$0")" && python update.py "$@"
