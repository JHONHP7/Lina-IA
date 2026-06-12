#!/usr/bin/env bash

# Garante que o script rode a partir da raiz do projeto, independentemente de onde for chamado
cd "$(dirname "$0")/.."

PYTHONPATH=src python src/ingestion/ingest-tb.py
