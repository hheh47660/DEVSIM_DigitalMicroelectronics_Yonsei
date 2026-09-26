#!/bin/bash
# Posizionati nella directory dello script
cd "$(dirname "$0")"
# Aggiungi i binari di Conda al PATH
export PATH="$(pwd)/.conda/bin:$PATH"
# Esegui lo script Python tramite il modulo streamlit di Python3
python3 -m streamlit run demo/_launcher.py
# Se si verifica un errore, sospendi prima di chiudere
if [ $? -ne 0 ]; then
    read -p "Premere Invio per continuare..."
fi
