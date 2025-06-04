This seems to work:

```bash
#!/bin/bash

# Configuration
API_BASE_URL="https://ai.bemade.org"
AUTH_TOKEN="..." # Fill with a real token
PDF_PATH="..." # Fill with a real value

# Vérifier si le fichier existe
if [ ! -f "$PDF_PATH" ]; then
    echo "Erreur: Le fichier $PDF_PATH n'existe pas."
    exit 1
fi

# Télécharger le fichier PDF
echo "Téléchargement du fichier PDF..."
FILE_RESPONSE=$(curl -s -X POST \
  "${API_BASE_URL}/api/v1/files/" \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@${PDF_PATH}" \
  -F "process=true")

# Afficher la réponse complète pour le débogage
echo "Réponse complète du serveur:"
echo "$FILE_RESPONSE"

# Vérifier si la réponse contient un ID de fichier
if echo "$FILE_RESPONSE" | grep -q "id"; then
    FILE_ID=$(echo $FILE_RESPONSE | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
    echo "Fichier téléchargé avec succès!"
    echo "ID du fichier: $FILE_ID"
else
    echo "Erreur lors du téléchargement du fichier."
fi
```