# LDTO to Elasticsearch Converter

Dit script converteert LDTO (Linked Data Theater Objects) in TTL formaat naar een Elasticsearch index.

## Configuratie

Maak een `.env` bestand aan met de volgende instellingen:

```env
ES_USERNAME=username
ES_PASSWORD=your_password
ES_HOST=your.elasticsearch.host
ES_PORT=443
```

## Gebruik

Het script kan op twee manieren worden aangeroepen:

### Index Optimalisatie en Privacy

De index is geoptimaliseerd voor prestaties, opslagruimte en privacy/AVG-compliance:
- Het `full_text` veld wordt uitgesloten van de `_source` om:
  1. De index grootte te beperken
  2. Privacy gevoelige informatie te beschermen (AVG/GDPR)
  3. Te voorkomen dat de volledige documentinhoud wordt opgeslagen
- De tekst blijft wel doorzoekbaar voor geautoriseerde gebruikers
- Deze instelling is cruciaal en mag niet worden verwijderd

Deze aanpak zorgt ervoor dat:
- Documenten doorzoekbaar zijn voor gebruikers met de juiste rechten
- De volledige documentinhoud alleen toegankelijk is via de originele bronbestanden
- Er geen onnodige kopieën van mogelijk privacy-gevoelige informatie worden opgeslagen

### 1. Eenvoudige manier (Legacy mode)

```bash
python ldto2es.py pad/naar/bestand.ttl --docs-dir pad/naar/docs
```

Deze manier gebruikt de standaard instellingen:
- Index naam: 'ldto-objects'
- Inner result window size: 500

### 2. Uitgebreide manier (met extra opties)

#### Converteren van TTL naar Elasticsearch

```bash
python ldto2es.py convert \
    --file pad/naar/bestand.ttl \
    --docs-dir pad/naar/docs \
    --index mijn-index-naam \
    --window-size 1000
```

Alle parameters:
- `--file`: Het TTL bestand om te converteren
- `--docs-dir`: Map met de documenten
- `--index`: Naam van de Elasticsearch index (standaard: 'ldto-objects')
- `--window-size`: Grootte van max_inner_result_window (standaard: 500)

#### Aanpassen van inner result window

Je kunt de max_inner_result_window instelling van een bestaande index aanpassen met:

```bash
python ldto2es.py update-window --index mijn-index-naam --window-size 1000
```

Alle parameters:
- `--index`: Naam van de Elasticsearch index (standaard: 'ldto-objects')
- `--window-size`: Nieuwe grootte voor max_inner_result_window (standaard: 500)

## Voorbeelden

1. Simpele conversie:
```bash
python ldto2es.py data/archief.ttl --docs-dir data/documenten
```

2. Conversie met aangepaste instellingen:
```bash
python ldto2es.py convert --file data/archief.ttl --docs-dir data/documenten --index archief-2024 --window-size 1000
```

3. Window size aanpassen van bestaande index:
```bash
python ldto2es.py update-window --index archief-2024 --window-size 2000
```
