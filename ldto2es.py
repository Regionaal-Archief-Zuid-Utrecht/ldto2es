from rdflib import Graph, Namespace, URIRef, BNode, Literal
from elasticsearch import Elasticsearch
import os
import requests
from urllib.parse import urljoin, urlparse, unquote
import textract
import io
import logging
import time
from functools import lru_cache, wraps
import dotenv
from pathlib import Path
import argparse

# Load environment variables from .env file
dotenv.load_dotenv()

# Define namespaces
LDTO = Namespace("https://data.razu.nl/def/ldto/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DCT = Namespace("http://purl.org/dc/terms/")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_last_segment(uri):
    """Extract the last segment of a URI."""
    return str(uri).split('/')[-1]

def normalize_label(label):
    """Normalize a label to be used as a field name"""
    return label.lower().replace(' ', '_')

def dereference_uri(uri):
    """Dereference a URI and return a Graph with its content"""
    try:
        response = requests.get(uri, headers={'Accept': 'text/turtle'})
        if response.status_code == 429:  # Rate limit hit
            time.sleep(1)  # Wait a bit
            response = requests.get(uri, headers={'Accept': 'text/turtle'})
            
        if response.status_code != 200:
            logger.warning(f"Failed to dereference {uri}: {response.status_code}")
            return None
            
        # Parse the response into a temporary graph
        temp_g = Graph()
        temp_g.parse(data=response.text, format='turtle')
        return temp_g
        
    except Exception as e:
        logger.error(f"Error dereferencing {uri}: {str(e)}")
        return None

def get_skos_triple_value(g, subject, predicate):
    """Get a single value from a SKOS triple"""
    for _, _, value in g.triples((URIRef(subject), predicate, None)):
        return str(value)
    return None

@lru_cache(maxsize=1000)
def get_skos_label(uri):
    """Fetch SKOS prefLabel for a given URI. Results are cached using lru_cache."""
    g = dereference_uri(uri)
    if not g:
        return None
        
    label = get_skos_triple_value(g, uri, SKOS.prefLabel)
    if not label:
        logger.warning(f"No prefLabel found for {uri}")
    return label

@lru_cache(maxsize=1000)
def get_beperking_gebruik_labels(uri):
    """Get the scheme and label for a beperkingGebruik URI by dereferencing the URIs"""
    logger.info(f"Dereferencing URI: {uri}")
    
    # Get the beperking type label and scheme
    g = dereference_uri(uri)
    if not g:
        return None
    
    # Get the prefLabel and inScheme
    type_label = get_skos_triple_value(g, uri, SKOS.prefLabel)
    scheme_uri = get_skos_triple_value(g, uri, SKOS.inScheme)
    
    if not type_label or not scheme_uri:
        logger.warning(f"No type_label or scheme found for {uri}")
        return None
    
    # Now dereference the scheme to get its label
    logger.info(f"Dereferencing scheme: {scheme_uri}")
    scheme_g = dereference_uri(scheme_uri)
    if not scheme_g:
        return None
    
    # Get the scheme label
    scheme_label = get_skos_triple_value(scheme_g, scheme_uri, SKOS.prefLabel)
    if not scheme_label:
        logger.warning(f"No scheme label found for {scheme_uri}")
        return None
    
    normalized = normalize_label(scheme_label)
    logger.info(f"Found labels - scheme: {scheme_label} -> {normalized}, type: {type_label}")
    
    return {
        'scheme': scheme_uri,
        'scheme_label': normalized,
        'type_label': type_label
    }

def get_scheme_labels(g):
    """Get labels for all schemes and their types."""
    query = """
    PREFIX ldto: <https://data.razu.nl/def/ldto/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?scheme ?scheme_label ?type ?type_label
    WHERE {
        ?scheme a skos:ConceptScheme ;
                skos:prefLabel ?scheme_label .
        ?type skos:inScheme ?scheme ;
              skos:prefLabel ?type_label .
    }
    """
    scheme_labels = {}
    for row in g.query(query):
        scheme_uri = str(row['scheme'])
        scheme_label = str(row['scheme_label'])
        type_label = str(row['type_label'])
        
        # Convert scheme label to snake_case for use as field name
        field_name = scheme_label.lower().replace(' ', '_')
        scheme_labels[scheme_uri] = field_name
        
        logger.info(f"Found labels - scheme: {scheme_label} -> {field_name}, type: {type_label}")
    
    return scheme_labels

def get_dekking_types(g):
    """Get all dekkingInTijdType values."""
    query = """
    PREFIX ldto: <https://data.razu.nl/def/ldto/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?type ?label
    WHERE {
        ?type a ldto:DekkingInTijdType ;
              skos:prefLabel ?label .
    }
    """
    dekking_types = {}
    for row in g.query(query):
        type_uri = str(row['type'])
        label = str(row['label'])
        dekking_types[type_uri] = label
        logger.info(f"Found dekkingInTijdType: {label} ({type_uri})")
    return dekking_types

def find_root_archive(g, doc_uri, documents):
    """Find the root archive by following ldto:isOnderdeelVan until we reach the top level."""
    current_uri = doc_uri
    while True:
        # Get parent URI through ldto:isOnderdeelVan
        parent_query = """
        PREFIX ldto: <https://data.razu.nl/def/ldto/>
        SELECT ?parent
        WHERE {
            <%s> ldto:isOnderdeelVan ?parent .
        }
        """ % current_uri
        
        results = g.query(parent_query)
        parent_uris = [str(row['parent']) for row in results]
        
        if not parent_uris:  # No parent found, this is the root
            # Get the naam of the current document
            naam_query = """
            PREFIX ldto: <https://data.razu.nl/def/ldto/>
            SELECT ?naam
            WHERE {
                <%s> ldto:naam ?naam .
            }
            """ % current_uri
            naam_results = g.query(naam_query)
            for row in naam_results:
                return str(row['naam'])
            return None  # No naam found
            
        current_uri = parent_uris[0]  # Follow the first parent

def ensure_cache_dir():
    """Ensure the cache directory exists"""
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def get_cache_path(original_path):
    """Get the cache file path for a given original file path"""
    cache_dir = ensure_cache_dir()
    original_name = os.path.basename(original_path)
    base_name = os.path.splitext(original_name)[0]
    return os.path.join(cache_dir, f"{base_name}.txt")

def get_cached_text(cache_path):
    """Get text from cache if it exists"""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def save_to_cache(cache_path, text):
    """Save extracted text to cache"""
    # Ensure text is a string, even if None
    text = text or ""
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(text)

def extract_text_from_file(file_path):
    """Extract text from a file using textract, with caching"""
    # Get cache path and check if cached version exists
    cache_path = get_cache_path(file_path)
    cached_text = get_cached_text(cache_path)
    
    if cached_text is not None:
        logger.info(f"Using cached text for {file_path}")
        return cached_text if cached_text != "" else None
    
    # If not in cache, extract text
    try:
        text = textract.process(file_path, encoding='utf-8').decode('utf-8')
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {e}")
        text = None
    
    # Save to cache (even if extraction failed)
    save_to_cache(cache_path, text)
    
    return text

def get_file_content(url, docs_dir='docs'):
    """Get text content from a file in the specified directory"""
    try:
        # Extract filename from URL and decode it
        filename = unquote(os.path.basename(urlparse(url).path))
        file_path = os.path.join(docs_dir, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
            
        # Extract text using textract
        return extract_text_from_file(file_path)
            
    except Exception as e:
        logger.error(f"Error processing file from URL {url}: {e}")
        return None

def create_index(es, index_name, dekking_types, scheme_labels, window_size=500):
    """Create Elasticsearch index with mapping for LDTO fields if it doesn't exist
    
    Args:
        es: Elasticsearch client
        index_name: Name of the index to create
        dekking_types: List of dekking types for mapping
        scheme_labels: List of scheme labels for mapping
        window_size: Size for max_inner_result_window (default: 100)
    """
    # Basic mapping for known fields
    mapping = {
        "settings": {
            "index": {
                "max_inner_result_window": window_size
            }
        },
        "mappings": {
            "dynamic": "true",  # Allow dynamic fields for event dates
            "dynamic_templates": [
                {
                    "dates": {
                        "match_pattern": "regex",
                        "match": "^(created|modified|published|approved|received|sent|processed|archived|registered|completed|reviewed|validated).*",
                        "mapping": {
                            "type": "date"
                        }
                    }
                }
            ],
            "properties": {
                "@id": {
                    "type": "keyword"
                },
                "id": {"type": "keyword"},
                "naam": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "omschrijving": {"type": "text"},
                "full_text": {"type": "text"},
                "classificatie": {"type": "keyword"},
                "classificatie_uri": {"type": "keyword"},
                "archiefvormer": {"type": "keyword"},
                "archiefvormer_uri": {"type": "keyword"},
                "aggregatieniveau": {"type": "keyword"},
                "aggregatieniveau_uri": {"type": "keyword"},
                "bestand_url": {"type": "keyword"},
                "archief": {"type": "keyword"}
            }
        }
    }
    
    # Add dynamic mappings for dekking types
    for dekking_type in dekking_types:
        field_name = normalize_label(dekking_type)
        mapping["mappings"]["properties"][field_name] = {
            "type": "date"
        }
    
    # Add mappings for scheme labels
    for scheme in scheme_labels:
        scheme_name = normalize_label(scheme['label'])
        mapping["mappings"]["properties"][scheme_name] = {
            "type": "keyword"
        }
        mapping["mappings"]["properties"][f"{scheme_name}_uri"] = {
            "type": "keyword"
        }
    
    # Create the index if it doesn't exist
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index {index_name}")
    else:
        logger.info(f"Index {index_name} already exists")

def update_inner_result_window(es, index_name, size=100):
    """Update max_inner_result_window setting for an existing index.
    
    Args:
        es: Elasticsearch client
        index_name: Name of the index to update
        size: New size for max_inner_result_window (default: 100)
    """
    try:
        es.indices.put_settings(
            index=index_name,
            body={
                "index": {
                    "max_inner_result_window": size
                }
            }
        )
        print(f"Successfully updated max_inner_result_window to {size} for index {index_name}")
    except Exception as e:
        print(f"Error updating max_inner_result_window: {e}")

def create_elasticsearch_client():
    """Create and configure Elasticsearch client"""
    es_host = os.getenv('ES_HOST', 'es.digitopia.nl')
    es_port = int(os.getenv('ES_PORT', '443'))
    es_username = os.getenv('ES_USERNAME')
    es_password = os.getenv('ES_PASSWORD')
    
    if not all([es_username, es_password]):
        raise ValueError("ES_USERNAME and ES_PASSWORD must be set in .env file")
    
    return Elasticsearch(
        f"https://{es_host}:{es_port}",
        basic_auth=(es_username, es_password),
        verify_certs=False
    )

def process_time_field(row, type_field, time_field, doc):
    """Process a time-related field and add it to the document.
    
    Args:
        row: The query result row
        type_field: The field containing the type URI (e.g., 'eventType' or 'dekkingType')
        time_field: The field containing the time value (e.g., 'eventTijd' or 'begin'/'eind')
        doc: The document to add the field to
    """
    if row[type_field] and row[time_field]:
        type_label = get_skos_label(row[type_field])
        if type_label:
            field_name = normalize_label(type_label)
            doc[field_name] = str(row[time_field])

def process_skos_field(row, field_name, doc):
    """Process a SKOS-related field and add it to the document.
    
    Args:
        row: The query result row
        field_name: The name of the field (e.g., 'classificatie', 'archiefvormer')
        doc: The document to add the field to
    """
    if row[field_name]:
        label = get_skos_label(row[field_name])
        if label:
            doc[field_name] = label
            doc[f'{field_name}_uri'] = str(row[field_name])
            return label
    return None

def process_relation_field(row, field_name, doc, target_field=None):
    """Process a relation field and add it to the document.
    
    Args:
        row: The query result row
        field_name: The name of the field in the row
        doc: The document to add the field to
        target_field: Optional different name for the target field in doc
    """
    if row[field_name]:
        target = target_field or field_name
        doc[target] = extract_last_segment(row[field_name])

def append_to_full_text(doc, content, separator="\n"):
    """Veilig tekst toevoegen aan full_text veld"""
    if not content:
        return
        
    if isinstance(content, (list, tuple)):
        text = separator.join(str(item) for item in content if item)
    else:
        text = str(content).strip()
    
    if text:
        if 'full_text' not in doc:
            doc['full_text'] = text
        else:
            doc['full_text'] += f"{separator}{text}"

def init_document(row):
    """Basis documentstructuur initialiseren"""
    doc = {
        'id': extract_last_segment(row['obj']),
        '@id': str(row['obj']),
        'naam': str(row['naam'])
    }
    append_to_full_text(doc, row['naam'])
    return doc

def process_optional_field(row, doc, field_name, processor=None):
    """Generieke verwerking van optionele velden"""
    if row.get(field_name):
        if processor:
            processor(row[field_name], doc)
        else:
            doc[field_name] = str(row[field_name])

def process_bestand(url, doc, docs_dir):
    """Verwerk bestand gerelateerde data"""
    doc['bestand_url'] = str(url)
    file_content = get_file_content(str(url), docs_dir)
    append_to_full_text(doc, file_content)

def convert_ttl_to_es(ttl_file, docs_dir='docs'):
    """Convert TTL file to Elasticsearch documents."""
    g = Graph()
    g.parse(ttl_file, format="turtle")
    
    # First get all scheme labels
    scheme_labels = get_scheme_labels(g)
    
    # Get all dekking types
    dekking_types = get_dekking_types(g)
    
    # Field processors configuratie
    field_processors = {
        'omschrijving': lambda val, doc: (
            doc.update({'omschrijving': str(val)}) or 
            append_to_full_text(doc, val)
        ),
        'bestandURL': lambda val, doc: process_bestand(val, doc, docs_dir),
        'trefwoorden': lambda val, doc: (
            doc.update({'trefwoorden': [kw.strip() for kw in str(val).split(',')]}) or
            append_to_full_text(doc, doc['trefwoorden'], ", ")
        ),
        'betrokkeneActor': lambda val, doc: (
            process_skos_field(row, 'betrokkeneActor', doc) and 
            append_to_full_text(doc, doc.get('betrokkeneActor'))
        )
    }

    # Find all ldto:Informatieobject instances
    query = """
    PREFIX ldto: <https://data.razu.nl/def/ldto/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT DISTINCT ?obj ?naam ?omschrijving ?classificatie ?archiefvormer ?aggregatieniveau 
           ?dekking ?dekkingType ?begin ?eind
           ?bestand ?bestandURL ?bestandNaam ?beperkingen
           ?bevatOnderdeel ?isOnderdeelVan
           ?betrokkene ?betrokkeneActor ?event ?eventType ?eventTijd
           (GROUP_CONCAT(?trefwoord; separator=',') as ?trefwoorden)
    WHERE {
        ?obj a ldto:Informatieobject ;
             ldto:naam ?naam .
        OPTIONAL { ?obj ldto:classificatie ?classificatie }
        OPTIONAL { ?obj ldto:archiefvormer ?archiefvormer }
        OPTIONAL { ?obj ldto:aggregatieniveau ?aggregatieniveau }
        OPTIONAL { ?obj ldto:omschrijving ?omschrijving }
        OPTIONAL { 
            ?obj ldto:dekkingInTijd ?dekking .
            ?dekking ldto:dekkingInTijdType ?dekkingType .
            OPTIONAL { ?dekking ldto:begin ?begin }
            OPTIONAL { ?dekking ldto:eind ?eind }
        }
        OPTIONAL { 
            ?obj ldto:heeftRepresentatie ?bestand .
            ?bestand ldto:URLBestand ?bestandURL
        }
        OPTIONAL { ?obj ldto:beperkingGebruik ?beperkingen }
        OPTIONAL { ?obj ldto:bevatOnderdeel ?bevatOnderdeel }
        OPTIONAL { ?obj ldto:isOnderdeelVan ?isOnderdeelVan }
        OPTIONAL { ?obj ldto:trefwoord ?trefwoord }
        OPTIONAL { 
            ?obj ldto:betrokkene ?betrokkene .
            ?betrokkene ldto:betrokkeneActor ?betrokkeneActor
        }
        OPTIONAL {
            ?obj ldto:event ?event .
            ?event ldto:eventType ?eventType ;
                   ldto:eventTijd ?eventTijd .
        }
    }
    GROUP BY ?obj ?naam ?omschrijving ?classificatie ?archiefvormer ?aggregatieniveau 
             ?dekking ?dekkingType ?begin ?eind
             ?bestand ?bestandURL ?bestandNaam ?beperkingen
             ?bevatOnderdeel ?isOnderdeelVan
             ?betrokkene ?betrokkeneActor ?event ?eventType ?eventTijd
    """
    
    results = g.query(query)
    
    # Process results into documents
    documents = {}
    
    for row in results:
        doc = init_document(row)
        
        # Find and add the root archive name
        root_archive = find_root_archive(g, str(row['obj']), documents)
        if root_archive:
            doc['archief'] = root_archive
            
        # Process time-related fields
        # Handle events
        process_time_field(row, 'eventType', 'eventTijd', doc)
        
        # Handle dekking begin and end
        if row['dekkingType']:
            process_time_field(row, 'dekkingType', 'begin', doc)
            process_time_field(row, 'dekkingType', 'eind', doc)
        
        # Automatische veldverwerking
        for field, processor in field_processors.items():
            process_optional_field(row, doc, field, processor)
        
        # Process SKOS fields
        for field in ['classificatie', 'archiefvormer', 'aggregatieniveau']:
            label = process_skos_field(row, field, doc)
            if label:
                append_to_full_text(doc, label)
        
        # Add hierarchical relationships if present
        process_relation_field(row, 'bevatOnderdeel', doc, 'bevat_onderdeel')
        process_relation_field(row, 'isOnderdeelVan', doc, 'is_onderdeel_van')

        # Store document
        documents[doc['id']] = doc

    # Process beperkingGebruik for each document
    for doc_id, doc in documents.items():
        # Get all beperkingGebruik values for this document
        query = f"""
        PREFIX ldto: <https://data.razu.nl/def/ldto/>
        SELECT ?beperking
        WHERE {{
            ?obj ldto:beperkingGebruik ?beperking .
            FILTER(STRENDS(str(?obj), "{doc_id}"))
        }}
        """
        
        beperkingen = []
        for row in g.query(query, initNs={'ldto': LDTO}):
            beperkingen.append(str(row[0]))
            
        if beperkingen:
            logger.info(f"Processing beperkingen for {doc_id}: {','.join(beperkingen)}")
            
            # Process each beperking
            for beperking_uri in beperkingen:
                result = get_beperking_gebruik_labels(beperking_uri)
                if result:
                    # Add the type label to the corresponding scheme field
                    scheme_label = result['scheme_label']
                    if scheme_label not in doc:
                        doc[scheme_label] = []
                    doc[scheme_label].append(result['type_label'])
                    logger.info(f"Added beperking to {scheme_label}: {result['type_label']}")
                    
    # Process dekkingInTijd for each document
    for doc_id, doc in documents.items():
        # Get all dekkingInTijd values for this document
        query = f"""
        PREFIX ldto: <https://data.razu.nl/def/ldto/>
        SELECT ?type ?begin ?eind
        WHERE {{
            ?obj ldto:dekkingInTijd ?dekking .
            ?dekking ldto:dekkingInTijdType ?type .
            OPTIONAL {{ ?dekking ldto:begin ?begin }}
            OPTIONAL {{ ?dekking ldto:eind ?eind }}
            FILTER(STRENDS(str(?obj), "{doc_id}"))
        }}
        """
        
        for row in g.query(query, initNs={'ldto': LDTO}):
            dekking_type, begin, eind = row
            
            if str(dekking_type) in dekking_types:
                field_name = dekking_types[str(dekking_type)]
                
                # Create date range
                date_range = {}
                if begin:
                    date_range['gte'] = str(begin)
                if eind:
                    date_range['lte'] = str(eind)
                    
                if date_range:
                    doc[f"{field_name}_range"] = date_range
                    logger.info(f"Added date range for {doc_id}: {begin} - {eind}")
                    
        logger.info(f"Processed document {doc_id} with fields: {list(doc.keys())}")
        
    return documents, dekking_types, scheme_labels

def index_documents(es, index_name, documents):
    """Index documents using Elasticsearch bulk API."""
    if not documents:
        logger.warning("No documents to index")
        return

    # Prepare bulk indexing actions
    actions = []
    for doc_id, doc in documents.items():
        # Actie regel
        actions.append({"index": {"_index": index_name, "_id": doc_id}})
        # Data regel
        actions.append(doc)

    # Start with smaller batch size (in paren van 2 omdat elke doc twee regels heeft)
    batch_size = 500  # Dit betekent 100 documenten per batch
    successful_docs = 0
    max_retries = 3
    failed_docs = []

    for i in range(0, len(actions), batch_size):
        batch = actions[i:i + batch_size]
        retry_count = 0
        batch_success = False
        
        while retry_count < max_retries and not batch_success:
            try:
                response = es.bulk(body=batch, refresh=True)
                if response["errors"]:
                    # Log details of failed operations
                    for item in response["items"]:
                        if "error" in item["index"]:
                            error_id = item["index"]["_id"]
                            error_msg = item["index"]["error"]
                            logger.error(f"Error indexing document {error_id}: {error_msg}")
                            failed_docs.append(error_id)
                else:
                    successful_docs += len(batch) // 2  # Deel door 2 omdat elke doc twee regels heeft
                    logger.info(f"Successfully indexed batch of {len(batch) // 2} documents ({successful_docs}/{len(documents)} total)")
                batch_success = True
                
            except Exception as e:
                retry_count += 1
                if "413 Request Entity Too Large" in str(e):
                    # Als we een 413 krijgen, probeer de batch te splitsen
                    if batch_size > 20:  # Niet kleiner dan 10 documenten (20 regels) maken
                        batch_size = max(20, batch_size // 2)
                        logger.warning(f"Request too large, reducing batch size to {batch_size} actions ({batch_size // 2} documents)")
                        # Herstart met deze batch met nieuwe grootte
                        i -= len(batch)
                        break
                    else:
                        logger.error("Batch size already at minimum, cannot reduce further")
                        # Voeg de gefaalde documenten toe aan de failed list
                        failed_docs.extend([action["index"]["_id"] for action in batch[::2]])  # Pak alleen de index acties
                        break
                elif retry_count == max_retries:
                    logger.error(f"Failed to index batch after {max_retries} retries: {e}")
                    # Voeg de gefaalde documenten toe aan de failed list
                    failed_docs.extend([action["index"]["_id"] for action in batch[::2]])  # Pak alleen de index acties
                else:
                    wait_time = 2 ** retry_count
                    logger.warning(f"Retry {retry_count}/{max_retries} after error: {e}. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)  

    # Final status report
    if failed_docs:
        logger.error(f"Failed to index {len(failed_docs)} documents: {failed_docs}")
    logger.info(f"Finished indexing: {successful_docs} successful, {len(failed_docs)} failed out of {len(documents)} total documents")

def search_and_print_results(es, index_name, query):
    result = es.search(index=index_name, body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["naam", "omschrijving", "classificatie", "archiefvormer", "aggregatieniveau", "full_text"],
                "type": "best_fields"
            }
        }
    })
    print(f"\nSearch for '{query}':")
    for hit in result['hits']['hits']:
        source = hit['_source']
        print(f"- {source['naam']} (Score: {hit['_score']})")
        if 'omschrijving' in source:
            print(f"  Omschrijving: {source['omschrijving']}")
        if 'classificatie' in source:
            print(f"  Classificatie: {source['classificatie']}")
        if 'aggregatieniveau' in source:
            print(f"  Aggregatieniveau: {source['aggregatieniveau']}")
        if 'archiefvormer' in source:
            print(f"  Archiefvormer: {source['archiefvormer']}")
        if 'archief' in source:
            print(f"  Archief: {source['archief']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LDTO to Elasticsearch converter')
    
    # Maak subparsers voor verschillende commando's
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Parser voor het convert commando
    convert_parser = subparsers.add_parser('convert', help='Convert TTL file to Elasticsearch')
    convert_parser.add_argument('--file', help='TTL file to convert')
    convert_parser.add_argument('--docs-dir', help='Directory containing the documents')
    convert_parser.add_argument('--index', default='ldto-objects', help='Elasticsearch index name')
    convert_parser.add_argument('--window-size', type=int, default=500, help='Size for max_inner_result_window')
    
    # Parser voor het update-window commando
    window_parser = subparsers.add_parser('update-window', help='Update max_inner_result_window setting')
    window_parser.add_argument('--index', default='ldto-objects', help='Elasticsearch index name')
    window_parser.add_argument('--window-size', type=int, default=500, help='Size for max_inner_result_window')
    
    # Voeg positional arguments toe voor backward compatibility
    parser.add_argument('ttl_file', nargs='?', help='Path to TTL file (legacy mode)')
    parser.add_argument('--docs-dir', default='docs', help='Directory containing the documents (legacy mode)')
    
    args = parser.parse_args()
    
    es = create_elasticsearch_client()
    
    # Als er een ttl_file is opgegeven zonder action, gebruik legacy mode
    if args.ttl_file and not args.action:
        documents, dekking_types, scheme_labels = convert_ttl_to_es(args.ttl_file, args.docs_dir)
        create_index(es, 'ldto-objects', dekking_types, scheme_labels)
        index_documents(es, 'ldto-objects', documents)
    # Anders gebruik de nieuwe command structuur
    elif args.action == 'convert':
        if not args.file:
            parser.error("--file is required for convert action")
        docs_dir = args.docs_dir if args.docs_dir else 'docs'
        documents, dekking_types, scheme_labels = convert_ttl_to_es(args.file, docs_dir)
        create_index(es, args.index, dekking_types, scheme_labels, window_size=args.window_size)
        index_documents(es, args.index, documents)
    elif args.action == 'update-window':
        update_inner_result_window(es, args.index, args.window_size)
