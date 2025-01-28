from rdflib import Graph, Namespace, URIRef
from elasticsearch import Elasticsearch
import os
import requests
from urllib.parse import urljoin
import logging
import time
from functools import lru_cache, wraps

# Define namespaces
LDTO = Namespace("https://data.razu.nl/def/ldto/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_last_segment(uri):
    """Extract the last segment of a URI."""
    return str(uri).split('/')[-1]

def normalize_label(label):
    """Normalize a label to be used as a field name"""
    return label.lower().replace(' ', '_')

@lru_cache(maxsize=1000)
def get_skos_label(uri):
    """Fetch SKOS prefLabel for a given URI. Results are cached using lru_cache."""
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
        
        # Find the prefLabel
        for _, _, label in temp_g.triples((URIRef(uri), SKOS.prefLabel, None)):
            return str(label)
            
        logger.warning(f"No prefLabel found for {uri}")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching skos:prefLabel for {uri}: {str(e)}")
        return None

def get_beperking_gebruik_labels(uri, g):
    """Get the scheme and label for a beperkingGebruik URI by dereferencing the URIs"""
    import requests
    import time
    
    logger.info(f"Dereferencing URI: {uri}")
    try:
        # Get the beperking type label and scheme
        response = requests.get(uri, headers={'Accept': 'text/turtle'})
        if response.status_code == 429:
            # Wait a bit and try again
            time.sleep(1)
            response = requests.get(uri, headers={'Accept': 'text/turtle'})
            
        if response.status_code != 200:
            logger.warning(f"Failed to dereference {uri}: {response.status_code}")
            return None
            
        # Parse the response into a temporary graph
        temp_g = Graph()
        temp_g.parse(data=response.text, format='turtle')
        
        # Get the prefLabel and inScheme
        type_label = None
        scheme_uri = None
        
        # Find the prefLabel for the type
        for _, _, label in temp_g.triples((URIRef(uri), SKOS.prefLabel, None)):
            type_label = str(label)
            break
            
        # Find the scheme URI
        for _, _, scheme in temp_g.triples((URIRef(uri), SKOS.inScheme, None)):
            scheme_uri = str(scheme)
            break
            
        if not type_label or not scheme_uri:
            logger.warning(f"No type_label or scheme found for {uri}")
            return None
            
        # Now dereference the scheme to get its label
        logger.info(f"Dereferencing scheme: {scheme_uri}")
        response = requests.get(scheme_uri, headers={'Accept': 'text/turtle'})
        if response.status_code == 429:
            # Wait a bit and try again
            time.sleep(1)
            response = requests.get(scheme_uri, headers={'Accept': 'text/turtle'})
            
        if response.status_code != 200:
            logger.warning(f"Failed to dereference scheme {scheme_uri}: {response.status_code}")
            return None
            
        # Parse the scheme response
        temp_g = Graph()
        temp_g.parse(data=response.text, format='turtle')
        
        # Get the scheme label
        scheme_label = None
        for _, _, label in temp_g.triples((URIRef(scheme_uri), SKOS.prefLabel, None)):
            scheme_label = str(label)
            break
            
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
        
    except Exception as e:
        logger.error(f"Error dereferencing {uri}: {e}")
        return None

def get_scheme_labels(g):
    """Get all unique scheme labels from beperkingGebruik URIs"""
    scheme_labels = {}
    
    # Find all beperkingGebruik URIs
    query = """
    SELECT DISTINCT ?beperking
    WHERE {
        ?s ldto:beperkingGebruik ?beperking .
    }
    """
    
    for row in g.query(query):
        beperking_uri = str(row.beperking)
        result = get_beperking_gebruik_labels(beperking_uri, g)
        if result and result['scheme_label']:
            scheme_labels[result['scheme']] = result['scheme_label']
    
    logger.info(f"Found scheme labels: {scheme_labels}")
    return scheme_labels

def get_dekking_types(g):
    """Get all unique dekkingInTijdType values and their SKOS labels"""
    query = """
    SELECT DISTINCT ?type
    WHERE {
        ?obj <https://data.razu.nl/def/ldto/dekkingInTijd> ?dekking .
        ?dekking <https://data.razu.nl/def/ldto/dekkingInTijdType> ?type .
    }
    """
    
    dekking_types = {}
    for row in g.query(query):
        type_uri = row[0]
        # Get the SKOS label for this type
        label = get_skos_label(type_uri)
        if label:
            # Convert label to lowercase with underscores
            field_name = normalize_label(label)
            dekking_types[str(type_uri)] = field_name
            logger.info(f"Found dekkingInTijdType: {label} ({type_uri})")
    
    return dekking_types

def create_index(es, index_name, dekking_types, scheme_labels):
    """Create Elasticsearch index with mapping for LDTO fields"""
    # Delete index if it exists
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        logger.info(f"Deleted existing index {index_name}")
        
    # Basic field mapping
    mapping = {
        "_source": {
            "excludes": ["full_text"]
        },
        "properties": {
            "id": {"type": "keyword"},
            "naam": {
                "type": "text",
                "analyzer": "dutch",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "omschrijving": {
                "type": "text",
                "analyzer": "dutch"
            },
            "trefwoorden": {
                "type": "text",
                "analyzer": "dutch",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "full_text": {
                "type": "text",
                "analyzer": "dutch",
                "store": False,
                "index_options": "docs",
                "norms": True
            },
            "bestand_url": {"type": "keyword"},
            "bestand_naam": {
                "type": "text",
                "analyzer": "dutch",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            # Metadata fields - only for faceting
            "classificatie": {
                "type": "keyword"
            },
            "classificatie_uri": {
                "type": "keyword",
                "index": False
            },
            "archiefvormer": {
                "type": "keyword"
            },
            "archiefvormer_uri": {
                "type": "keyword",
                "index": False
            },
            "aggregatieniveau": {
                "type": "keyword"
            },
            "aggregatieniveau_uri": {
                "type": "keyword",
                "index": False
            },
            # Hierarchical relationships - not searchable
            "bevat_onderdeel": {
                "type": "keyword",
                "index": False
            },
            "is_onderdeel_van": {
                "type": "keyword",
                "index": False
            }
        }
    }
    
    # Add date range fields for each dekking type
    for dekking_type, field_name in dekking_types.items():
        mapping["properties"][f"{field_name}_range"] = {
            "type": "date_range",
            "format": "yyyy-MM-dd||yyyy"
        }
        
    # Add fields for each scheme
    for scheme_uri, scheme_label in scheme_labels.items():
        mapping["properties"][scheme_label] = {
            "type": "keyword"
        }
        
    # Create index with mapping
    es.indices.create(
        index=index_name,
        mappings=mapping,
        settings={
            "analysis": {
                "analyzer": {
                    "dutch": {
                        "tokenizer": "standard",
                        "filter": ["lowercase", "dutch_stop", "dutch_stemmer"]
                    }
                },
                "filter": {
                    "dutch_stop": {
                        "type": "stop",
                        "stopwords": "_dutch_"
                    },
                    "dutch_stemmer": {
                        "type": "stemmer",
                        "language": "dutch"
                    }
                }
            }
        }
    )
    logger.info(f"Created index {index_name} with facet mapping")

def cache_uri_response(func):
    """Cache responses from URI dereferencing"""
    cache = {}
    
    @wraps(func)
    def wrapper(uri, g):
        if uri in cache:
            logger.info(f"Cache hit for {uri}")
            return cache[uri]
        result = func(uri, g)
        cache[uri] = result
        return result
    return wrapper

@cache_uri_response
def get_beperking_gebruik_labels(uri, g):
    """Get the scheme and label for a beperkingGebruik URI by dereferencing the URIs"""
    import requests
    import time
    
    logger.info(f"Dereferencing URI: {uri}")
    try:
        # Get the beperking type label and scheme
        response = requests.get(uri, headers={'Accept': 'text/turtle'})
        if response.status_code == 429:
            # Wait a bit and try again
            time.sleep(1)
            response = requests.get(uri, headers={'Accept': 'text/turtle'})
            
        if response.status_code != 200:
            logger.warning(f"Failed to dereference {uri}: {response.status_code}")
            return None
            
        # Parse the response into a temporary graph
        temp_g = Graph()
        temp_g.parse(data=response.text, format='turtle')
        
        # Get the prefLabel and inScheme
        type_label = None
        scheme_uri = None
        
        # Find the prefLabel for the type
        for _, _, label in temp_g.triples((URIRef(uri), SKOS.prefLabel, None)):
            type_label = str(label)
            break
            
        # Find the scheme URI
        for _, _, scheme in temp_g.triples((URIRef(uri), SKOS.inScheme, None)):
            scheme_uri = str(scheme)
            break
            
        if not type_label or not scheme_uri:
            logger.warning(f"No type_label or scheme found for {uri}")
            return None
            
        # Now dereference the scheme to get its label
        logger.info(f"Dereferencing scheme: {scheme_uri}")
        response = requests.get(scheme_uri, headers={'Accept': 'text/turtle'})
        if response.status_code == 429:
            # Wait a bit and try again
            time.sleep(1)
            response = requests.get(scheme_uri, headers={'Accept': 'text/turtle'})
            
        if response.status_code != 200:
            logger.warning(f"Failed to dereference scheme {scheme_uri}: {response.status_code}")
            return None
            
        # Parse the scheme response
        temp_g = Graph()
        temp_g.parse(data=response.text, format='turtle')
        
        # Get the scheme label
        scheme_label = None
        for _, _, label in temp_g.triples((URIRef(scheme_uri), SKOS.prefLabel, None)):
            scheme_label = str(label)
            break
            
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
        
    except Exception as e:
        logger.error(f"Error dereferencing {uri}: {e}")
        return None

def convert_ttl_to_es(ttl_file):
    # Create RDF graph and parse TTL file
    g = Graph()
    g.parse(ttl_file, format='turtle')
    
    # First get all scheme labels
    scheme_labels = get_scheme_labels(g)
    
    # Get all dekking types
    dekking_types = get_dekking_types(g)
    
    # Find all ldto:Informatieobject instances with their properties
    query = """
    SELECT DISTINCT ?obj ?naam ?omschrijving ?classificatie ?archiefvormer ?aggregatieniveau 
           ?dekking ?dekkingType ?begin ?eind
           ?bestand ?bestandURL ?bestandNaam ?beperkingen
           ?bevatOnderdeel ?isOnderdeelVan
           (GROUP_CONCAT(?trefwoord; separator=',') as ?trefwoorden)
    WHERE {
        ?obj a ldto:Informatieobject ;
             ldto:naam ?naam ;
             ldto:classificatie ?classificatie ;
             ldto:archiefvormer ?archiefvormer ;
             ldto:aggregatieniveau ?aggregatieniveau .
        OPTIONAL { ?obj ldto:omschrijving ?omschrijving }
        OPTIONAL { 
            ?obj ldto:dekkingInTijd ?dekking .
            ?dekking ldto:dekkingInTijdType ?dekkingType .
            OPTIONAL { ?dekking ldto:begin ?begin }
            OPTIONAL { ?dekking ldto:eind ?eind }
        }
        OPTIONAL { 
            ?obj ldto:bestand ?bestand .
            ?bestand ldto:url ?bestandURL ;
                     ldto:naam ?bestandNaam .
        }
        OPTIONAL { ?obj ldto:beperkingGebruik ?beperkingen }
        OPTIONAL { ?obj ldto:bevatOnderdeel ?bevatOnderdeel }
        OPTIONAL { ?obj ldto:isOnderdeelVan ?isOnderdeelVan }
        OPTIONAL { ?obj ldto:trefwoord ?trefwoord }
    }
    GROUP BY ?obj ?naam ?omschrijving ?classificatie ?archiefvormer ?aggregatieniveau 
             ?dekking ?dekkingType ?begin ?eind
             ?bestand ?bestandURL ?bestandNaam ?beperkingen
             ?bevatOnderdeel ?isOnderdeelVan
    """
    
    documents = {}
    
    # Dummy full text for testing
    dummy_text = "Wijk bij Duurstede is een vestingstad en gemeente in het zuiden van de Nederlandse provincie Utrecht."
    
    for row in g.query(query, initNs={'ldto': LDTO}):
        obj, naam, omschrijving, classificatie, archiefvormer, aggregatieniveau, dekking, dekkingType, begin, eind, bestand, bestandURL, bestandNaam, beperkingen, bevatOnderdeel, isOnderdeelVan, trefwoorden = row
        
        doc_id = extract_last_segment(obj)
        doc = {
            'id': doc_id,
            'naam': str(naam),
            'full_text': dummy_text
        }
        
        # Add optional fields if present
        if omschrijving:
            doc['omschrijving'] = str(omschrijving)
            
        if classificatie:
            classificatie_label = get_skos_label(classificatie)
            if classificatie_label:
                doc['classificatie'] = classificatie_label
                doc['classificatie_uri'] = str(classificatie)
            
        if archiefvormer:
            archiefvormer_label = get_skos_label(archiefvormer)
            if archiefvormer_label:
                doc['archiefvormer'] = archiefvormer_label
                doc['archiefvormer_uri'] = str(archiefvormer)
            
        if aggregatieniveau:
            aggregatieniveau_label = get_skos_label(aggregatieniveau)
            if aggregatieniveau_label:
                doc['aggregatieniveau'] = aggregatieniveau_label
                doc['aggregatieniveau_uri'] = str(aggregatieniveau)
        
        # Process bestand if present
        if bestand and bestandURL and bestandNaam:
            doc['bestand_url'] = str(bestandURL)
            doc['bestand_naam'] = str(bestandNaam)
            
        # Add hierarchical relationships if present
        if bevatOnderdeel:
            doc['bevat_onderdeel'] = extract_last_segment(bevatOnderdeel)
            
        if isOnderdeelVan:
            doc['is_onderdeel_van'] = extract_last_segment(isOnderdeelVan)

        # Add keywords if present
        if trefwoorden:
            doc['trefwoorden'] = [kw.strip() for kw in str(trefwoorden).split(',')]

        # Store document
        documents[doc_id] = doc
        
    # Process beperkingGebruik for each document
    for doc_id, doc in documents.items():
        # Get all beperkingGebruik values for this document
        query = f"""
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
                result = get_beperking_gebruik_labels(beperking_uri, g)
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

def index_documents(documents, index_name='ldto-objects'):
    # Initialize Elasticsearch client with authentication
    es = Elasticsearch(
        "https://es.digitopia.nl",
        basic_auth=("elastic", "search"),
        verify_certs=False  # Note: In production, you should verify SSL certificates
    )
    
    # Create index with proper mapping
    dekking_types = documents[1]
    scheme_labels = documents[2]
    documents = documents[0]
    create_index(es, index_name, dekking_types, scheme_labels)
    
    # Create or update documents
    for doc_id, doc in documents.items():
        es.index(
            index=index_name,
            id=doc_id,
            document=doc
        )
        logger.info(f"Indexed document {doc_id}")
    
    # Refresh index to make documents immediately available
    es.indices.refresh(index=index_name)
    logger.info("Refreshed index")
    
    # Show example searches and facets
    print("\nTesting searches:")
    
    # Test searching for 'vergaderstuk'
    print("\nSearch for 'vergaderstuk':")
    search_and_print_results(es, index_name, "vergaderstuk")
    
    # Test case-insensitive search
    print("\nSearch for 'VERGADERSTUK':")
    search_and_print_results(es, index_name, "VERGADERSTUK")
    
    # Test partial word search
    print("\nSearch for 'Bestuurs':")
    search_and_print_results(es, index_name, "Bestuurs")
    
    # Test searching with underscore
    print("\nSearch for 'Bijlage_1':")
    search_and_print_results(es, index_name, "Bijlage_1")
    
    print("\nSearch for 'Bijlage 1':")
    search_and_print_results(es, index_name, "Bijlage 1")

    # Example date range query
    print("\nExample date range query:")
    date_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "van_toepassing_range": {
                                "gte": "2010",
                                "lte": "2024",
                                "relation": "intersects"
                            }
                        }
                    }
                ]
            }
        }
    }
    result = es.search(index=index_name, body=date_query)
    print(f"\nDocuments between 2010-2024:")
    for hit in result['hits']['hits']:
        source = hit['_source']
        print(f"- {source['naam']}")
        if 'van_toepassing_range' in source:
            print(f"  Period: {source['van_toepassing_range']['gte']} - {source['van_toepassing_range']['lte']}")

def search_and_print_results(es, index_name, query):
    result = es.search(index=index_name, body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["naam", "omschrijving", "classificatie", "archiefvormer", "aggregatieniveau", "trefwoorden", "full_text"],
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
        if 'archiefvormer' in source:
            print(f"  Archiefvormer: {source['archiefvormer']}")
        if 'aggregatieniveau' in source:
            print(f"  Aggregatieniveau: {source['aggregatieniveau']}")
        if 'trefwoorden' in source:
            print(f"  Trefwoorden: {source['trefwoorden']}")

if __name__ == "__main__":
    # Convert example_1.ttl
    ttl_file = "source/example_1.ttl"
    documents, dekking_types, scheme_labels = convert_ttl_to_es(ttl_file)
    
    # Print results for verification
    print(f"\nFound {len(documents)} Informatieobject instances:")
    for doc_id, doc in documents.items():
        print(f"\nID: {doc_id}")
        print(f"Naam: {doc['naam']}")
        if 'omschrijving' in doc:
            print(f"Omschrijving: {doc['omschrijving']}")
        if 'classificatie' in doc:
            print(f"Classificatie: {doc['classificatie']}")
        if 'archiefvormer' in doc:
            print(f"Archiefvormer: {doc['archiefvormer']}")
        if 'aggregatieniveau' in doc:
            print(f"Aggregatieniveau: {doc['aggregatieniveau']}")
        if 'bestand_url' in doc:
            print(f"Bestand URL: {doc['bestand_url']}")
        if 'bestand_naam' in doc:
            print(f"Bestand Naam: {doc['bestand_naam']}")
        if 'bevat_onderdeel' in doc:
            print(f"Bevat onderdeel: {doc['bevat_onderdeel']}")
        if 'is_onderdeel_van' in doc:
            print(f"Is onderdeel van: {doc['is_onderdeel_van']}")
        if 'trefwoorden' in doc:
            print(f"Trefwoorden: {doc['trefwoorden']}")

    # Index in Elasticsearch
    index_documents((documents, dekking_types, scheme_labels))
