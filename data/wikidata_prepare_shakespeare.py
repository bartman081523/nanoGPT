import os
import pickle
import ijson  # For streaming JSON parsing
import rdflib  # For N-Triples parsing
import spacy
import numpy as np
from tqdm import tqdm
import requests  # For Wikidata SPARQL queries
import bz2 #For handling bzip2 compression

def get_concept_qid(concept_name):
    """
    Queries Wikidata SPARQL endpoint to get QID from a label.
    """
    service = "https://query.wikidata.org/sparql"
    query = f"""
        SELECT ?concept WHERE {{
            ?concept rdfs:label "{concept_name}"@en .
        }} LIMIT 1
    """
    try:
        response = requests.get(service, params={'query': query, 'format': 'json'}, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            concept_uri = bindings[0]["concept"]["value"]
            return concept_uri.split("/")[-1]  # Extract QID
        return None
    except requests.RequestException as e:
        print(f"Error querying Wikidata for '{concept_name}': {e}")
        return None

def process_json_dump(dump_file, output_file, target_concepts):
    """
    Processes a Wikidata JSON dump file, extracting triples and writing to a TSV.
    """
    print(f"Processing JSON dump: {dump_file}")
    triple_count = 0

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write("subject\tpredicate\tobject\n")  # TSV header

        if dump_file.endswith(".bz2"):
            file_open_func = bz2.open
        else:
            file_open_func = open

        with file_open_func(dump_file, "rb") as f:  # Open as binary for ijson
            # Use ijson.items with a prefix that gets us to each item document
            for item in ijson.items(f, "item"):
                if not isinstance(item, dict) or "id" not in item:
                    continue  # Skip malformed items

                subject_qid = item["id"]

                if target_concepts and subject_qid not in target_concepts:
                    continue

                claims = item.get("claims", {})
                for predicate, statements in claims.items():
                    if not predicate.startswith("P"):
                        continue  # Skip non-property claims
                    for statement in statements:
                        mainsnak = statement.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            object_value = datavalue.get("value", {})
                            object_qid = object_value.get("id")
                            if object_qid and object_qid.startswith("Q"):
                                out_file.write(f"{subject_qid}\t{predicate}\t{object_qid}\n")
                                triple_count += 1
                #Also add qids from labels
                if "labels" in item:
                    for lang, label_data in item["labels"].items():
                        label_text = label_data.get("value")
                        if label_text:
                            new_qid = get_concept_qid(label_text)
                            if new_qid:
                                target_concepts.add(new_qid)

    print(f"Finished processing JSON.  Wrote {triple_count} triples.")


def process_nt_dump(dump_file, output_file, target_concepts):
    """
    Processes an N-Triples dump file, extracting triples and writing to a TSV.
    """
    print(f"Processing N-Triples dump: {dump_file}")
    triple_count = 0

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write("subject\tpredicate\tobject\n")

        if dump_file.endswith(".bz2"):
            file_open_func = bz2.open
        else:
            file_open_func = open

        with file_open_func(dump_file, "rt", encoding="utf-8") as f:  # Open as text
            # Iterate through lines (each line is an N-Triple)
            for line in tqdm(f, desc="Processing N-Triples", unit="triple"):
                line = line.strip()
                if not line or line.startswith("#"):  # Skip comments and empty lines
                    continue

                try:
                    # Basic N-Triples parsing (splitting on spaces, handling URIs)
                    parts = line.split(" ", 2)  # Split into 3 parts (max)
                    if len(parts) != 3:
                        continue  # Skip malformed lines
                    subject_uri = parts[0][1:-1]  # Remove < >
                    predicate_uri = parts[1][1:-1]
                    object_part = parts[2]

                    # Extract QID/PID from URIs
                    subject_qid = subject_uri.split("/")[-1]

                    if target_concepts and subject_qid not in target_concepts:
                        continue #Skip if qid not in our target

                    predicate_pid = predicate_uri.split("/")[-1]

                    if not predicate_pid.startswith("P"):
                        continue

                    # Handle object (which could be a URI or a literal)
                    if object_part.startswith("<"):  # URI
                        object_qid = object_part[1:-1].split("/")[-1] # Remove <>
                        if object_qid.startswith("Q"): #Only save Q-number objects
                            out_file.write(f"{subject_qid}\t{predicate_pid}\t{object_qid}\n")
                            triple_count += 1

                    # We don't process literal values.
                except Exception as e:
                    print(f"Error processing line: {line} - {e}") #Print the exception
                    continue

    print(f"Finished processing N-Triples. Wrote {triple_count} triples.")




def prepare_symbolic_data(input_file, output_dir, wikidata_dump_path, dataset_name="symbolic_shakespeare"):

    # --- 0. Create Wikidata Subset (if it doesn't exist) ---
    tsv_path = os.path.join(output_dir, 'wikidata_subset.tsv')
    if not os.path.exists(tsv_path):
        print(f"Wikidata subset file not found: {tsv_path}")

        #Initial QID file:
        qid_file = os.path.join(output_dir, "initial_qids.txt")

        # Initial QID extraction using spaCy (before KG subset exists)
        print("Performing initial NER with spaCy to generate qids.txt...")
        try:
            # Disable unnecessary components for initial QID extraction
            nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"])
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download("en_core_web_lg")
            nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"])

        nlp.max_length = 2000000  # Increase max_length temporarily

        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        initial_qids = set()
        # Process in chunks for initial QID extraction
        for doc in process_text_chunked(nlp, text):
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "NORP"):
                    initial_qids.add("Q" + ent.text.replace(" ", "_").replace("'", ""))  # Basic QID creation

        with open(qid_file, "w", encoding="utf-8") as f:
            for qid in initial_qids:
                f.write(qid + "\n")
        print(f"Initial QIDs written to {qid_file}")


        # Determine dump file type and process accordingly
        if wikidata_dump_path.endswith(('.json', '.json.bz2')):
            process_json_dump(wikidata_dump_path, tsv_path, initial_qids)
        elif wikidata_dump_path.endswith(('.nt', '.nt.bz2')):
            process_nt_dump(wikidata_dump_path, tsv_path, initial_qids)
        else:
            print(f"Unsupported dump file type: {wikidata_dump_path}")
            exit(1)

    # --- 1. Load Offline Wikidata Knowledge Graph ---
    print(f"Loading Wikidata subset from {tsv_path}...")
    knowledge_graph = OfflineWikidataKnowledgeGraph(tsv_path) #Use our KG class

    # --- 2. Load spaCy Model (and download if necessary) ---
    try:
        nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"])
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"])

    nlp.max_length = 2000000

    # --- 3. Load and Process Text ---
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- 4 & 5. Entity Linking and Relation Extraction (Chunked) ---
    print("Performing entity recognition and relation extraction...")
    concept_ids = []
    relation_ids = set()
    for doc in tqdm(process_text_chunked(nlp, text), desc="Processing chunks"):
        sentence_qids = []
        for ent in doc.ents:
             if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "NORP"):
                qid = knowledge_graph.get_concept_id(ent.text) #Lookup in the offline KG
                if qid:
                    sentence_qids.append(qid)

        for i in range(len(sentence_qids) - 1):
            qid1 = sentence_qids[i]
            qid2 = sentence_qids[i+1]
            if knowledge_graph.has_relation(qid1, qid2):
                for rel_id in knowledge_graph.data.get(qid1, {}).keys():
                    if knowledge_graph.has_relation(qid1, qid2, rel_id):
                        relation_ids.add(rel_id)
                        break

        concept_ids.extend(sentence_qids)


    # --- 6. Create Integer Mappings (stoi, itos) ---
    stoi_concepts = {cid: i for i, cid in enumerate(sorted(list(knowledge_graph.concept_ids)))}
    itos_concepts = {i: cid for cid, i in stoi_concepts.items()}

    stoi_relations = {rid: i for i, rid in enumerate(sorted(list(knowledge_graph.relation_ids)))}
    itos_relations = {i: rid for rid, i in stoi_relations.items()}

    # --- 7. Encode Data ---
    encoded_concept_ids = [stoi_concepts[cid] for cid in concept_ids]

    # --- 8. Split into Train/Val ---
    n = len(encoded_concept_ids)
    train_data = encoded_concept_ids[:int(n*0.9)]
    val_data = encoded_concept_ids[int(n*0.9):]

    # --- 9. Export to Binary Files ---
    train_ids = np.array(train_data, dtype=np.uint16)  # Or uint32
    val_ids = np.array(val_data, dtype=np.uint16)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    # --- 10. Save Metadata ---
    meta = {
        'vocab_size': len(stoi_concepts),
        'stoi': stoi_concepts,
        'itos': itos_concepts,
        'relation_stoi': stoi_relations,
        'relation_itos': itos_relations,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Processed {len(concept_ids)} concept occurrences.")
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Saved metadata to {os.path.join(output_dir, 'meta.pkl')}")



def process_text_chunked(nlp, text, chunk_size=100000):
    """
    Processes text in chunks using spaCy, yielding Doc objects for each chunk.

    Args:
        nlp: The loaded spaCy language model.
        text: The full text to process.
        chunk_size: The size of each chunk (in characters).

    Yields:
        spaCy Doc objects for each processed chunk.
    """
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield nlp(chunk)

if __name__ == '__main__':
    input_file = 'data/shakespeare/input.txt'
    output_dir = 'data/symbolic_shakespeare'
    wikidata_dump_path = 'wikidata/latest-truthy.nt.bz2' #Path

    os.makedirs(output_dir, exist_ok=True)
    prepare_symbolic_data(input_file, output_dir,  wikidata_dump_path, dataset_name="symbolic_shakespeare") #tsv_path removed
