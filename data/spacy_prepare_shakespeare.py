import os
import pickle
import spacy
import numpy as np
from tqdm import tqdm

def prepare_symbolic_data(input_file, output_dir, dataset_name="symbolic_spacy"):
    """
    Prepares symbolic data using spaCy for entity recognition.
    Dynamically builds a vocabulary of entities and relations.
    """
    print(f"Preparing data from {input_file}...")

    try:
        nlp = spacy.load("en_core_web_lg", disable=["tagger", "attribute_ruler", "lemmatizer"])
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg", disable=["tagger", "attribute_ruler", "lemmatizer"])
    nlp.max_length = 2000000

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # --- Entity and Relation Extraction ---
    print("Performing entity recognition and relation extraction...")
    entities = []       # List of entity strings
    entity_map = {}     # Keep track of unique entities: {entity_str: index}
    relations = []      # List of (subj_idx, relation_str, obj_idx)
    relation_map = {}   # Keep track of unique relations: {relation_str: index}

    next_entity_index = 0
    next_relation_index = 0

    for doc in process_text_chunked(nlp, text): #Using the same chunking as before
        for sent in doc.sents: #Process per sentence
            sentence_entities = []
            for ent in sent.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "NORP"):
                    ent_text = ent.text.strip()
                    if ent_text not in entity_map:
                        entity_map[ent_text] = next_entity_index
                        entities.append(ent_text)
                        next_entity_index += 1
                    sentence_entities.append(entity_map[ent_text])

            # Very basic relation extraction: adjacent entities
            for i in range(len(sentence_entities) - 1):
                subj_idx = sentence_entities[i]
                obj_idx = sentence_entities[i+1]
                # Create a simple relation string (can be improved)
                relation_str = f"RELATED_TO_{i}"  # Placeholder relation
                if relation_str not in relation_map:
                    relation_map[relation_str] = next_relation_index
                    next_relation_index += 1
                relations.append((subj_idx, relation_map[relation_str], obj_idx))


    print(f"Extracted {len(entities)} unique entities and {len(relation_map)} unique relations.")

    # --- Create Integer Mappings (stoi, itos) ---
    stoi_concepts = {ent: i for i, ent in enumerate(entities)}
    itos_concepts = {i: ent for ent, i in stoi_concepts.items()}
    stoi_relations = {rel: i for i, rel in enumerate(relation_map)}
    itos_relations = {i: rel for rel, i in stoi_relations.items()}

    # --- Encode Data ---
    # Now, instead of storing QIDs, we store *indices* into our entity list.
    encoded_concept_ids = []  # List of *indices* into the entities list
    for doc in process_text_chunked(nlp, text):
        for sent in doc.sents:
            for ent in sent.ents:
                 if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "NORP"):
                    ent_text = ent.text.strip()
                    encoded_concept_ids.append(stoi_concepts[ent_text])


    # --- Split into Train/Val ---
    n = len(encoded_concept_ids)
    train_data = encoded_concept_ids[:int(n * 0.9)]
    val_data = encoded_concept_ids[int(n * 0.9):]

    # --- Export to Binary Files ---
    train_ids = np.array(train_data, dtype=np.uint32)  # Use uint32 for larger vocabularies
    val_ids = np.array(val_data, dtype=np.uint32)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    # --- Save Metadata ---
    meta = {
        'vocab_size': len(stoi_concepts),  # Number of unique entities
        'relation_vocab_size': len(stoi_relations), #Number of relations
        'stoi': stoi_concepts,
        'itos': itos_concepts,
        'relation_stoi': stoi_relations,
        'relation_itos': itos_relations,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"Processed {len(encoded_concept_ids)} concept occurrences.")
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Saved metadata to {os.path.join(output_dir, 'meta.pkl')}")

def process_text_chunked(nlp, text, chunk_size=100000):
    """
    Processes text in chunks using spaCy, yielding Doc objects for each chunk.
    """
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield nlp(chunk)

if __name__ == '__main__':
    input_file = 'data/shakespeare/input.txt'
    output_dir = 'data/symbolic_shakespeare'  # Use a different output directory
    os.makedirs(output_dir, exist_ok=True)
    prepare_symbolic_data(input_file, output_dir, dataset_name="symbolic_shakespeare")
