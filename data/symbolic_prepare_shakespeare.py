import os
import pickle
import subprocess
import spacy
import numpy as np
from tqdm import tqdm
from spacy.tokens import Span, Doc


class OfflineWikidataKnowledgeGraph:
    """
    Interface to an offline Wikidata subset (TSV format).
    """
    def __init__(self, tsv_path):
        self.data = {}  # Store triples: {subject: {predicate: [objects]}}
        self.concept_ids = set() #Keep track of all concepts
        self.relation_ids = set()
        self.load_tsv(tsv_path)


    def load_tsv(self, tsv_path):
        """Loads the Wikidata subset from a TSV file."""
        print(f"Loading Wikidata subset from {tsv_path}...")
        # Check if file exists and is not empty
        if not os.path.exists(tsv_path) or os.stat(tsv_path).st_size == 0:
            print(f"Error: TSV file is empty or does not exist: {tsv_path}")
            raise FileNotFoundError(f"TSV file is empty or missing: {tsv_path}")


        with open(tsv_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header line (subject predicate object)
            for line in tqdm(f, desc="Loading TSV"):
                try:
                    subject, predicate, object_ = line.strip().split('\t')
                    self.concept_ids.add(subject)
                    self.relation_ids.add(predicate)
                    self.concept_ids.add(object_)

                    if subject not in self.data:
                        self.data[subject] = {}
                    if predicate not in self.data[subject]:
                        self.data[subject][predicate] = []
                    self.data[subject][predicate].append(object_)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}") #Handle lines that do not have 3 values.

        print(f"Loaded {len(self.data)} subjects, {len(self.concept_ids)} unique concepts, and {len(self.relation_ids)} relations.")

    def get_concept_id(self, concept_name):
        """
        Placeholder: Returns the QID.
        """
        return concept_name  # Return the QID itself

    def has_relation(self, concept_id1, concept_id2, relation_id=None):
        """Checks if a relation exists between two concepts."""
        if concept_id1 not in self.data:
            return False
        if relation_id is None:
            # Check for *any* relation
            for rel, objects in self.data[concept_id1].items():
                if concept_id2 in objects:
                    return True
            return False
        else:
            #Check for the specific relation.
            return (relation_id in self.data[concept_id1] and
                    concept_id2 in self.data[concept_id1][relation_id])

    def get_related_concepts(self, concept_id, relation_id=None, limit=None):
        """Gets concepts related to a given concept."""
        related_concepts = []
        if concept_id in self.data:
            if relation_id:
                if relation_id in self.data[concept_id]:
                    related_concepts.extend(self.data[concept_id][relation_id])
            else:
                # Get all related concepts
                for rel, objects in self.data[concept_id].items():
                    related_concepts.extend(objects)
        if limit:
            return related_concepts[:limit]
        else:
            return related_concepts

    def get_relation_id(self, relation_name):
        """
        Placeholder
        """
        return relation_name



def create_wikidata_subset(dump_file, output_tsv, qid_file, java_jar_path):
    """
    Creates the Wikidata subset using the compiled Java program.
    """
    command = [
        "java",
        "-jar",  # Use -jar to execute the JAR file
        java_jar_path,
        dump_file,
        output_tsv,
        qid_file,
    ]
    print(f"Running Java command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running Java command: {e}")
        print(e.stderr)
        exit(1)
    except FileNotFoundError:
        print("Error: Java not found.  Make sure Java is installed and in your PATH.")
        exit(1)

def process_text_chunked(nlp, text, chunk_size=100000):
    """
    Processes text in chunks.
    """
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield nlp(chunk)

def prepare_symbolic_data(input_file, output_dir, tsv_path, wikidata_dump_path, dataset_name="symbolic_shakespeare"):
    qid_file = os.path.join(output_dir, "initial_qids.txt")
    # Corrected path to the JAR file created by Maven
    java_jar_path = os.path.join("wikidata", "wikidata-subset-creator.jar")

    # --- 0. Create Wikidata Subset (if it doesn't exist) ---
    if not os.path.exists(tsv_path):
        print(f"Wikidata subset file not found: {tsv_path}")
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
                    initial_qids.add("Q" + ent.text.replace(" ", "_").replace("'", "")) #Basic QID creation

        with open(qid_file, "w", encoding="utf-8") as f:
            for qid in initial_qids:
                f.write(qid + "\n")
        print(f"Initial QIDs written to {qid_file}")

        create_wikidata_subset(wikidata_dump_path, tsv_path, qid_file, java_jar_path)


    # --- 1. Load Offline Wikidata Knowledge Graph ---
    try:
        knowledge_graph = OfflineWikidataKnowledgeGraph(tsv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}.  Could not load Wikidata subset. Please check the file path and ensure the subset creation was successful.")
        exit(1)

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

if __name__ == '__main__':
    input_file = 'data/shakespeare/input.txt'
    output_dir = 'data/symbolic_shakespeare'
    tsv_path = os.path.join(output_dir, 'wikidata_subset.tsv')  # Output TSV
    wikidata_dump_path = 'wikidata/latest-truthy.nt.bz2' #Path

    os.makedirs(output_dir, exist_ok=True)
    prepare_symbolic_data(input_file, output_dir, tsv_path, wikidata_dump_path, dataset_name="symbolic_shakespeare")
