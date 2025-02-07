#!/usr/bin/env python3
"""
wikidata_subset_creator.py

Usage:
    python wikidata_subset_creator.py <dump_file> <output_file> <qid_file>

- <dump_file> can be:
  - *.json, *.json.bz2 (Wikidata JSON dump)
  - *.nt, *.nt.bz2     (N-Triples)

- <output_file> is the TSV to write:
  subject  predicate  object

- <qid_file> is a text file with QIDs, one per line. If non-empty, we keep only
  those triples whose subject is in the set. If it's empty, we keep all.

Additionally, if a subject is in targetConcepts, we can also try to discover
QIDs from each label by querying WDQS (like the Java code).

You need these packages:
  ijson, rdflib, requests

Example:
    python wikidata_subset_creator.py dump.json.bz2 out.tsv qids.txt
"""

import sys
import bz2
import json
import ijson
import requests
from rdflib import Graph
from rdflib.plugins.parsers.ntriples import NTriplesParser

def main():
    if len(sys.argv) != 4:
        print("Usage: python wikidata_subset_creator.py <dumpFile> <outputFile> <qidFile>")
        sys.exit(1)

    dump_file = sys.argv[1]
    output_file = sys.argv[2]
    qid_file = sys.argv[3]

    # 1) Read the QIDs
    target_concepts = set()
    with open(qid_file, 'r', encoding='utf-8') as f:
        for line in f:
            qid = line.strip()
            if qid:
                target_concepts.add(qid)

    # 2) Dispatch
    if dump_file.endswith(".json") or dump_file.endswith(".json.bz2"):
        process_json_dump(dump_file, output_file, target_concepts)
    elif dump_file.endswith(".nt") or dump_file.endswith(".nt.bz2"):
        process_nt_dump(dump_file, output_file, target_concepts)
    else:
        print("Unsupported format. Must be *.json(.bz2) or *.nt(.bz2).")
        sys.exit(1)

def process_json_dump(dump_file, output_file, target_concepts):
    """
    Parse a Wikidata JSON dump, extracting triple-like edges (Qxxx Pyyy Qzzz).
    We'll do a streaming parse with ijson. Each itemDocument is an entity with
    an 'id' like 'Q31' and 'claims', etc.

    Because the official Wikidata JSON dump is huge, we do not want to load
    it all into memory at once. ijson allows for streaming parse.
    """
    print(f"Processing JSON dump: {dump_file}")

    out = open(output_file, 'w', encoding='utf-8')
    out.write("subject\tpredicate\tobject\n")
    triple_count = 0

    # The official Wikidata JSON dump has each entity as a separate line after the first lines
    # But ijson usually expects an array. We'll handle the "wikibase" format:
    # if it's line-based, we can parse them one by one.

    # If it's compressed:
    if dump_file.endswith(".bz2"):
        f_in = bz2.open(dump_file, 'rb')
    else:
        f_in = open(dump_file, 'rb')

    # The Wikidata JSON dump can come in two forms:
    #  - "full" JSON array of entity objects
    #  - "line-based" JSON with one entity per line
    # We'll attempt the "line-based" approach:
    #    Each line is a full JSON object with "id", "claims", etc.
    #    except possibly the first/last lines with '[' or ']'
    #
    # We'll read line by line, parse only valid JSON objects:
    for line in f_in:
        line = line.strip()
        if not line or line.startswith(b"[") or line.startswith(b"]"):
            continue
        # If there's a trailing comma, remove it
        if line.endswith(b","):
            line = line[:-1]
        try:
            entity = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "id" not in entity:
            # Possibly some meta structure
            continue

        subject_qid = entity["id"]
        if (not target_concepts) or (subject_qid in target_concepts):
            # parse statements
            if "claims" in entity:
                claims = entity["claims"]
                for prop, statements in claims.items():
                    if not prop.startswith("P"):
                        continue
                    # statements is a list of dicts
                    for st in statements:
                        mainsnak = st.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            # e.g. {"value": {"id": "Qxx", ...}}
                            value_obj = datavalue.get("value", {})
                            object_qid = value_obj.get("id")
                            if object_qid and object_qid.startswith("Q"):
                                out.write(f"{subject_qid}\t{prop}\t{object_qid}\n")
                                triple_count += 1

            # Optionally, do label-based QID expansions like the Java getConceptQid
            if subject_qid in target_concepts and "labels" in entity:
                for lang_code, label_info in entity["labels"].items():
                    label_text = label_info.get("value")
                    if label_text:
                        discovered_qid = get_concept_qid(label_text)
                        if discovered_qid:
                            target_concepts.add(discovered_qid)

    f_in.close()
    out.close()
    print(f"Done with JSON. Wrote {triple_count} triples to {output_file}.")

def process_nt_dump(dump_file, output_file, target_concepts):
    """
    Parse an N-Triples dump using RDFLib or a simpler parser,
    then write out subject, predicate, object if subject is in target_concepts.
    """
    print(f"Processing N-Triples dump: {dump_file}")

    out = open(output_file, 'w', encoding='utf-8')
    out.write("subject\tpredicate\tobject\n")
    triple_count = 0

    # We'll parse with rdflib's Graph
    # But to do it in a streaming fashion, let's use the NTriplesParser directly.
    parser = NTriplesParser()

    # We'll define a local handler
    def handle_statement(s, p, o):
        nonlocal triple_count
        # Typical subject URI: "http://www.wikidata.org/entity/Qxxx"
        subj_str = str(s)
        qid = parse_qid_from_uri(subj_str)
        if not target_concepts or (qid and qid in target_concepts):
            out.write(f"{qid}\t{p}\t{o}\n")
            triple_count += 1

    class MySink:
        def triple(self, s, p, o):
            handle_statement(s, p, o)

    sink = MySink()
    parser.sink = sink

    if dump_file.endswith(".bz2"):
        f_in = bz2.open(dump_file, 'rb')
    else:
        f_in = open(dump_file, 'rb')

    parser.parse(f_in)
    f_in.close()
    out.close()

    print(f"Done with NT. Wrote {triple_count} triples to {output_file}.")

def parse_qid_from_uri(uri):
    prefix = "http://www.wikidata.org/entity/"
    if uri.startswith(prefix):
        return uri[len(prefix):]
    return None

def get_concept_qid(label_text):
    """
    Try to find a QID by label in English using the WDQS.
    This replicates the naive 'getConceptQid()' from the Java version:
    SELECT ?concept WHERE { ?concept rdfs:label "label_text"@en } LIMIT 1

    Returns e.g. 'Qxx' or None
    """
    service = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?concept WHERE {{
      ?concept rdfs:label "{label_text}"@en .
    }} LIMIT 1
    """.strip()

    params = {
        "query": query,
        "format": "json"
    }
    try:
        r = requests.get(service, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # We look at data["results"]["bindings"][0]["concept"]["value"]
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                concept_uri = bindings[0]["concept"]["value"]
                # e.g. "http://www.wikidata.org/entity/Qxxx"
                if concept_uri.startswith("http://www.wikidata.org/entity/"):
                    return concept_uri.rsplit("/", 1)[-1]
        return None
    except requests.RequestException as e:
        print(f"SPARQL request error for label '{label_text}': {e}")
        return None

if __name__ == "__main__":
    main()
