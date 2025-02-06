import org.wikidata.wdtk.datamodel.interfaces.*;
import org.wikidata.wdtk.dumpfiles.*;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.HashSet;
import java.util.Set;

public class WikidataSubsetCreator {

    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: java WikidataSubsetCreator <dumpFilePath> <outputFilePath> <qidFilePath>");
            System.exit(1);
        }
        String dumpFilePath = args[0];
        String outputFilePath = args[1];
        String qidFilePath = args[2];

        // Collect target QIDs from the user-provided file:
        Set<String> targetConcepts = new HashSet<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(qidFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                targetConcepts.add(line.trim());
            }
        } catch (IOException e) {
            System.err.println("Error reading QID file: " + e.getMessage());
            System.exit(1);
        }

        try {
            // Prepare the dump file and processing controller
            MwDumpFile dumpFile = new MwLocalDumpFile(dumpFilePath);
            DumpProcessingController dumpProcessingController = new DumpProcessingController("wikidatawiki");

            // Create our processor and register it
            NTriplesSubsetProcessor processor = new NTriplesSubsetProcessor(outputFilePath, targetConcepts);
            dumpProcessingController.registerEntityDocumentProcessor(processor, null, true);

            // Process the dump
            dumpProcessingController.processDump(dumpFile);

            // Close resources (so we flush and close the output writer)
            processor.close();

            System.out.println("Finished processing Wikidata dump. Output written to: " + outputFilePath);

        } catch (IOException e) {
            System.err.println("Error processing Wikidata dump: " + e.getMessage());
            System.exit(1);
        }
    }

    /**
     * Processor that implements EntityDocumentProcessor in order to extract
     * simple triple-like statements (subject QID, property, object QID) and
     * write them to a TSV file.
     */
    static class NTriplesSubsetProcessor implements EntityDocumentProcessor {

        private final BufferedWriter writer;
        private final Set<String> targetConcepts;
        private int tripleCount = 0;

        public NTriplesSubsetProcessor(String outputFilePath, Set<String> targetConcepts) throws IOException {
            this.writer = new BufferedWriter(new FileWriter(outputFilePath));
            this.targetConcepts = targetConcepts;
            // For demonstration, we write a simple tab-separated header:
            writer.write("subject\tpredicate\tobject\n");
        }

        @Override
        public void processItemDocument(ItemDocument itemDocument) {
            try {
                String subjectQid = itemDocument.getEntityId().getId();

                // Check if the current item is relevant:
                // If we have a non-empty target set, only process if it's in that set;
                // if the set is empty, we process everything.
                if (targetConcepts.contains(subjectQid) || targetConcepts.isEmpty()) {
                    writeStatements(subjectQid, itemDocument.getStatementGroups());
                }

                // Also, scan labels to find new QIDs (based on label matching in WDQS):
                if (targetConcepts.contains(subjectQid)) {
                    for (MonolingualTextValue label : itemDocument.getLabels().values()) {
                        String concept_id = getConceptQid(label.getText());
                        if (concept_id != null) {
                            targetConcepts.add(concept_id);
                        }
                    }
                }

            } catch (IOException e) {
                System.err.println("Error writing to file: " + e.getMessage());
            }
        }

        @Override
        public void processPropertyDocument(PropertyDocument propertyDocument) {
            // We are not storing properties in this subset, so do nothing here.
        }

        @Override
        public void processLexemeDocument(LexemeDocument lexemeDocument) {
            // We are not processing lexemes here, so do nothing.
        }

        /**
         * Write statements as TSV lines: subject\tpredicate\tobject
         */
        private void writeStatements(String subjectQid, Iterable<StatementGroup> statementGroups) throws IOException {
            for (StatementGroup statementGroup : statementGroups) {
                for (Statement statement : statementGroup) {
                    Snak mainSnak = statement.getMainSnak();
                    if (mainSnak instanceof ValueSnak) {
                        Value value = ((ValueSnak) mainSnak).getValue();
                        if (value instanceof ItemIdValue) {
                            // The property used is the statementGroup's property ID (e.g. P31).
                            String predicate = statementGroup.getProperty().getId();

                            // We only consider direct property statements, which start with "P"
                            if (!predicate.startsWith("P")) {
                                continue;
                            }

                            // The object item
                            String objectQid = ((ItemIdValue) value).getId();

                            // Write out the triple-like line
                            writer.write(subjectQid + "\t" + predicate + "\t" + objectQid + "\n");
                            tripleCount++;
                        }
                    }
                }
            }
        }

        /**
         * Helper method: queries Wikidata by label to find a QID.
         * Returns null if none found.
         */
        public String getConceptQid(String conceptName) {
            String service = "https://query.wikidata.org/sparql";
            String query = "SELECT ?concept WHERE { ?concept rdfs:label \"" + conceptName + "\"@en .}";

            try {
                URL url = new URL(service + "?query=" + URLEncoder.encode(query, "UTF-8") + "&format=json");
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("GET");
                connection.setRequestProperty("Accept", "application/sparql-results+json");

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        response.append(line);
                    }

                    String jsonString = response.toString();
                    int start = jsonString.indexOf("value") + 9;
                    if (start >= 9) {
                        int end = jsonString.indexOf("\"", start);
                        String conceptUri = jsonString.substring(start, end);
                        // QID is everything after the last slash
                        return conceptUri.substring(conceptUri.lastIndexOf("/") + 1);
                    } else {
                        return null;
                    }
                }
            } catch (IOException e) {
                System.err.println("Error during getConceptQid: " + e);
                return null;
            }
        }

        /**
         * Close the BufferedWriter.
         * Make sure to call this after processing completes.
         */
        public void close() throws IOException {
            writer.close();
            System.out.println("Finished NTriples processing. Wrote " + tripleCount + " triples.");
        }
    }
}
