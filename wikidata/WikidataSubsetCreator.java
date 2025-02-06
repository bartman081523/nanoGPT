// WikidataSubsetCreator.java
import org.wikidata.wdtk.datamodel.interfaces.*;
import org.wikidata.wdtk.dumpfiles.*;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

public class WikidataSubsetCreator {

    public static void main(String[] args) {
        // --- Argument Handling ---
        if (args.length != 3) {
            System.err.println("Usage: java WikidataSubsetCreator <dumpFilePath> <outputFilePath> <qidFilePath>");
            System.exit(1);
        }
        String dumpFilePath = args[0];
        String outputFilePath = args[1];
        String qidFilePath = args[2];


        // --- Load target concept QIDs ---
        Set<String> targetConcepts = new HashSet<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(qidFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                targetConcepts.add(line.trim());
            }
        } catch (IOException e) {
            System.err.println("Error reading QID file: " + e.getMessage());
            System.exit(1); // Exit on error
        }


        // --- WDTK Setup and Processing ---
        try {
            MwDumpFile dumpFile = new MwDumpFile(dumpFilePath);
			DumpProcessingController dumpProcessingController = new DumpProcessingController("wikidatawiki");
            dumpProcessingController.registerEntityDocumentProcessor(
                new SubsetProcessor(outputFilePath, targetConcepts), null, true);

            // Process the Dump
            dumpProcessingController.processDump(dumpFile);
            System.out.println("Finished processing Wikidata dump. Output written to: " + outputFilePath);

        }  catch (IOException e) {
            System.err.println("Error processing Wikidata dump: " + e.getMessage());
            System.exit(1);
        }
    }
    //Inner class (SubsetProcessor)
     static class SubsetProcessor implements EntityDocumentProcessor {
        private final BufferedWriter writer;
        private final Set<String> targetConcepts;

        public SubsetProcessor(String outputFilePath, Set<String> targetConcepts) throws IOException {
            this.writer = new BufferedWriter(new FileWriter(outputFilePath));
            this.targetConcepts = targetConcepts;
            // Write header to TSV file
            writer.write("subject\tpredicate\tobject\n");
        }


        @Override
        public void processItemDocument(ItemDocument itemDocument) {
            try {
                String subjectQid = itemDocument.getEntityId().getId();
                // Check if the current item is relevant
                if (targetConcepts.contains(subjectQid) || targetConcepts.isEmpty()) { // Process if in target list OR if list is empty
                   writeStatements(subjectQid, itemDocument.getStatementGroups());

                }

                //Also, scan labels to find new QIDs, and add them to targetConcepts
                if (targetConcepts.contains(subjectQid)) {
                    for (MonolingualTextValue label : itemDocument.getLabels().values()) {
                      String concept_id = getConceptQid(label.getText()); //Use the existing method
                      if(concept_id != null){
                        targetConcepts.add(concept_id);
                      }
                    }
                  }
                //Process sitelinks (e.g. to Wikipedia pages), if needed
                //for (SiteLink sitelink : itemDocument.getSiteLinks().values()) {
                //    ...
                //}

            } catch (IOException e) {
                System.err.println("Error writing to file: " + e.getMessage());
            }
        }

        @Override
        public void processPropertyDocument(PropertyDocument propertyDocument) {
            // In this example we are not storing properties separately
            //We could do it if needed.
        }

        @Override
		public void processLexemeDocument(LexemeDocument lexemeDocument) {
			// Nothing to do
		}

		@Override
		public void processFormDocument(FormDocument formDocument) {
			// Nothing to do
		}

		@Override
		public void processSenseDocument(SenseDocument senseDocument) {
			// Nothing to do
		}

        //Helper to write statements
        private void writeStatements(String subjectQid, Iterable<StatementGroup> statementGroups) throws IOException{
            for (StatementGroup statementGroup : statementGroups) {
                    String propertyId = statementGroup.getProperty().getId();

                    for (Statement statement : statementGroup) {
                        Snak mainSnak = statement.getMainSnak();
                        if (mainSnak instanceof ValueSnak) {
                            Value value = ((ValueSnak) mainSnak).getValue();
                            if (value instanceof ItemIdValue) { // If the value is another item (common case)
                                String objectQid = ((ItemIdValue) value).getId();
                                writer.write(subjectQid + "\t" + propertyId + "\t" + objectQid + "\n");

                            } else if (value instanceof StringValue){ //String values (e.g. for labels)
                                String objectString = ((StringValue) value).getString();
                                writer.write(subjectQid + "\t" + propertyId + "\t" + objectString + "\n");
                            } //else: Other types of values (e.g., quantities, dates) - handle as needed
                        }
                    }
                }
        }


    //Helper method to get the QID, using Wikidata online (SPARQL)
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

                // Parse JSON response using a simple approach (you might use a JSON library for more robust parsing)
                String jsonString = response.toString();
                int start = jsonString.indexOf("value") + 9;
                if(start >= 9){ //Check if result was found
                    int end = jsonString.indexOf("\"", start);
                    String conceptUri = jsonString.substring(start, end);
                    String conceptQid = conceptUri.substring(conceptUri.lastIndexOf("/") + 1);
                    return conceptQid;
                }
                else{
                    return null; //No concept found
                }

            }
        } catch (IOException e) {
           System.err.println("Error during getConceptQid: " + e); //Print error
            return null;
        }
    }
  }
}
