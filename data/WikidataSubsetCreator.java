import org.wikidata.wdtk.datamodel.interfaces.*;
import org.wikidata.wdtk.dumpfiles.*;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

public class WikidataSubsetCreator {

    public static void main(String[] args) throws IOException {
        // --- Configuration ---
        String dumpFilePath = "/path/to/your/wikidata-dump.json.bz2"; // *** CHANGE THIS ***
        String outputFilePath = "/path/to/your/output.tsv";        // *** CHANGE THIS ***
        Set<String> targetConcepts = new HashSet<>();
        // Add the QIDs you care about to targetConcepts. Example:
        // targetConcepts.add("Q24229398"); // Example: "William Shakespeare"
        // You'll load these QIDs in your Python script.

        // --- Load target concept QIDs (from a file, for example) ---
        String qidFilePath = "/path/to/qids.txt"; // *** CHANGE THIS *** Or load inline
        try (BufferedReader reader = new BufferedReader(new FileReader(qidFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                targetConcepts.add(line.trim());
            }
        } catch (IOException e) {
            System.err.println("Error reading QID file: " + e.getMessage());
            // If you don't want to use a file, comment out the try-catch and add QIDs directly above.
        }


        // --- WDTK Setup ---
        MwDumpFile dumpFile = new MwDumpFile(dumpFilePath);
        ExampleDumpProcessingHelper dumpProcessingHelper = new ExampleDumpProcessingHelper(); // Use built-in helper
        DumpProcessingController dumpProcessingController = new DumpProcessingController("wikidatawiki");
        dumpProcessingController.registerEntityDocumentProcessor(
                new SubsetProcessor(outputFilePath, targetConcepts), null, true);

        // --- Process the Dump ---
        dumpProcessingController.processDump(dumpFile);
        System.out.println("Finished processing Wikidata dump.");
    }

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
                      String concept_id = ExampleDumpProcessingHelper.getConceptQid(label.getText()); //Use the existing method
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
			// Nothing to do (we don't use Lexemes in this example)
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
    }
}