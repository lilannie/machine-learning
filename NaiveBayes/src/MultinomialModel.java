import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

/**
 * @author Annie Steenson
 *
 * Pertaining to COM S 474 lab 1, label is synonymous with the category/newsgroup of a document.
 *
 */
class MultinomialModel {
    // ( key = labelId, val = [ doc_ids ]
    private HashMap<Integer, HashSet<Integer>> label_docs = null;
    // ( key = doc_id, < key = word_id, val = count> )
    private HashMap<Integer, HashMap<Integer, Integer>> doc_word_count = null;

    private int num_words = 0;
    private int num_docs = 0;
    private int num_labels = 0;

    // This array holds the priors of each label
    private double[] priors;

    // These arrays ([label_id][word_id]) contain each estimates for a label given a word
    private double[][] max_likelihood_estimate;
    private double[][] baysian_estimate;

    /**
     * Parses the file_name and counts the number of vocabulary words
     * @param file_name
     */
    void initVocab(String file_name) {
        File file = new File(file_name);

        try {
            Scanner scanner = new Scanner(file);
            while (scanner.hasNext()) {
                // We do not store the text of the words themselves because we do not use it later on
                scanner.next();
                num_words++;
            }
            scanner.close();
            System.out.println("Processed "+num_words+" terms.");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * This function instantiates all data structures needed to store information on which document belongs to which
     * label and the word count of each word in each document
     * @param train_label_file_name
     * @param train_data_file_name
     */
    void initTrainingData(String train_label_file_name, String train_data_file_name) {
        File label_file = new File(train_label_file_name);
        File data_file = new File(train_data_file_name);

        try {
            // Load train_label.csv
            Scanner label_scan = new Scanner(label_file);
            label_docs = new HashMap<>();

            while (label_scan.hasNext()) {
                num_docs++;
                String label_id = label_scan.next();
                insertLabelDoc(Integer.parseInt(label_id), num_docs);
            }
            label_scan.close();
            num_labels = label_docs.size();
            System.out.println("Found "+num_labels+" newsgroups.");
            System.out.println("Processed "+num_docs+" documents.");

            // Load train_data.csv
            Scanner data_scan = new Scanner(data_file);
            doc_word_count = new HashMap<>();
            while (data_scan.hasNext()) {
                // ASSUME: data = <doc_id, word_id, word_frequency>
                String[] data =  data_scan.next().split(",");
                insertDocTerm(Integer.parseInt(data[0]), Integer.parseInt(data[1]), Integer.parseInt(data[2]));
            }
            data_scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * This function is a public wrapper to train our model.
     */
    void train() {
        calculatePriors();
        calculateEstimates();
    }

    /**
     * This function will test our model against given test data and output statistics about accuracy
     * @param test_label_file_name
     * @param test_data_file_name
     * @param useMLE
     */
    void test(String test_label_file_name, String test_data_file_name, boolean useMLE) {
        File label_file = new File(test_label_file_name);
        File data_file = new File(test_data_file_name);

        try {
            Scanner label_scan = new Scanner(label_file);
            Scanner data_scan = new Scanner(data_file);

            // Data structures for performance evaluation
            int total_documents = 1;
            int total_correct = 0;
            HashMap<Integer, Integer> group_total = new HashMap<>();
            HashMap<Integer, Integer> group_correct = new HashMap<>();
            int[][] confusionMatrix = new int[num_labels + 1][num_labels + 1];

            // Variables to keep track which document we are currently parsing and the associated estimates
            int curr_doc = 1;   // ASSUME the first document id is 1
            double[] estimates = new double[num_labels + 1];    // size + 1 because ids start at 1

            while (data_scan.hasNext()) {
                // ASSUME: data = <doc_id, word_id, word_frequency>
                String[] data =  data_scan.next().split(",");
                int doc_id = Integer.parseInt(data[0]);
                int word_id = Integer.parseInt(data[1]);
                int word_frequency = Integer.parseInt(data[2]);

                // If we are finished with the prior document, we need to finish determining the predicted label
                if (doc_id != curr_doc) {
                    // We want to find the label, w_j, that maximizes the estimate

                    // Set max equal to the first category
                    estimates[1] += Math.log(priors[1]);
                    double max_classification = estimates[1];
                    int max_label = 1;

                    // Add P( w_j ) with the running estimated sum of w_NB
                    for (int label = 2; label <= num_labels; label++) {
                        estimates[label] += Math.log(priors[label]);

                        if (estimates[label] > max_classification) {
                            max_classification = estimates[label];
                            max_label = label;
                        }
                    }

                    // Update the count for total and number correct
                    int correct_label = Integer.parseInt(label_scan.next());
                    if (max_label == correct_label) {
                        total_correct++;
                        group_correct.put(correct_label, group_correct.getOrDefault(correct_label, 0)+1);
                    }
                    // If it is incorrect add to the confusion matrix
                    else {
                        confusionMatrix[correct_label][max_label] += 1;
                    }

                    total_documents++;
                    group_total.put(correct_label, group_total.getOrDefault(correct_label, 0)+1);

                    // Re-initialize our document variables for the next document
                    curr_doc = doc_id;
                    estimates = new double[num_labels + 1];  // size + 1 because ids start at 1
                }

                // Update estimates based on the word we just parsed from the input
                // Multiply P( x_i | w_j ) with the running estimate product w_NB
                for (int label = 1; label <= num_labels; label++) {
                    for (int i = 0; i < word_frequency; i++) {
                        if (useMLE) {
                            estimates[label] += Math.log(max_likelihood_estimate[label][word_id]);
                        }
                        else {
                            estimates[label] += Math.log(baysian_estimate[label][word_id]);
                        }
                    }
                }
            }

            label_scan.close();
            data_scan.close();

            // Print performance statistics
            System.out.printf("Overall Accuracy: %.4f %% \n", (( total_correct + 0.0) / total_documents ) *100 );
            System.out.println("Class Accuracy:");
            for (int group: group_total.keySet()) {
                System.out.printf("Group %d: %.4f %% \n", group, (( group_correct.getOrDefault(group, 0) + 0.0) / group_total.get(group)) *100 );
            }

            final PrettyPrinter p = new PrettyPrinter(System.out);
            p.print(confusionMatrix);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /************* Helper Functions *************/

    /**
     * Checks for already inserted labels and adds the doc_id to that label's document set
     * @param label_id
     * @param doc_id
     */
    private void insertLabelDoc(int label_id, int doc_id) {
        if(!label_docs.containsKey(label_id)) {
            label_docs.put(label_id, new HashSet<>());
        }

        label_docs.get(label_id).add(doc_id);
    }

    /**
     * Checks for already inserted documents and add the word and the word's count to that documents word map
     * @param doc_id
     * @param word_id
     * @param count
     */
    private void insertDocTerm(int doc_id, int word_id, int count) {
        if (!doc_word_count.containsKey(doc_id)) {
            doc_word_count.put(doc_id, new HashMap<>());
        }

        HashMap<Integer, Integer> term_count = doc_word_count.get(doc_id);
        int temp_count = count;

        if (term_count.containsKey(word_id)) {
            temp_count += term_count.get(word_id);
        }

        term_count.put(word_id, temp_count);
    }

    /**
     * Calculates the prior probability for each label based on the number of instances of that label
     * and the total number of instances
     */
    private void calculatePriors() {
        System.out.println("-----------------------------------------------------");
        System.out.println("Class priors for P ( w_j ) where w_j is a newsgroup j:");
        System.out.println("-----------------------------------------------------");

        priors = new double[num_labels + 1];  // size + 1 because ids start at 1

        for (int label = 1; label <= num_labels; label++) {
            priors[label] = ( label_docs.get(label).size() + 0.0 ) / num_docs;

            // Print Prior for each newsgroup
            System.out.printf("P ( w_" + (label) +  " ) = %.4f \n", priors[label]);
        }
    }

    /**
     *      ASSUME:
     *              w_k = a word in Vocabulary
     *              w_j = a label
     *              n_k = number of times word w_k occurs in all documents in class w_j
     *              n = total number of words in all documents in class w_j
     *
     *      Maximum Likelinhood Estimate:
     *              P ( w_k | w_j ) = n_k / n
     *
     *      Bayesian Estimate:
     *               P ( w_k | w_j ) = ( n_k + 1 ) / ( n + |Vocabulary| )
     */
    private void calculateEstimates() {
        // size + 1 because ids start at 1
        max_likelihood_estimate = new double[num_labels + 1][num_words + 1];
        baysian_estimate = new double[num_labels + 1][num_words + 1];

        // For each label, w_j
        for (int label = 1; label <= num_labels; label++) {
            // Get all documents in w_j
            HashSet<Integer> docs = label_docs.get(label);

            // Calculate n - the total number of words in all documents in w_j
            int total_words = 0;
            for (int doc : docs) {
                // Key = word_id, Value = # of occurances in doc
                HashMap<Integer, Integer> word_count = doc_word_count.get(doc);
                for (int word: word_count.keySet()) {
                    total_words += word_count.get(word);
                }
            }

            // for each word, w_k
            for (int word = 1; word <= num_words; word++) {

                // Calculate n_j - the number of times w_k occurs in all documents in class w_j
                int total_occurences = 0;
                for (int doc : docs) {
                    total_occurences += doc_word_count.get(doc).getOrDefault(word, 0);
                }

                // Set estimate values
                max_likelihood_estimate[label][word] = (double) ((total_occurences) / total_words);
                baysian_estimate[label][word] = (double)  (total_occurences + 1) / (total_words + num_words);
            }
        }
    }
}
