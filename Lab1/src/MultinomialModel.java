import sun.awt.image.IntegerComponentRaster;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

class MultinomialModel {
    private HashMap<Integer, String> words = null;
    private HashMap<Integer, String> labels = null;
    private HashMap<Integer, HashSet<Integer>> label_docs = null;
    private HashMap<Integer, HashMap<Integer, Integer>> doc_word_count = null;

    private int num_docs = 0;

    private double[] priors;
    private double[][] max_likelihood_estimate;
    private double[][] baysian_estimate;

    void initVocab(String file_name) {
        File file = new File(file_name);

        try {
            Scanner scanner = new Scanner(file);
            int count = 1;
            words = new HashMap<>();

            while (scanner.hasNext()) {
                words.put(count, scanner.next());
                count++;
            }

            scanner.close();
            System.out.println("Processed "+words.size()+" terms.");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    void initLabels(String file_name) {
        File file = new File(file_name);

        try {
            Scanner scanner = new Scanner(file);
            labels = new HashMap<>();

            while (scanner.hasNext()) {
                // ASSUME: data = <labelId, labelName>
                String[] data = scanner.next().split(",");
                int label_id = Integer.parseInt(data[0]);
                labels.put(label_id, data[1]);
            }

            scanner.close();
            System.out.println("Processed "+labels.size()+" labels.");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

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

    void train() {
        calculatePriors();
        calculateEstimates();
    }

    void test(String test_label_file_name, String test_data_file_name) {
        File label_file = new File(test_label_file_name);
        File data_file = new File(test_data_file_name);

        try {
            Scanner label_scan = new Scanner(label_file);
            Scanner data_scan = new Scanner(data_file);

            int total_documents = 1;
            int total_correct = 0;
            HashMap<Integer, Integer> group_total = new HashMap<>();
            HashMap<Integer, Integer> group_correct = new HashMap<>();

            int curr_doc = 1;   // ASSUME the first document id is 1
            double[] estimates = new double[labels.size() + 1];    // size + 1 because ids start at 1

            while (data_scan.hasNext()) {
                // ASSUME: data = <doc_id, word_id, word_frequency>
                String[] data =  data_scan.next().split(",");
                int doc_id = Integer.parseInt(data[0]);
                int word_id = Integer.parseInt(data[1]);
                int word_frequency = Integer.parseInt(data[2]);     // TODO do we count each frequency as a position

                // If we are finished with the prior document, we need to finish determining the predicted label
                if (doc_id != curr_doc) {
                    // Find the label, w_j, that maximizes the estimate
                    double max_classification = Double.MIN_VALUE;
                    int max_label = 0;

                    // Multiply P( w_j ) with the running estimate product w_NB
                    for (int label = 1; label <= labels.size(); label++) {
                        estimates[label] *= priors[label];

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

                    System.out.println(correct_label+" vs. "+max_label);

                    total_documents++;
                    group_total.put(correct_label, group_total.getOrDefault(correct_label, 0)+1);

                    curr_doc = doc_id;
                    estimates = new double[labels.size() + 1];  // size + 1 because ids start at 1
                }

                // Multiply P( x_i | w_j ) with the running estimate product w_NB
                for (int label = 1; label <= labels.size(); label++) {
                    for (int i = 0; i < word_frequency; i++) {
                        estimates[label] *= baysian_estimate[label][word_id];
                    }
                }
            }

            label_scan.close();
            data_scan.close();

            System.out.println("Test Results using Baysian Estimate");
            System.out.printf("Overall Accuracy: %.4f \n", ( total_correct + 0.0) / total_documents);
            System.out.println("Class Accuracy:");
            for (int group: group_total.keySet()) {
                System.out.printf("Group %d: %.4f \n", group, ( group_correct.getOrDefault(group, 0) + 0.0) / group_total.get(group));
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /************* Helper Functions *************/

    private void insertLabelDoc(int label_id, int doc_id) {
        if(!label_docs.containsKey(label_id)) {
            label_docs.put(label_id, new HashSet<>());
        }

        label_docs.get(label_id).add(doc_id);
    }

    private void insertDocTerm(int doc_id, int term_id, int count) {
        if (!doc_word_count.containsKey(doc_id)) {
            doc_word_count.put(doc_id, new HashMap<>());
        }

        HashMap<Integer, Integer> term_count = doc_word_count.get(doc_id);
        int temp_count = count;

        if (term_count.containsKey(term_id)) {
            temp_count += term_count.get(term_id);
        }

        term_count.put(term_id, temp_count);
    }

    private void calculatePriors() {
        System.out.println("Calculating class priors for P ( w_j ):");
        priors = new double[labels.size() + 1];  // size + 1 because ids start at 1

        for (int label = 1; label <= labels.size(); label++) {
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
        max_likelihood_estimate = new double[labels.size() + 1][words.size() + 1];
        baysian_estimate = new double[labels.size() + 1][words.size() + 1];

        // For each label, w_j
        for (int label = 1; label <= labels.size(); label++) {
            // Get all documents in w_j
            HashSet<Integer> docs = label_docs.get(label);

            // Calculate n - the total number of words in all documents in w_j
            int total_words = 0;
            for (int doc : docs) {
                // Key = word_id, Value = # of occurances in doc
                total_words += doc_word_count.get(doc).size();
            }

            // for each word, w_k
            for (int word = 1; word <= words.size(); word++) {

                // Calculate n_j - the number of times w_k occurs in all documents in class w_j
                int total_occurances = 0;
                for (int doc : docs) {
                    total_occurances += doc_word_count.get(doc).getOrDefault(word, 0);
                }

                // Set estimate values
                max_likelihood_estimate[label][word] = (total_occurances + 0.0) / total_words;
                baysian_estimate[label][word] = (total_occurances + 1.0) / (total_occurances + words.size());
            }
        }
        System.out.println("Calculated estimates for P ( w_k | w_j )");

    }
}
