public class NaiveBayes {
    private static MultinomialModel model;

    public static void main(String[] args) {
        if (args.length < 6) {
            System.out.println("Please, include these arguments in this order with the program:");
            System.out.println("1. vocabulary.txt");
            System.out.println("2. map.csv");
            System.out.println("3. train_label.csv");
            System.out.println("4. train_data.csv");
            System.out.println("5. test_label.csv");
            System.out.println("6. test_data.csv");

            System.exit(1);
        }

        model = new MultinomialModel();
        model.initVocab(args[0]);
        model.initTrainingData(args[2], args[3]);
        model.train();

        System.out.println("-----------------------------------------------------");
        System.out.println("Testing the training data with Baysian Estimate");
        model.test(args[2], args[3], false);
        System.out.println("-----------------------------------------------------");

        System.out.println("-----------------------------------------------------");
        System.out.println("Testing the test data with Baysian Estimate");
        System.out.println("-----------------------------------------------------");
        model.test(args[4], args[5], false);

        System.out.println("-----------------------------------------------------");
        System.out.println("Testing the test data with Maximum Likelihood Estimate");
        System.out.println("-----------------------------------------------------");
        model.test(args[4], args[5], true);

    }
}
