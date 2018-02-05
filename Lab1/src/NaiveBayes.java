public class NaiveBayes {
    private static MultinomialModel model;

    public static void main(String[] args) {
        if (args.length < 6) {
            System.out.println("Please, include these arguments in this order with the program:");
            System.out.println("1. vocabulary.txt");
            System.out.println("2. map.csv");
            System.out.println("3. train_label.csv");
            System.out.println("4. train_data.csv");
            System.out.println("5. testing_label.csv");
            System.out.println("6. testing_data.csv");

            System.exit(1);
        }

        model = new MultinomialModel();
        model.initTerms(args[0]);
        model.initLabels(args[1]);
    }
}
