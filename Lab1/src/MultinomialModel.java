import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;
import java.util.StringTokenizer;

public class MultinomialModel {
    HashSet<String> terms;
    HashMap<Integer, String> labels;

    public void initTerms(String file_name) {
        terms = new HashSet<>();
        File f = new File(file_name);
        try {
            Scanner scanner = new Scanner(f);
            while (scanner.hasNext()) {
                terms.add(scanner.next());
            }

            scanner.close();
            System.out.println("Found "+terms.size()+" terms.");
        } catch (FileNotFoundException e) {
            System.out.println("ERROR reading from the file: "+file_name);
            e.printStackTrace();
        }
    }

    public void initLabels(String file_name) {
        labels = new HashMap<>();
        File f = new File(file_name);
        try {
            Scanner scanner = new Scanner(f);
            while (scanner.hasNext()) {
                String[] data = scanner.next().split(",");
                int id = Integer.parseInt(data[0]);
                labels.put(id, data[1]);
            }

            scanner.close();
            System.out.println("Found "+labels.size()+" labels.");

        } catch (FileNotFoundException e) {
            System.out.println("ERROR reading from the file: "+file_name);
            e.printStackTrace();
        }
    }
}
