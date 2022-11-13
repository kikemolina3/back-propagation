import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class Main {
    public static void main(String[] args) throws IOException {
        Properties props = new Properties();
        try {
            props.load(new FileInputStream("src\\resources\\config.properties"));
        } catch (IOException e) {
            Color.printError("Error loading config.properties file");
        }
        Dataset d = new Dataset(props);
        NeuralNetwork neuralNetwork = new NeuralNetwork(d, props);
        if (args[0].equals("from_file")) {
            neuralNetwork.loadFromConfigFile(props);
            neuralNetwork.onlineBackPropagation();
            System.out.println("Done");
        } else if (args[0].equals("get_best_params")) {
            neuralNetwork.crossValidation();
        } else {
            Color.printError("Invalid argument");
        }
    }
}
