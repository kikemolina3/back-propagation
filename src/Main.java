import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class Main {
    public static void main(String[] args) {
        Properties props = new Properties();
        try {
            props.load(new FileInputStream("src\\resources\\config.properties"));
        } catch (IOException e) {
            Color.printError("Error loading config.properties file");
        }
        Dataset d = new Dataset(props);
        MNN mnn = new MNN(d, props);
        mnn.onlineBackPropagation();
        System.out.println("Done");
    }
}
