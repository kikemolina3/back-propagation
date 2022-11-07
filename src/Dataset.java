import java.io.*;
import java.util.*;

public class Dataset {

    private String filename;
    private int num_samples;
    private int num_features;
    private int num_outputs;
    private float[][] inputs;

    private float[][] outputs;
    private float[] input_mins;
    private float[] input_maxs;
    private float[] output_mins;
    private float[] output_maxs;

    public Dataset(Properties props) {
        this.validateProps(props);

        this.filename = props.getProperty("dataset");

        this.num_samples = Integer.parseInt(props.getProperty("n_samples"));
        this.num_features = Integer.parseInt(props.getProperty("n_features"));
        this.num_outputs = Integer.parseInt(props.getProperty("n_outputs"));

        this.inputs = new float[this.num_samples][this.num_features];
        this.outputs = new float[this.num_samples][this.num_outputs];

        if (!Objects.equals(props.getProperty("dataset"), "A1-insurance-third-dataset.txt")) {
            this.normalDatasetLoading(props.getProperty("dataset"));
        } else {
            this.a1InsuranceDatasetLoading(props.getProperty("dataset"));
        }

    }

    private void normalDatasetLoading(String filename) {
        try {
            BufferedReader br = new BufferedReader(new FileReader("src\\resources\\" + filename));
            String line;
            br.readLine();
            for (int i = 0; i < this.num_samples; i++) {
                line = br.readLine();
                String[] values = line.split("\t");
                for (int j = 0; j < this.num_features + this.num_outputs; j++) {
                    if (j < this.num_features) {
                        this.inputs[i][j] = Float.parseFloat(values[j]);
                    } else {
                        this.outputs[i][j - this.num_features] = Float.parseFloat(values[j]);
                    }
                }
            }
            br.close();
        } catch (Exception e) {
            Color.printError("Error reading dataset file. Please review the dataset file format and config.properties");
            System.exit(1);
        }
    }

    private void a1InsuranceDatasetLoading(String filename) {
        List<Integer> normalIndexes = new ArrayList<>(Arrays.asList(0, 2, 3));
        try {
            BufferedReader br = new BufferedReader(new FileReader("src\\resources\\" + filename));
            String line;
            br.readLine();
            for (int i = 0; i < this.num_samples; i++) {
                line = br.readLine();
                String[] values = line.split("\t");
                for (int j = 0; j < this.num_features + this.num_outputs; j++) {
                    if (j < this.num_features) {
                        if (normalIndexes.contains(j)) {
                            this.inputs[i][j] = Float.parseFloat(values[j]);
                        } else {
                            if (j == 1) {
                                if (Objects.equals(values[j], "male"))
                                    this.inputs[i][j] = 0;
                                else
                                    this.inputs[i][j] = 1;
                            } else if (j == 4) {
                                if (Objects.equals(values[j], "yes"))
                                    this.inputs[i][j] = 0;
                                else
                                    this.inputs[i][j] = 1;
                            } else if (j == 5) {
                                if (Objects.equals(values[j], "southwest"))
                                    this.inputs[i][j] = 0;
                                else if (Objects.equals(values[j], "southeast"))
                                    this.inputs[i][j] = 1;
                                else if (Objects.equals(values[j], "northwest"))
                                    this.inputs[i][j] = 2;
                                else
                                    this.inputs[i][j] = 3;
                            }
                        }
                    } else {
                        this.outputs[i][j - this.num_features] = Float.parseFloat(values[j]);
                    }
                }
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
            Color.printError("Error reading dataset file. Please review the dataset file format and config.properties");
            System.exit(1);
        }
        this.invalidateOutliers();
    }

    public void invalidateOutliers() {
        for (int i = 0; i < this.num_features; i++) {
            float mean = 0;
            for (int j = 0; j < this.num_samples; j++) {
                mean += this.inputs[j][i];
            }
            mean /= this.num_samples;
            float std = 0;
            for (int j = 0; j < this.num_samples; j++) {
                std += Math.pow(this.inputs[j][i] - mean, 2);
            }
            std /= this.num_samples;
            std = (float) Math.sqrt(std);
            for (int j = 0; j < this.num_samples; j++) {
                if (Math.abs(this.inputs[j][i] - mean) > 3 * std) {
//                    System.out.println("Num_feature: " + i + "\tNum_sample: " + j + "\tValue: " + this.x_values[j][i]);
                    this.inputs[j][i] = mean;
                }
            }
        }
    }

    public void scale(float min_value, float max_value) {
        this.input_mins = new float[this.num_features];
        this.input_maxs = new float[this.num_features];
        this.output_mins = new float[this.num_outputs];
        this.output_maxs = new float[this.num_outputs];
        for (int i = 0; i < this.num_features + this.num_outputs; i++) {
            if (i < this.num_features) {
                this.input_mins[i] = this.inputs[0][i];
                this.input_maxs[i] = this.inputs[0][i];
                for (int j = 0; j < this.num_samples; j++) {
                    if (this.inputs[j][i] < this.input_mins[i]) {
                        this.input_mins[i] = this.inputs[j][i];
                    }
                    if (this.inputs[j][i] > this.input_maxs[i]) {
                        this.input_maxs[i] = this.inputs[j][i];
                    }
                }
                for (int j = 0; j < this.num_samples; j++) {
                    this.inputs[j][i] = (this.inputs[j][i] - this.input_mins[i]) / (this.input_maxs[i] - this.input_mins[i]) * (max_value - min_value) + min_value;
                }
            } else {
                int index = i - this.num_features;
                this.output_mins[index] = this.outputs[0][index];
                this.output_maxs[index] = this.outputs[0][index];
                for (int j = 0; j < this.num_samples; j++) {
                    if (this.outputs[j][index] < this.output_mins[index]) {
                        this.output_mins[index] = this.outputs[j][index];
                    }
                    if (this.outputs[j][index] > this.output_maxs[index]) {
                        this.output_maxs[index] = this.outputs[j][index];
                    }
                }
                for (int j = 0; j < this.num_samples; j++) {
                    this.outputs[j][index] = (this.outputs[j][index] - this.output_mins[index]) / (this.output_maxs[index] - this.output_mins[index]) * (max_value - min_value) + min_value;
                }
            }
        }
    }

    public void unscale(float min_value, float max_value) {
        for (int i = 0; i < this.num_features + this.num_outputs; i++) {
            if (i < this.num_features) {
                for (int j = 0; j < this.num_samples; j++) {
                    this.inputs[j][i] = (this.inputs[j][i] - min_value) / (max_value - min_value) * (this.input_maxs[i] - this.input_mins[i]) + this.input_mins[i];
                }
            } else {
                int index = i - this.num_features;
                for (int j = 0; j < this.num_samples; j++) {
                    this.outputs[j][index] = (this.outputs[j][index] - min_value) / (max_value - min_value) * (this.output_maxs[index] - this.output_mins[index]) + this.output_mins[index];
                }
            }
        }
    }

    private void validateProps(Properties props) {
        String[] mandatoryProps = {
                "dataset",
                "training_percentage",
                "activation_function",
                "number_of_hidden_layers",
                "number_of_neurons_per_hidden_layer",
                "number_of_epochs",
                "learning_rate",
                "momentum",
                "n_samples",
                "n_features",
                "n_outputs"
        };

        String[] integerProps = {
                "training_percentage",
                "number_of_hidden_layers",
                "number_of_epochs",
                "n_samples",
                "n_features",
                "n_outputs"
        };

        String[] floatProps = {
                "learning_rate",
                "momentum"
        };


        // existance check
        for (String prop : mandatoryProps) {
            if (!props.containsKey(prop)) {
                Color.printError("property " + prop + " not found in config.properties");
                System.exit(1);
            }
        }

        // integer check
        for (String prop : integerProps) {
            try {
                Integer.parseInt(props.getProperty(prop));
            } catch (NumberFormatException e) {
                Color.printError("property " + prop + " must be an integer");
                System.exit(1);
            }
        }

        // float check
        for (String prop : floatProps) {
            try {
                Float.parseFloat(props.getProperty(prop));
            } catch (NumberFormatException e) {
                Color.printError("property " + prop + " must be a float");
                System.exit(1);
            }
        }

        // percentage_training check [0-100]
        if (Integer.parseInt(props.getProperty("training_percentage")) < 0 || Integer.parseInt(props.getProperty("training_percentage")) > 100) {
            Color.printError("percentage_training must be between 0 and 100");
            System.exit(1);
        }

        // activation_function check
        if (!props.getProperty("activation_function").equals("sigmoid") && !props.getProperty("activation_function").equals("tanh") && !props.getProperty("activation_function").equals("linear") && !props.getProperty("activation_function").equals("relu")) {
            Color.printError("activation_function must be sigmoid, tanh, linear or relu");
            System.exit(1);
        }


        // dataset file check
        try {
            FileInputStream dataset = new FileInputStream("src\\resources\\" + props.getProperty("dataset"));
        } catch (FileNotFoundException e) {
            Color.printError("dataset file not found");
            System.exit(1);
        }
    }


    public int getNum_samples() {
        return num_samples;
    }

    public float[][] getInputs() {
        return inputs;
    }

    public float[][] getOutputs() {
        return outputs;
    }

    public String getFilename() {
        return filename;
    }
}
