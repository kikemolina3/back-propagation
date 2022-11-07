import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

public class MNN {
    private Dataset dataset;
    private float learning_rate;
    private float momentum;
    private float epochs;
    private int L;
    private int[] n;
    private float[][] h;
    private float[][] xi;
    private float[][][] w;
    private float[][] theta;
    private float[][] delta;
    private float[][][] d_w;
    private float[][] d_theta;
    private float[][][] d_w_prev;
    private float[][] d_theta_prev;
    private FactType fact;

    private float training_ratio;


    public MNN(Dataset dataset, Properties props) {
        this.dataset = dataset;

        learning_rate = Float.parseFloat(props.getProperty("learning_rate"));
        momentum = Float.parseFloat(props.getProperty("momentum"));
        epochs = Float.parseFloat(props.getProperty("number_of_epochs"));
        L = Integer.parseInt(props.getProperty("number_of_hidden_layers")) + 2;
        n = new int[L];
        String n_str = props.getProperty("number_of_neurons_per_hidden_layer");
        String[] n_str_arr = n_str.split(",");
        n[0] = Integer.parseInt(props.getProperty("n_features"));
        for (int i = 1; i < L - 1; i++) {
            n[i] = Integer.parseInt(n_str_arr[i - 1]);
        }
        n[L - 1] = Integer.parseInt(props.getProperty("n_outputs"));
        fact = FactType.valueOf(props.getProperty("activation_function"));
        training_ratio = Float.parseFloat(props.getProperty("training_percentage")) / 100;
    }

    public void init() {

        w = new float[L - 1][][];
        for (int l = 0; l < L - 1; l++) {
            w[l] = new float[n[l + 1]][n[l]];
            for (int i = 0; i < n[l + 1]; i++) {
                for (int j = 0; j < n[l]; j++) {
                    w[l][i][j] = (float) Math.random();
                }
            }
        }

        theta = new float[L - 1][];
        for (int l = 0; l < L - 1; l++) {
            theta[l] = new float[n[l + 1]];
            for (int i = 0; i < n[l + 1]; i++) {
                theta[l][i] = (float) Math.random();
            }
        }

        h = new float[L][];
        for (int l = 0; l < L; l++) {
            h[l] = new float[n[l]];
        }

        xi = new float[L][];
        for (int l = 0; l < L; l++) {
            xi[l] = new float[n[l]];
        }

        delta = new float[L - 1][];
        for (int l = 0; l < L - 1; l++) {
            delta[l] = new float[n[l + 1]];
        }

        d_w = new float[L - 1][][];
        for (int l = 0; l < L - 1; l++) {
            d_w[l] = new float[n[l + 1]][n[l]];
            for (int i = 0; i < n[l + 1]; i++) {
                for (int j = 0; j < n[l]; j++) {
                    d_w[l][i][j] = 0;
                }
            }
        }

        d_theta = new float[L - 1][];
        for (int l = 0; l < L - 1; l++) {
            d_theta[l] = new float[n[l + 1]];
            for (int i = 0; i < n[l + 1]; i++) {
                d_theta[l][i] = 0;
            }
        }

        d_w_prev = new float[L - 1][][];
        for (int l = 1; l < L - 1; l++) {
            d_w_prev[l] = new float[n[l + 1]][n[l]];
            for (int i = 0; i < n[l + 1]; i++) {
                for (int j = 0; j < n[l]; j++) {
                    d_w_prev[l][i][j] = 0;
                }
            }
        }

        d_theta_prev = new float[L - 1][];
        for (int l = 1; l < L - 1; l++) {
            d_theta_prev[l] = new float[n[l + 1]];
            for (int i = 0; i < n[l + 1]; i++) {
                d_theta_prev[l][i] = 0;
            }
        }

    }

    public void feedForwardPropagation() {
        for (int l = 0; l < L - 1; l++) {
            for (int i = 0; i < n[l + 1]; i++) {
                float sum = 0;
                for (int j = 0; j < n[l]; j++) {
                    sum += w[l][i][j] * xi[l][j];
                }
                h[l + 1][i] = sum - theta[l][i];
                if (fact == FactType.sigmoid) {
                    xi[l + 1][i] = (float) (1 / (1 + Math.exp(-h[l + 1][i])));
                } else if (fact == FactType.tanh) {
                    xi[l + 1][i] = (float) Math.tanh(sum - theta[l][i]);
                } else if (fact == FactType.relu) {
                    xi[l + 1][i] = Math.max(0, sum - theta[l][i]);
                } else if (fact == FactType.linear) {
                    xi[l + 1][i] = sum - theta[l][i];
                }
            }
        }
    }

    public void errorBackPropagation(int sampleIndex) {
        for (int l = L - 2; l >= 0; l--) {
            for (int i = 0; i < n[l + 1]; i++) {
                float sum = 0;
                if (l == L - 2) {
                    sum = xi[l + 1][i] - dataset.getOutputs()[sampleIndex][i];
                } else {
                    for (int j = 0; j < n[l + 2]; j++) {
                        sum += w[l + 1][j][i] * delta[l + 1][j];
                    }
                }
                if (fact == FactType.sigmoid) {
                    delta[l][i] = sum * xi[l + 1][i] * (1 - xi[l + 1][i]);
                } else if (fact == FactType.tanh) {
                    delta[l][i] = sum * (1 - xi[l + 1][i] * xi[l + 1][i]);
                } else if (fact == FactType.relu) {
                    delta[l][i] = sum * (xi[l + 1][i] > 0 ? 1 : 0);
                } else if (fact == FactType.linear) {
                    delta[l][i] = sum;
                }
            }
        }
    }

    public void updateWeightsAndTheta() {
        for (int l = 0; l < L - 1; l++) {
            for (int i = 0; i < n[l + 1]; i++) {
                for (int j = 0; j < n[l]; j++) {
                    d_w[l][i][j] = (-learning_rate * delta[l][i] * xi[l][j]) + (momentum * d_w[l][i][j]);
                    w[l][i][j] += d_w[l][i][j];
                }
                d_theta[l][i] = (learning_rate * delta[l][i]) + (momentum * d_theta[l][i]);
                theta[l][i] += d_theta[l][i];
            }
        }
    }

    public void onlineBackPropagation() {
        init();
        dataset.scale(0.1f, 0.9f);
        float globalError = 0;
        List<Integer> trainingIndexes = getTrainingIndexes();
        List<Integer> testingIndexes = getTestIndexes(trainingIndexes);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Integer i : trainingIndexes) {
                xi[0] = dataset.getInputs()[i];
                feedForwardPropagation();
                errorBackPropagation(i);
                updateWeightsAndTheta();
            }
        }
        System.out.println("BEGIN TEST SET");
        for (Integer i : testingIndexes) {
            xi[0] = dataset.getInputs()[i];
            feedForwardPropagation();
            for (int j = 0; j < n[L - 1]; j++) {
                System.out.println("Expected: " + dataset.getOutputs()[i][j] + " - Obtained: " + xi[L - 1][j]);
            }
            float sum = 0;
            for (int j = 0; j < n[L - 1]; j++) {
                sum += Math.abs((dataset.getOutputs()[i][j] - xi[L - 1][j]) / dataset.getOutputs()[i][j]);
            }
            globalError += sum / n[L - 1];
        }
        System.out.println("Global MAPE: " + (globalError / (dataset.getNum_samples() * (1 - training_ratio))) * 100 + "%");
        dataset.unscale(0.1f, 0.9f);

    }

    private List<Integer> getTrainingIndexes() {
        List<Integer> trainingIndexes = new ArrayList<>();
        if (dataset.getFilename().equals("A1-insurance-third-dataset.txt")) {
            List<Integer> indexes = new ArrayList<>();
            for (int i = 0; i < dataset.getNum_samples(); i++) {
                indexes.add(i);
            }
            Collections.shuffle(indexes);
            for (int i = 0; i < dataset.getNum_samples() * training_ratio; i++) {
                trainingIndexes.add(indexes.get(i));
            }
        } else {
            for (int i = 0; i < dataset.getNum_samples() * training_ratio; i++) {
                trainingIndexes.add(i);
            }
        }
        System.out.println("Training indexes: " + trainingIndexes);
        return trainingIndexes;
    }

    private List<Integer> getTestIndexes(List<Integer> trainingIndexes) {
        List<Integer> testIndexes = new ArrayList<>();
        for (int i = 0; i < dataset.getNum_samples(); i++) {
            if (!trainingIndexes.contains(i)) {
                testIndexes.add(i);
            }
        }
        System.out.println("Test indexes: " + testIndexes);
        return testIndexes;
    }

}
