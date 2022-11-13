import java.io.*;
import java.util.*;

public class NeuralNetwork {
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
    private float[] learning_rate_list = {0.1f, 0.01f, 0.001f};
    private float[] momentum_list = {0.7f, 0.9f, 0.95f};
    private int hidden_layers_list[] = {1, 2, 3};
    int[] possible_neurons_per_layer = {2, 3, 4, 5};
    private FactType fact_list[] = {FactType.sigmoid/*, FactType.tanh, FactType.linear, FactType.relu*/};
    private float best_learning_rate;
    private float best_momentum;
    private int best_epochs;
    private int best_hidden_layers;
    private int[] best_hidden_neurons;
    private FactType best_fact;


    public NeuralNetwork(Dataset dataset, Properties props) {
        this.dataset = dataset;

        this.loadFromConfigFile(props);

        training_ratio = Float.parseFloat(props.getProperty("training_percentage")) / 100;
    }

    public void loadFromConfigFile(Properties props) {
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
        fact = FactType.valueOf(props.getProperty("activation_function"));
        n[L - 1] = Integer.parseInt(props.getProperty("n_outputs"));
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
            for (int i = 0; i < n[l]; i++) {
                h[l][i] = 0;
            }
        }

        xi = new float[L][];
        for (int l = 0; l < L; l++) {
            xi[l] = new float[n[l]];
            for (int i = 0; i < n[l]; i++) {
                xi[l][i] = 0;
            }
        }

        delta = new float[L - 1][];
        for (int l = 0; l < L - 1; l++) {
            delta[l] = new float[n[l + 1]];
            for (int i = 0; i < n[l + 1]; i++) {
                delta[l][i] = 0;
            }
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

    public float onlineBackPropagation() throws IOException {
        init();
        dataset.scale(0.1f, 0.9f);
        File errorFile = new File("src\\resources\\" + dataset.getFilename().substring(0, dataset.getFilename().length() - 4) + "_error.txt");
        File predictionFile = new File("src\\resources\\" + dataset.getFilename().substring(0, dataset.getFilename().length() - 4) + "_prediction.txt");
        createOrEraseFile(errorFile);
        createOrEraseFile(predictionFile);
        float globalError = 0;
        List<Integer> trainingIndexes = getTrainingIndexes();
        List<Integer> testingIndexes = getTestIndexes(trainingIndexes);
        for (int epoch = 0; epoch < epochs; epoch++) {
            globalError = 0;
            for (Integer i : trainingIndexes) {
                xi[0] = dataset.getInputs()[i];
                feedForwardPropagation();
                errorBackPropagation(i);
                updateWeightsAndTheta();
                float sum = 0;
                for (int j = 0; j < n[L - 1]; j++) {
                    sum += Math.pow(xi[L - 1][j] - dataset.getOutputs()[i][j], 2);
                }
                globalError += sum / n[L - 1];
            }
            globalError /= trainingIndexes.size();
            dumpErrorToFile(errorFile, globalError);
        }
        globalError = 0;
        FileWriter fw = new FileWriter(predictionFile, true);
        for (Integer i : testingIndexes) {
            xi[0] = dataset.getInputs()[i];
            feedForwardPropagation();
            float sum = 0;
            for (int j = 0; j < n[L - 1]; j++) {
                sum += Math.abs((dataset.getOutputs()[i][j] - xi[L - 1][j]) / dataset.getOutputs()[i][j]);
            }
            globalError += sum / n[L - 1];
            fw.write(dataset.unscaleValue(0, 0.1f, 0.9f, dataset.getOutputs()[i][0]) + " " +
                    dataset.unscaleValue(0, 0.1f, 0.9f, xi[L - 1][0]));
            fw.write(System.lineSeparator());
        }
        fw.close();
        System.out.println("Global MAPE: " + (globalError / testingIndexes.size() * 100 + "%"));
        dataset.unscale(0.1f, 0.9f);
        return globalError / (dataset.getNum_samples() * (1 - training_ratio));
    }

    private static void dumpErrorToFile(File errorFile, float globalError) {
        try {
            FileWriter fw = new FileWriter(errorFile, true);
            fw.write(globalError + "\n");
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void createOrEraseFile(File errorFile) {
        try {
            errorFile.createNewFile();
            PrintWriter writer = new PrintWriter(errorFile);
            writer.print("");
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private ErrorAndEpoch cvBackPropagation(List<Integer> validationIndexes, List<Integer>... trainingIndexes) {
        init();
        List<Integer> mergedTrainingIndexes = new ArrayList<>();
        for (List<Integer> trainingIndex : trainingIndexes) {
            mergedTrainingIndexes.addAll(trainingIndex);
        }
        float globalError;
        int best_epoch = (int) epochs;
        float global_min_error = Float.MAX_VALUE;
        for (int epoch = 0; epoch < epochs; epoch++) {
            globalError = 0;
            for (Integer i : mergedTrainingIndexes) {
                xi[0] = dataset.getInputs()[i];
                feedForwardPropagation();
                errorBackPropagation(i);
                updateWeightsAndTheta();
                float sum = 0;
                for (int j = 0; j < n[L - 1]; j++) {
                    sum += Math.pow(xi[L - 1][j] - dataset.getOutputs()[i][j], 2);
                }
                globalError += sum / n[L - 1];
            }
            globalError /= mergedTrainingIndexes.size();
            if (globalError < global_min_error) {
                global_min_error = globalError;
                best_epoch = epoch;
            }
        }
        globalError = 0;
        for (Integer i : validationIndexes) {
            xi[0] = dataset.getInputs()[i];
            feedForwardPropagation();
            float sum = 0;
            for (int j = 0; j < n[L - 1]; j++) {
                sum += Math.pow(xi[L - 1][j] - dataset.getOutputs()[i][j], 2);
            }
            globalError += sum / n[L - 1];
        }
        return new ErrorAndEpoch(globalError / validationIndexes.size(), best_epoch);
    }

    public void crossValidation() {
        float globalError = 0;
        dataset.scale(0.1f, 0.9f);
        List<Integer> trainingIndexes = getTrainingIndexes();
        List<Integer> testIndexes = getTestIndexes(trainingIndexes);
        List<Integer> s1 = new ArrayList<>();
        List<Integer> s2 = new ArrayList<>();
        List<Integer> s3 = new ArrayList<>();
        List<Integer> s4 = new ArrayList<>();
        for (int i = 0; i < trainingIndexes.size(); i++) {
            if (i % 4 == 0) {
                s1.add(trainingIndexes.get(i));
            } else if (i % 4 == 1) {
                s2.add(trainingIndexes.get(i));
            } else if (i % 4 == 2) {
                s3.add(trainingIndexes.get(i));
            } else {
                s4.add(trainingIndexes.get(i));
            }
        }
        float bestError = Float.MAX_VALUE;
        for (float learning_rate : learning_rate_list) {
            for (float momentum : momentum_list) {
                for (FactType fact : fact_list) {
                    for (int hidden_layers : hidden_layers_list) {
                        List<List<Integer>> allCombinations = getAllCombinations(hidden_layers);
                        for (List<Integer> hidden_neurons_per_layer : allCombinations) {
                            this.loadFromGetBestParams(500, learning_rate, momentum, hidden_layers, hidden_neurons_per_layer.stream().mapToInt(i -> i).toArray(), fact);
                            ErrorAndEpoch error1 = cvBackPropagation(s1, s2, s3, s4);
                            ErrorAndEpoch error2 = cvBackPropagation(s2, s1, s3, s4);
                            ErrorAndEpoch error3 = cvBackPropagation(s3, s1, s2, s4);
                            ErrorAndEpoch error4 = cvBackPropagation(s4, s1, s2, s3);
                            float error = (error1.getError() + error2.getError() + error3.getError() + error4.getError()) / 4;
                            int epoch = (error1.getEpoch() + error2.getEpoch() + error3.getEpoch() + error4.getEpoch()) / 4;
                            if (error < bestError) {
                                bestError = error;
                                setBestValue(epoch, learning_rate, momentum, hidden_layers, hidden_neurons_per_layer.stream().mapToInt(i -> i).toArray(), fact);
                            }
                        }
                    }
                }
            }
        }
        this.loadFromGetBestParams(this.best_epochs, this.best_learning_rate, this.best_momentum, this.best_hidden_layers, this.best_hidden_neurons, this.best_fact);
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Integer i : trainingIndexes) {
                xi[0] = dataset.getInputs()[i];
                feedForwardPropagation();
                errorBackPropagation(i);
                updateWeightsAndTheta();
            }
        }
        for (Integer i : testIndexes) {
            xi[0] = dataset.getInputs()[i];
            feedForwardPropagation();
            float sum = 0;
            for (int j = 0; j < n[L - 1]; j++) {
                sum += Math.abs((dataset.getOutputs()[i][j] - xi[L - 1][j]) / dataset.getOutputs()[i][j]);
            }
            globalError += sum / n[L - 1];
        }
        System.out.println("Global MAPE: " + (globalError / testIndexes.size()) * 100 + "%");
        System.out.println("Best epochs: " + best_epochs);
        System.out.println("Best learning rate: " + best_learning_rate);
        System.out.println("Best momentum: " + best_momentum);
        System.out.println("Best hidden layers: " + best_hidden_layers);
        System.out.println("Best hidden neurons: " + Arrays.toString(best_hidden_neurons));
        System.out.println("Best fact: " + best_fact);
    }

    private void loadFromGetBestParams(int epoch, float learning_rate, float momentum, int hidden_layers, int[] hidden_neurons, FactType fact) {
        this.epochs = epoch;
        this.learning_rate = learning_rate;
        this.momentum = momentum;
        this.L = hidden_layers + 2;
        n = new int[L];
        n[0] = dataset.getInputs()[0].length;
        if (hidden_layers >= 0) System.arraycopy(hidden_neurons, 0, n, 1, hidden_layers);
        n[L - 1] = 1;
        this.fact = fact;

    }

    private List<List<Integer>> getAllCombinations(int hidden_layers) {
        List<List<Integer>> combinations = new ArrayList<>();
        for (int k : possible_neurons_per_layer) {
            List<Integer> combination = new ArrayList<>();
            combination.add(k);
            combinations.add(combination);
        }
        for (int i = 1; i < hidden_layers; i++) {
            List<List<Integer>> newCombinations = new ArrayList<>();
            for (List<Integer> combination : combinations) {
                for (int k : possible_neurons_per_layer) {
                    List<Integer> newCombination = new ArrayList<>(combination);
                    newCombination.add(k);
                    newCombinations.add(newCombination);
                }
            }
            combinations = newCombinations;
        }
        return combinations;
    }


    private void setBestValue(int epoch, float learning_rate, float momentum, int hidden_layers, int[] hidden_neurons, FactType fact) {
        this.best_epochs = epoch;
        this.best_learning_rate = learning_rate;
        this.best_momentum = momentum;
        this.best_hidden_layers = hidden_layers;
        this.best_hidden_neurons = hidden_neurons;
        this.best_fact = fact;
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
        return trainingIndexes;
    }

    private List<Integer> getTestIndexes(List<Integer> trainingIndexes) {
        List<Integer> testIndexes = new ArrayList<>();
        for (int i = 0; i < dataset.getNum_samples(); i++) {
            if (!trainingIndexes.contains(i)) {
                testIndexes.add(i);
            }
        }
        return testIndexes;
    }
}
