package neuralnetworks.layer;

import neuralnetworks.functions.activationfunctions.ActivationFunction;
import neuralnetworks.tools.ArrayTools;

/**
 * Created by finne on 02.02.2018.
 */
public class DenseLayer extends Layer {

    private double[][] weights;
    private double[]   bias;

    public DenseLayer(int OUTPUT_HEIGHT) {
        super(1, 1, OUTPUT_HEIGHT);
    }


    private double lowerWeightsRange = 0, upperWeigthsRange = 1;
    private double lowerBiasRange = 0, upperBiasRange = 1;

    public DenseLayer weightsRange(double lower, double upper) {
        this.lowerWeightsRange = lower;
        this.upperWeigthsRange = upper;
        return this;
    }

    public DenseLayer biasRange(double lower, double upper) {
        this.lowerBiasRange = lower;
        this.upperBiasRange = upper;
        return this;
    }


    private ActivationFunction activationFunction;

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public DenseLayer setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    @Override
    protected void calculateOutputDimensions() throws Exception {

    }

    @Override
    protected void on_build() throws Exception {
        weights = new double[this.OUTPUT_HEIGHT][this.INPUT_HEIGHT];
        bias = new double[this.OUTPUT_HEIGHT];

        ArrayTools.randomiseArray(weights, lowerWeightsRange, upperWeigthsRange);
        ArrayTools.randomiseArray(bias, lowerBiasRange, upperBiasRange);
    }

    @Override
    public void calculate() {
        for(int i = 0; i < this.OUTPUT_HEIGHT; i ++) {
            double sum = bias[i];
            for(int prevNeuron = 0; prevNeuron < this.INPUT_HEIGHT; prevNeuron++) {
                sum += weights[i][prevNeuron] * this.getPrev_layer().getOutput_values()[0][0][prevNeuron];
            }
            this.output_values[0][0][i] = activationFunction.activation(sum);
            this.output_derivative_values[0][0][i] = activationFunction.activation_prime(sum);
        }
    }

    @Override
    public void calculateSignalErrors() {
        for(int i = 0; i < this.INPUT_HEIGHT; i++) {
            double sum = 0;
            for(int n = 0; n < this.getOUTPUT_HEIGHT(); n++) {
                sum += weights[n][i] * output_error_values[0][0][n];
            }
            this.getPrev_layer().getOutput_error_values()[0][0][i] = this.getPrev_layer().getOutput_derivative_values()[0][0][i] * sum;
        }
    }

    @Override
    public void update_weights(double eta) {
        for(int i = 0; i < this.OUTPUT_HEIGHT; i++) {
            double delta = -eta * this.getOutput_error_values()[0][0][i];
            bias[i] += delta;

            for(int p = 0; p < this.INPUT_HEIGHT; p++) {
                weights[i][p] += delta * getPrev_layer().getOutput_values()[0][0][p];
            }
        }
    }
}
