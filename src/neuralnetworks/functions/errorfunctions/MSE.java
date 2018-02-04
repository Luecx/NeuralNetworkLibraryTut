package neuralnetworks.functions.errorfunctions;

import neuralnetworks.layer.OutputLayer;

/**
 * Created by finne on 02.02.2018.
 */
public class MSE extends ErrorFunction {
    @Override
    public double overall_error(OutputLayer outputLayer, double[][][] ex) {
        double v = 0;
        double c = 0;

        for(int i = 0; i < outputLayer.getOutput_values().length; i++){
            for(int n = 0; n < outputLayer.getOutput_values()[0].length; n++) {
                for(int j = 0; j < outputLayer.getOutput_values()[0][0].length; j++){
                    v += (outputLayer.getOutput_values()[i][n][j] - ex[i][n][j]) * (outputLayer.getOutput_values()[i][n][j] - ex[i][n][j]);
                    c++;
                }
            }
        }

        return v / (2 * c);
    }

    @Override
    public void apply(OutputLayer outputLayer, double[][][] exp) {

        double[][][] output = outputLayer.getOutput_values();
        double[][][] output_derivative = outputLayer.getOutput_derivative_values();
        double[][][] error_signals = outputLayer.getOutput_error_values();

        for(int i = 0; i < outputLayer.getOutput_values().length; i++){
            for(int n = 0; n < outputLayer.getOutput_values()[0].length; n++) {
                for(int j = 0; j < outputLayer.getOutput_values()[0][0].length; j++){
                    error_signals[i][n][j] = output_derivative[i][n][j] * (output[i][n][j] - exp[i][n][j]);
                }
            }
        }

    }
}
