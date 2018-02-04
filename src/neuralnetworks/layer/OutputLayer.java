package neuralnetworks.layer;

import neuralnetworks.functions.errorfunctions.ErrorFunction;
import neuralnetworks.functions.errorfunctions.MSE;

/**
 * Created by finne on 02.02.2018.
 */
public class OutputLayer extends Layer {

    public OutputLayer() {
    }

    private ErrorFunction errorFunction;

    public ErrorFunction getErrorFunction() {
        return errorFunction;
    }

    public OutputLayer setErrorFunction(ErrorFunction errorFunction) {
        this.errorFunction = errorFunction;
        return this;
    }

    public void calculateOutputErrorSignals(double[][][] ex){
        errorFunction.apply(this, ex);
    }

    public double overall_error(double[][][] ex) {
        return errorFunction.overall_error(this, ex);
    }

    @Override
    protected void calculateOutputDimensions() throws Exception {
        this.OUTPUT_HEIGHT = this.INPUT_HEIGHT;
        this.OUTPUT_WIDTH = this.INPUT_WIDTH;
        this.OUTPUT_DEPTH = this.INPUT_DEPTH;
    }

    @Override
    protected void on_build() throws Exception {
        if(this.errorFunction == null) this.errorFunction = new MSE();
    }

    @Override
    public void calculate() {
        this.output_values = this.getPrev_layer().output_values;
        this.output_derivative_values = this.getPrev_layer().output_derivative_values;
    }

    @Override
    public void calculateSignalErrors() {
        this.getPrev_layer().setOutput_error_values(this.output_error_values);
    }

    @Override
    public void update_weights(double eta) {

    }
}
