package neuralnetworks.functions.errorfunctions;

import neuralnetworks.layer.OutputLayer;

/**
 * Created by finne on 02.02.2018.
 */
public abstract class ErrorFunction {

    public abstract double overall_error(OutputLayer outputLayer, double[][][] ex);

    public abstract void apply(OutputLayer outputLayer, double[][][] exp);

}
