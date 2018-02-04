package neuralnetworks.functions.activationfunctions;

/**
 * Created by finne on 02.02.2018.
 */
public class Sigmoid extends ActivationFunction {

    @Override
    public double activation(double x) {
       return 1d / (1 + Math.exp(-x));
    }

    @Override
    public double activation_prime(double x) {
        return activation(x) * (1 - activation(x));
    }
}
