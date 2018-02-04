package neuralnetworks.functions.activationfunctions;

/**
 * Created by finne on 02.02.2018.
 */
public class ReLU extends ActivationFunction{

    @Override
    public double activation(double x) {
        if(x > 0) return x;
        return 0;
    }

    @Override
    public double activation_prime(double x) {
        if(x > 0) return 1;
        return 0;
    }
}
