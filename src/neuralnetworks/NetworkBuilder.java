package neuralnetworks;

import neuralnetworks.functions.activationfunctions.Sigmoid;
import neuralnetworks.layer.DenseLayer;
import neuralnetworks.layer.InputLayer;
import neuralnetworks.layer.Layer;
import neuralnetworks.layer.OutputLayer;
import neuralnetworks.tools.ArrayTools;

import java.util.ArrayList;

/**
 * Created by finne on 25.01.2018.
 */
public class NetworkBuilder {

    InputLayer inputLayer;

    ArrayList<Layer> layers = new ArrayList<>();

    public NetworkBuilder(int input_depth, int input_width, int input_height) {
        inputLayer = new InputLayer(input_depth, input_width, input_height);

        inputLayer.setOutput_error_values(new double[input_depth][input_width][input_height]);
        inputLayer.setOutput_derivative_values(new double[input_depth][input_width][input_height]);
        inputLayer.setOutput_values(new double[input_depth][input_width][input_height]);
    }

    public NetworkBuilder addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public Network buildNetwork() {
        try{
            Layer b = inputLayer;
            for(Layer l: layers){
                l.connectToPreviousLayer(b);
                b = l;
            }
            OutputLayer outputLayer = new OutputLayer();
            outputLayer.connectToPreviousLayer(b);

            return new Network(inputLayer, outputLayer);
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String[] args) {
        NetworkBuilder networkBuilder = new NetworkBuilder(1,1,5);
        networkBuilder.addLayer(new DenseLayer(5)
                .weightsRange(0,1)
                .biasRange(-1,1)
                .setActivationFunction(new Sigmoid()));
        networkBuilder.addLayer(new DenseLayer(3)
                .weightsRange(0,1)
                .biasRange(0,1)
                .setActivationFunction(new Sigmoid()));
        networkBuilder.addLayer(new DenseLayer(2)
                .weightsRange(0,1)
                .biasRange(-1,1)
                .setActivationFunction(new Sigmoid()));

        Network network = networkBuilder.buildNetwork();

        double[][][] input = ArrayTools.createComplexFlatArray(0.3,0.6,0.1,0.9,.1);
        double[][][] ex = ArrayTools.createComplexFlatArray(1,0);
        double eta = 0.3;

        for(int i = 0; i < 10000; i++) {
            network.train(input, ex, eta);
        }

        network.analyseNetwork();

        Layer.printArray(network.calculate(input));

    }

}
