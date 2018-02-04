package neuralnetworks;

import neuralnetworks.data.TrainSet;
import neuralnetworks.functions.errorfunctions.ErrorFunction;
import neuralnetworks.layer.InputLayer;
import neuralnetworks.layer.Layer;
import neuralnetworks.layer.OutputLayer;
import neuralnetworks.tools.ArrayTools;

/**
 * Created by finne on 02.02.2018.
 */
public class Network {

    private InputLayer inputLayer;
    private OutputLayer outputLayer;

    public Network(InputLayer inputLayer, OutputLayer outputLayer) {
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
    }

    public void printArchitecture() {
        Layer cur = inputLayer;

        System.out.println(cur.getClass().getSimpleName());
        System.out.println(">>>>> ns: [" + cur.getOUTPUT_DEPTH() + " "+ cur.getOUTPUT_WIDTH() + " " + cur.getOUTPUT_HEIGHT() +  "] <<<<<<<");

        while(cur.getNext_layer() != null) {
            System.out.println ("");
            cur = cur.getNext_layer();
            System.out.println(cur.getClass().getSimpleName());
            System.out.println(">>>>> ns: [" + cur.getOUTPUT_DEPTH() + " "+ cur.getOUTPUT_WIDTH() + " " + cur.getOUTPUT_HEIGHT() +  "] <<<<<<<");
        }
    }

    public Network setErrorFunction(ErrorFunction errorFunction) {
        this.outputLayer.setErrorFunction(errorFunction);
        return this;
    }

    public double[][][] calculate(double[][][] in) {
        if(this.inputLayer.matchingDimensions(in) == false) return null;
        this.inputLayer.setInput(in);
        this.inputLayer.feedForwardRecursive();
        return getOutput();
    }

    public void backpropagateError(double[][][] expectedOutput) {
        if(this.outputLayer.matchingDimensions(expectedOutput) == false) return;
        this.outputLayer.calculateOutputErrorSignals(expectedOutput);
        this.outputLayer.feedBackwardRecursive();
    }

    public void updateWeights(double eta) {
        this.inputLayer.updateWeightsRecursive(eta);
    }

    public void train(double[][][] input, double[][][] expected, double eta) {
        if(this.inputLayer.matchingDimensions(input) == false ||
                this.outputLayer.matchingDimensions(expected) == false) {
            return;
        }
        this.calculate(input);
        this.backpropagateError(expected);
        this.updateWeights(eta);
    }

    public void train(TrainSet trainSet, int iterations, int batch_size, double eta) {

        for(int it = 0; it < iterations; it++){
            TrainSet batch = trainSet.extractBatch(batch_size);
            for(int k = 0; k < batch.size(); k++) {
                train(batch.getInput(k), batch.getOutput(k),eta);
            }
            System.out.println(it + "   " + this.overall_error(batch));
        }
    }

    public double overall_error(TrainSet trainSet) {
        double t = 0;
        for(int i = 0; i < trainSet.size(); i++) {
            this.calculate(trainSet.getInput(i));
            t += this.outputLayer.overall_error(trainSet.getOutput(i));
        }
        return t/(double)trainSet.size();
    }

    public void analyseNetwork() {
        Layer cur = inputLayer;

        System.out.println(cur.getClass().getSimpleName());
        System.out.println(">>>>> ns: [" + cur.getOUTPUT_DEPTH() + " "+ cur.getOUTPUT_WIDTH() + " " + cur.getOUTPUT_HEIGHT() +  "] <<<<<<<");

        while(cur.getNext_layer() != null) {
            System.out.println("################################################################################################");
            System.out.println ("");
            cur = cur.getNext_layer();
            System.out.println(cur.getClass().getSimpleName());
            System.out.println(">>>>> ns: [" + cur.getOUTPUT_DEPTH() + " "+ cur.getOUTPUT_WIDTH() + " " + cur.getOUTPUT_HEIGHT() +  "] <<<<<<<");
            Layer.printArray(cur.getOutput_values());
            Layer.printArray(cur.getOutput_derivative_values());
            Layer.printArray(cur.getOutput_error_values());
        }
    }
    
    public double[][][] getOutput( ){
        return ArrayTools.copyArray(this.outputLayer.getOutput_values());
    }

    public int getINPUT_DEPTH(){
        return inputLayer.getOUTPUT_DEPTH();   
    }
    public int getINPUT_WIDTH(){
        return inputLayer.getOUTPUT_DEPTH();
    }
    public int getINPUT_HEIGHT(){
        return inputLayer.getOUTPUT_DEPTH();
    }
    public int getOUTPUT_DEPTH(){
        return outputLayer.getOUTPUT_DEPTH();
    }
    public int getOUTPUT_WIDTH(){
        return outputLayer.getOUTPUT_DEPTH();
    }
    public int getOUTPUT_HEIGHT(){
        return outputLayer.getOUTPUT_DEPTH();
    }
}
