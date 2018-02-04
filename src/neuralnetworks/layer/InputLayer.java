package neuralnetworks.layer;

/**
 * Created by finne on 02.02.2018.
 */
public class InputLayer extends Layer {

    public InputLayer(int OUTPUT_DEPTH, int OUTPUT_WIDTH, int OUTPUT_HEIGHT) {
        super(OUTPUT_DEPTH, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    }

    public void setInput(double[][][] in) {
        if(this.matchingDimensions(in)){
            this.output_values = in;
        }
    }

    @Override
    protected void calculateOutputDimensions() throws Exception {

    }

    @Override
    protected void on_build() throws Exception {

    }

    @Override
    public void calculate() {

    }

    @Override
    public void calculateSignalErrors() {

    }

    @Override
    public void update_weights(double eta) {

    }
}
