public class ErrorAndEpoch {
    private float error;
    private int epoch;

    public ErrorAndEpoch(float error, int epoch) {
        this.error = error;
        this.epoch = epoch;
    }

    public float getError() {
        return error;
    }

    public int getEpoch() {
        return epoch;
    }
}
