class Node {
  public:
    Node(int input_count, float *inputs, float *weights) {
      this->input_count = input_count;
      this->inputs = inputs;
      this->weights = weights;
    }
    float activate() {
      float sum = 0.0f;
      for (int i = 0; i < input_count; i++) {
        sum += inputs[i] * weights[i];
      }
      return tanh(sum);
    }
  private:
    int input_count;
    float *inputs;
    float *weights;
};

void invoke_nn(int input_length, float *inputs, int output_length, float *outputs, float *weights) {
  for (int i = 0; i < output_length; i++) {
    Node node = Node(input_length, inputs, weights);
    *outputs = node.activate();
    weights += input_length;
    outputs += 1;
  }
}