class Node {
  public:
    Node(int input_count, float **inputs, float **weights) {
      this->input_count = input_count;
      this->inputs = inputs;
      this->weights = weights;
    }
    float activate() {
      float sum = 0.0f;
      for (int i = 0; i < input_count; i++) {
        sum += *(inputs[i]) * *(weights[i]);
      }
      value = activation_function(sum);
      return value;
    }
    float value;
  private:
    float activation_function(float v) {
      return tanh(v);
    }
    int input_count;
    float **inputs;
    float **weights;
};
