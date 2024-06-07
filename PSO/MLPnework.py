import numpy as np

class MLPnetwork:
    def __init__(self, input_size=3, hidden_size=5, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化權重和偏差
        self.weights_input_hidden = np.zeros((self.hidden_size, self.input_size))
        self.bias_hidden = np.zeros((self.hidden_size, 1))
        self.weights_hidden_output = np.zeros((self.output_size, self.hidden_size))
        self.bias_output = np.zeros((self.output_size, 1))

    def set_params(self, params):
        # 將扁平化的參數設置回MLP網絡
        ih_size = self.hidden_size * self.input_size
        ho_size = self.output_size * self.hidden_size
        self.weights_input_hidden = params[:ih_size].reshape(self.hidden_size, self.input_size)
        self.bias_hidden = params[ih_size:ih_size + self.hidden_size].reshape(self.hidden_size, 1)
        self.weights_hidden_output = params[ih_size + self.hidden_size:ih_size + self.hidden_size + ho_size].reshape(self.output_size, self.hidden_size)
        self.bias_output = params[-self.output_size:].reshape(self.output_size, 1)

    def get_next_Th(self, inputs):
        inputs = np.array(inputs)
        hidden_inputs = np.dot(self.weights_input_hidden, inputs.reshape(-1, 1)) + self.bias_hidden
        hidden_outputs = hidden_inputs

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        final_outputs = final_inputs.flatten()  # 不使用激活函數或使用線性激活
        if final_outputs[0]<-40:
            final_outputs[0]=-40
        elif final_outputs[0]>40:
            final_outputs[0]=40
        return final_outputs[0]
