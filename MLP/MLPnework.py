import numpy as np

class MLPnetwork:
    def __init__(self, input_size, hidden_size=5, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化權重和偏差
        self.weights_input_hidden = np.random.uniform(-1, 1, (self.hidden_size, self.input_size))
        self.bias_hidden = np.zeros((self.hidden_size, 1))
        self.weights_hidden_output = np.random.uniform(-1, 1, (self.output_size, self.hidden_size))
        self.bias_output = np.zeros((self.output_size, 1))
    
    def load_data(self):
        if self.input_size == 3:
            filename="train4dAll.txt"
        else:
            filename="train6dAll.txt"
        with open(filename, 'r') as file:
            lines = file.readlines()
            dat = [list(map(float, line.split())) for line in lines]

        data = [line[:-1] for line in dat]
        target = [line[-1] for line in dat]
        
        # 正規化
        self.max_data = np.max(data, axis=0)
        generlized_data = data / self.max_data
         
        max_target = np.max(target)
        generlized_target = target / max_target

        self.data=generlized_data
        self.target=generlized_target

    def train(self, epochs=300):
        inputs=self.data
        targets=self.target
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 前向傳播
                hidden_inputs = np.dot(self.weights_input_hidden, inputs[i].reshape(-1, 1)) + self.bias_hidden
                hidden_outputs = hidden_inputs

                final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
                final_outputs = final_inputs

                # 計算誤差
                output_errors = targets[i].reshape(-1, 1) - final_outputs
                hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

                # 反向傳播，更新權重和偏差
                self.weights_hidden_output += self.learning_rate * np.dot(output_errors, hidden_outputs.T)
                self.bias_output += self.learning_rate * output_errors

                self.weights_input_hidden += self.learning_rate * np.dot(hidden_errors, inputs[i].reshape(1, -1))
                self.bias_hidden += self.learning_rate * hidden_errors


    def get_next_Th(self, inputs):
        inputs=np.array(inputs)/self.max_data
        hidden_inputs = np.dot(self.weights_input_hidden, inputs.reshape(-1, 1)) + self.bias_hidden
        hidden_outputs = hidden_inputs
    
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        final_outputs = final_inputs.flatten()  # 不使用激活函數或使用線性激活
        #print(len(final_outputs))
        
        return final_outputs[0]*40  #從-1 ~ 1 映射到 -40 ~ 40

#if __name__ == "__main__":
#    mlp_network=MLPnetwork()
#    # 載入訓練資料
#    mlp_network.load_data()
#
#    # 訓練 MLP 網路
#    epochs = 1000
#    mlp_network.train()
#
#    # 測試輸入
#    test_input = [19, 3, 3]
#
#    # 預測下一個方向盤角度
#    predicted_Th = mlp_network.get_next_Th(test_input)
#
#    # 顯示結果
#    print("Test Input:", test_input)
#    print("Predicted Steering Angle:", predicted_Th)
