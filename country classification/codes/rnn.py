import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from st import ALL_LETTERS, N_LETTERS
from st import load_data, lettor_to_tensor, line_to_tensor, random_training_example

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
category_lines, all_categories = load_data()
n_categories = len(all_categories)
print(n_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 120000
memory_pool = []

def add_to_memory(sentence, category):
    memory_pool.append((sentence, category))
    if len(memory_pool)> 1000:
        memory_pool.pop(0)

def periodic_retraining():
    for sentence, category in memory_pool:
        category_tensor = torch.tensor([all_categories.index(category)], dtype= torch.long)
        line_tensor = line_to_tensor(sentence)
        train(line_tensor, category_tensor)
        
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG({category})"
        print(f"{i} {i/n_iters*100} {loss:.4f} {line} / {guess} {correct}")


def predict(input_line, true_category = None):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
        print(f"Prediction: {guess}")

    if true_category is not None:
        category_tensor = torch.tensor([all_categories.index(true_category)], dtype= torch.long)
        train(line_tensor,category_tensor)

iteration_count = 0
while True:
    sentence = input("Input:")
    if sentence == "quit":
        break

    predict(sentence)
    feedback = input("Is the prediction correct? (y/n): ")
    if feedback.lower() == 'n':
        true_category = input("What's the correct category?").capitalize()
        predict(sentence, true_category)
        add_to_memory(sentence, true_category)

    iteration_count += 1
    if iteration_count % 10 == 0:
        periodic_retraining()
    

