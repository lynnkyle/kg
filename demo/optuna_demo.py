import optuna
import torch
from torch import nn, optim

x = torch.randn(1000, 100)
y = torch.randint(low=0, high=10, size=(1000,))


class MLP(nn.Module):
    def __init__(self, n_layers, n_units, input_dim, output_dim):
        """
        :param n_layers: MLP层数
        :param n_units: MLP隐藏层的神经元个数
        :param input_dim: MLP上一层的神经元个数
        :param output_dim: MLP输出层的神经元个数
        """
        super(MLP, self).__init__()
        layers = []
        in_features = input_dim  # 该层输入维度
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            in_features = n_units
        layers.append(nn.Linear(in_features, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = trial.suggest_int('n_units', 32, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    model = MLP(n_layers, n_units, input_dim=x.shape[1], output_dim=10)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)
print("Best Trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
for key, value in trial.params.items():
    print(f"    {key}:{value}")
