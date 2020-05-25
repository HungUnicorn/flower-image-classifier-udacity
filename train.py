from util import save_model, load_data
from model import get_img_model
import argparse
from torch import nn, optim
import torch

ACCURACY_TARGET = 0.7


def train():
    args = cli()

    device = torch.device("cuda" if args.gpu else "cpu")
    print(f'Device {device}')
    model = get_img_model(args.hidden_units, args.arch)

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    trainloader, _, validationloader, class_to_idx = load_data(args.data_dir)

    _train(optimizer, args.epochs, trainloader, validationloader, device, model)

    save_model(args.save_dir, model, class_to_idx, args.hidden_units, args.arch)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", default="./")
    parser.add_argument("--arch", default="vgg11")
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--hidden_units", default=512, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--gpu", action="store_true")

    return parser.parse_args()


def _train(optimizer, epochs, trainloader, validationloader, device, model):
    criterion = nn.NLLLoss()

    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {validation_loss / len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(validationloader):.3f}")
                running_loss = 0
                model.train()

                if (accuracy / len(validationloader)) > ACCURACY_TARGET:
                    print(f"Reached defined accuracy")
                    break

        if (accuracy / len(validationloader)) > ACCURACY_TARGET:
            print(f"Reached defined accuracy")
            break


if __name__ == "__main__":
    train()
