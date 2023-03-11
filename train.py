from dataset import *


melspectrogram_train_dataset, melspectrogram_test_dataset = load_images()
class_map=melspectrogram_train_dataset.class_to_idx
print("\nClass category and index of the images: {}\n".format(class_map))


train_dataloader = torch.utils.data.DataLoader(
    melspectrogram_train_dataset,
    batch_size=4,
    shuffle=True                                           
)
test_dataloader = torch.utils.data.DataLoader(
    melspectrogram_test_dataset,
    batch_size=4,
    shuffle=True
)



class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(6,6))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(6,6))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(6,6))
        self.conv4 = nn.Conv2d(64,128, kernel_size=(6,6))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(6,6))
        self.conv_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7680, 200)
        self.fc2 = nn.Linear(200, 50)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(F.max_pool2d(self.conv5(x), 2))
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

model = CNNet()


cost = torch.nn.CrossEntropyLoss()

# used to create optimal parameters
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        

        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f'loss: {loss.item()}')


# Create the validation/test function

def test(dataloader, model, k_classes):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, top_1_correct, top_k_correct, top_k_prev_correct = 0, 0, 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            pred = model(X)

            test_loss += cost(pred, Y).item()
            top_1_correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
            k_pred = pred.topk(k_classes, dim=1).indices
            k_pred_prev = pred.topk(k_classes-1, dim=1).indices
            for dim in range(len(Y)):
                if (Y[dim].item() in k_pred[dim]):
                    top_k_correct += 1
                if (Y[dim].item() in k_pred_prev[dim]):
                    top_k_prev_correct += 1
                

    top_1_correct /= size
    top_k_correct /= size
    top_k_prev_correct /= size
    print(f'\nTest Error:\ntop 1 acc: {(100*top_1_correct):>0.1f}%,top {k_classes-1} acc:{(100*top_k_prev_correct):>0.1f}%, top {k_classes} acc:{(100*top_k_correct):>0.1f}%')
    

epochs = 25

for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model, 3)
print('Done!')

torch.save(model.state_dict(), 'trained_for_spectrogram_resized_50.pth')