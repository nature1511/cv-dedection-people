from torch.autograd import Variable
from tqdm.auto import tqdm


def train_one_loop(model, optimizer, loss_func, train_dataloader, device):
    print("Training")
    loss_sum = 0
    i = 0
    for data in tqdm(train_dataloader, total=len(train_dataloader)):
        # for nbatch, data in enumerate(train_dataloader):
        i += 1
        optimizer.zero_grad()

        img, _, _, bbox_data, bbox_labels = data

        gloc = Variable(bbox_data.transpose(1, 2).contiguous(), requires_grad=False).to(
            device
        )
        glabel = Variable(bbox_labels, requires_grad=False).to(device)
        #
        ploc, plabel = model(img.to(device))
        # print(ploc.device, plabel.device, gloc.device, glabel.device)
        loss = loss_func(ploc=ploc, plabel=plabel, gloc=gloc, glabel=glabel)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

    return loss_sum / len(train_dataloader)
