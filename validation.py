import torch
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import AverageMeter, calculate_accuracy

def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    print('Validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # To collect predictions for confusion matrix
    all_preds = []
    all_labels = []
    class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    end_time = time.time()
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        # Apply masking if modality is dropped
        if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise':
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual)*torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise':
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio)*torch.randn(inputs_audio.size()))
            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())

        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1],
                                              inputs_visual.shape[2],
                                              inputs_visual.shape[3],
                                              inputs_visual.shape[4])

        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual.to(opt.device))
            inputs_audio = Variable(inputs_audio.to(opt.device))
            targets = Variable(targets)

        outputs = model(inputs_audio, inputs_visual)
        loss = criterion(outputs, targets)

        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))
        losses.update(loss.data, inputs_audio.size(0))

        # Collect predictions
        _, preds = torch.max(outputs.data, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
            epoch, i + 1, len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            top1=top1,
            top5=top5))

    # Log to TSV or console
    logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item()
    })

    # ✅ Draw confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix – Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_epoch{epoch}.png")
    plt.show()

    return losses.avg.item(), top1.avg.item()
