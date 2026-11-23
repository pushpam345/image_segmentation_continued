import torch


def calculate_iou_dataset(model, dataloader, num_classes, device):

    model.eval()

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for cls in range(num_classes):
                pred_cls = (preds == cls)
                target_cls = (masks == cls)

                intersection[cls] += (pred_cls & target_cls).sum()
                union[cls] += (pred_cls | target_cls).sum()

    iou_per_class = intersection / (union + 1e-6)

    mean_iou = torch.mean(iou_per_class)

    return iou_per_class, mean_iou
