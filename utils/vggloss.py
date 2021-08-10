import torch
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
  def __init__(self, resize=True, device='cuda'):
    super(VGGPerceptualLoss, self).__init__()
    blocks = []
    blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
    blocks.append(
      torchvision.models.vgg16(pretrained=True).features[4:9].eval())
    blocks.append(
      torchvision.models.vgg16(pretrained=True).features[9:16].eval())
    blocks.append(
      torchvision.models.vgg16(pretrained=True).features[16:23].eval())
    for bl in blocks:
      for p in bl:
        p.requires_grad = False
    self.blocks = torch.nn.ModuleList(blocks)
    self.blocks.to(device=device)
    self.transform = torch.nn.functional.interpolate
    self.mean = torch.nn.Parameter(
      torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)).to(device=device)
    self.std = torch.nn.Parameter(
      torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).to(device=device)
    self.resize = resize

  def forward(self, input, target, feature_layers=[0, 1, 2, 3],
              style_layers=[]):
    input = (input - self.mean) / self.std
    target = (target - self.mean) / self.std
    if self.resize:
      input = self.transform(input, mode='bilinear', size=(224, 224),
                             align_corners=False)
      target = self.transform(target, mode='bilinear', size=(224, 224),
                              align_corners=False)
    loss = 0.0
    x = input
    y = target
    for i, block in enumerate(self.blocks):
      x = block(x)
      y = block(y)
      if i in feature_layers:
        loss += torch.nn.functional.l1_loss(x, y)
      if i in style_layers:
        act_x = x.reshape(x.shape[0], x.shape[1], -1)
        act_y = y.reshape(y.shape[0], y.shape[1], -1)
        gram_x = act_x @ act_x.permute(0, 2, 1)
        gram_y = act_y @ act_y.permute(0, 2, 1)
        loss += torch.nn.functional.mse_loss(gram_x, gram_y)
    return loss
