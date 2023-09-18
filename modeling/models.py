import torch
import torchvision


# ---- ENCODER

class Resnet(torch.nn.Module):
    def __init__(self, in_channels, n_classes=3, n_blocks=4, pretrained=False, mode='instance', aggregation='max',
                 weights="IMAGENET1K_V2", backbone="RN18"):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks  # From 1 to 4
        self.n_classes = n_classes
        self.embedding = []
        self.pretrained = pretrained
        self.mode = mode  # 'embedding', 'instance'
        self.aggregation = aggregation  # 'max', 'mean'
        self.backbone = backbone

        if self.backbone == "RN50":
            print("Using Resnet-50 as backbone", end="\n")
            self.model = torchvision.models.resnet50(weights=weights)
            self.nfeats = 4*(512 // (2 ** (4 - n_blocks)))
        else:
            print("Using Resnet-18 as backbone", end="\n")
            self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.nfeats = 512 // (2 ** (4 - n_blocks))
        if in_channels != 3:
            self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                         padding=(3, 3), bias=False)
        else:
            self.input = list(self.model.children())[0]
        self.classifier = torch.nn.Conv2d(in_channels=self.nfeats, out_channels=self.n_classes, kernel_size=(1, 1),
                                          bias=False)

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x, reshape_cam=True):
        # Input dimensions
        _, _, H, W = x.size()

        # Input block - channels normalization
        x = self.input(x)
        for iBlock in range(1, 4):
            x = list(self.model.children())[iBlock](x)

        # Feature extraction
        F = []
        for iBlock in range(4, self.n_blocks+4):
            x = list(self.model.children())[iBlock](x)
            F.append(x)
        self.embedding = x

        # Output cams logits
        cam = self.classifier(x)
        if reshape_cam:
            cam = torch.nn.functional.interpolate(cam, size=(H, W), mode='bilinear', align_corners=True)

        # Image-level output
        if self.mode == 'instance':
            if self.aggregation == 'max':
                pred = torch.squeeze(torch.nn.AdaptiveMaxPool2d(1)(cam))
            if self.aggregation == 'mean':
                pred = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(cam))
        elif self.mode == 'embedding':
            if self.aggregation == 'max':
                embedding = torch.nn.AdaptiveMaxPool2d(1)(x)
            if self.aggregation == 'mean':
                embedding = torch.nn.AdaptiveMaxPool2d(1)(x)
            pred = torch.squeeze(self.classifier(embedding))

        return pred, cam