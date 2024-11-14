import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticDecoupling(nn.Module):

    def __init__(self, classNum, imgFeatureDim, wordFeatureDim, intermediaDim=1024):
        super(SemanticDecoupling, self).__init__()

        self.classNum = classNum
        self.imgFeatureDim = imgFeatureDim
        self.wordFeatureDim = wordFeatureDim
        self.intermediaDim = intermediaDim

        self.fc1 = nn.Linear(self.imgFeatureDim, self.intermediaDim, bias=False)
        self.fc2 = nn.Linear(self.wordFeatureDim, self.intermediaDim, bias=False)
        self.fc3 = nn.Linear(self.intermediaDim, self.intermediaDim)
        self.fc4 = nn.Linear(self.intermediaDim, 1)

    def forward(self, imgFeaturemap, wordFeatures, visualize=False):
        '''
        Shape of imgFeaturemap : (BatchSize, Channel, imgSize, imgSize), [32, 3, 224, 224]
        Shape of wordFeatures : (classNum, wordFeatureDim) [20,300]
        '''

        BatchSize, imgSize = imgFeaturemap.size()[0], imgFeaturemap.size()[3]
        #imgFeaturemap = torch.transpose(torch.transpose(imgFeaturemap, 1, 2), 2,3)  # BatchSize * imgSize * imgSize * Channel

        imgFeature = imgFeaturemap.contiguous().view(BatchSize * imgSize * imgSize,
                                                     -1)  # (BatchSize * imgSize * imgSize) * Channel
        #imgFeature = imgFeaturemap.contiguous().view(32, 49, -1)

        # this is what breaks, 32 * 224 * 224 is massive
        # should be passing in the featureMap which should be much smaller like 7x7
        # correction the OG is 32,
        # print("IMAGE FEATURE MAP: ", imgFeaturemap.shape)
        # print("IMAGE SHAPE2: ", imgFeature.shape)
        # print("WORD SHAPE2: ", wordFeatures.shape)
        wordFeatures = wordFeatures.to('cuda')
        imgFeature = self.fc1(imgFeature).view(BatchSize * imgSize * imgSize, 1, -1).repeat(1, self.classNum,
                                                                                            1)  # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        wordFeature = self.fc2(wordFeatures).view(1, self.classNum, self.intermediaDim).repeat(
            BatchSize * imgSize * imgSize, 1, 1)  # (BatchSize * imgSize * imgSize) * classNum * intermediaDim
        feature = self.fc3(torch.tanh(imgFeature * wordFeature).view(-1,
                                                                     self.intermediaDim))  # (BatchSize * imgSize * imgSize * classNum) * intermediaDim

        Coefficient = self.fc4(feature)  # (BatchSize * imgSize * imgSize * classNum) * 1
        Coefficient = torch.transpose(
            torch.transpose(Coefficient.view(BatchSize, imgSize, imgSize, self.classNum), 2, 3), 1, 2).view(BatchSize,
                                                                                                            self.classNum,
                                                                                                            -1)
        Coefficient = F.softmax(Coefficient, dim=2)  # BatchSize * classNum * (imgSize * imgSize))
        Coefficient = Coefficient.view(BatchSize, self.classNum, imgSize,
                                       imgSize)  # BatchSize * classNum * imgSize * imgSize
        Coefficient = torch.transpose(torch.transpose(Coefficient, 1, 2), 2,
                                      3)  # BatchSize * imgSize * imgSize * classNum
        Coefficient = Coefficient.view(BatchSize, imgSize, imgSize, self.classNum, 1).repeat(1, 1, 1, 1,
                                                                                             self.imgFeatureDim)  # BatchSize * imgSize * imgSize * classNum * imgFeatureDim

        featuremapWithCoefficient = imgFeaturemap.view(BatchSize, imgSize, imgSize, 1, self.imgFeatureDim).repeat(1, 1,
                                                                                                                  1,
                                                                                                                  self.classNum,
                                                                                                                  1) * Coefficient  # BatchSize * imgSize * imgSize * classNum * imgFeatureDim
        semanticFeature = torch.sum(torch.sum(featuremapWithCoefficient, 1), 1)  # BatchSize * classNum * imgFeatureDim

        if visualize:
            return semanticFeature, torch.sum(torch.abs(featuremapWithCoefficient), 4), Coefficient[:, :, :, :, 0]
        return semanticFeature, featuremapWithCoefficient, Coefficient[:, :, :, :, 0]
        # Transform image features and word features
        # wordFeatures = wordFeatures.to('cuda')
        # imgFeature = self.fc1(imgFeature)  # [BatchSize, imgSize * imgSize, intermediaDim]
        # wordFeature = self.fc2(wordFeatures)  # [classNum, intermediaDim]
        #
        # # Prepare for broadcasting
        # imgFeature = imgFeature.unsqueeze(2)  # [BatchSize, imgSize * imgSize, 1, intermediaDim]
        # wordFeature = wordFeature.unsqueeze(0).unsqueeze(0)  # [1, 1, classNum, intermediaDim]
        #
        # # Repeat word features to match image features' batch size and spatial dimensions
        # wordFeature = wordFeature.repeat(BatchSize, imgSize, 1,
        #                                  1)  # [BatchSize, imgSize * imgSize, classNum, intermediaDim]
        #
        # # Element-wise multiplication and reshaping
        # feature = torch.tanh(imgFeature * wordFeature)  # [BatchSize, imgSize * imgSize, classNum, intermediaDim]
        # feature = feature.view(-1, self.intermediaDim)  # [BatchSize * imgSize * imgSize * classNum, intermediaDim]
        #
        # # Computing coefficients
        # Coefficient = self.fc4(feature)  # [BatchSize * imgSize * imgSize * classNum, 1]
        # Coefficient = Coefficient.view(BatchSize, imgSize, self.classNum,
        #                                1)  # [BatchSize, imgSize * imgSize, classNum, 1]
        #
        # # Apply softmax across spatial dimension
        # Coefficient = F.softmax(Coefficient, dim=1)  # [BatchSize, imgSize * imgSize, classNum, 1]
        #
        # # Weighting image features with coefficients
        # featuremapWithCoefficient = imgFeaturemap.view(BatchSize, imgSize, 1,
        #                                                self.imgFeatureDim) * Coefficient  # [BatchSize, imgSize * imgSize, classNum, imgFeatureDim]
        #
        # # Summing across spatial dimensions
        # semanticFeature = torch.sum(torch.sum(featuremapWithCoefficient, 1), 1)  # [BatchSize, classNum, imgFeatureDim]
        #
        # if visualize:
        #     return semanticFeature, torch.sum(torch.abs(featuremapWithCoefficient), 3), Coefficient.squeeze(-1)
        # return semanticFeature, featuremapWithCoefficient, Coefficient.squeeze(-1)


class TransformerSemanticDecoupling(nn.Module):
    def __init__(self, class_embeddings):
        super().__init__()
        self.class_embeddings = class_embeddings  # pre-computed embeddings for each class

        # embed_dim is the dimension of the patch's embedding
        embed_dim = 512
        self.fc = nn.Linear(embed_dim, 1)  # for computing weights

    def forward(self, patch_embeddings, device):
        #patch_embeddings = self.transformer.forward_features(image)  # assuming this method gives patch level embeddings

        #print("patch_embeddings SHAPE: ", patch_embeddings.shape) # currently [32, 196, 512] 196 patches with a dimension of 512
        patch_embeddings = patch_embeddings.to(device)
        self.class_embeddings = self.class_embeddings.to(device)

        class_specific_features = []
        for class_embed in self.class_embeddings:
            interaction = patch_embeddings * class_embed  # broadcasted multiplication

            # squash the interaction terms to be in the range [-1, 1] to help stabilize training and ensure values dont explode or vanish
            interaction = torch.tanh(interaction)

            # Compute attention or weight for each patch with respect to the class
            # calculate how important each patch is for each class then use the importance to weight the patch's contribution
            weights = torch.sigmoid(self.fc(interaction))  # shape: [batch_size, num_patches]

            # Compute class-specific feature as weighted sum of patch embeddings
            class_feature = torch.sum(weights * patch_embeddings, dim=1)
            class_specific_features.append(class_feature)

        return torch.stack(class_specific_features, dim=1)
        # Expanding dimensions to allow broadcasting during multiplication
        # patch_embeddings: [batch_size, 1, num_patches, embed_dim]
        # class_embeddings: [1, num_classes, 1, embed_dim]
        # expanded_patch_embeddings = patch_embeddings[:, None, :, :]
        # expanded_class_embeddings = self.class_embeddings[None, :, None, :]
        #
        # # Interaction: [batch_size, num_classes, num_patches, embed_dim]
        # interaction = torch.tanh(expanded_patch_embeddings * expanded_class_embeddings)
        #
        # # Compute weights for each patch with respect to the class
        # # Here, ensure the linear layer and sigmoid apply along the correct dimension
        # # You might need to adjust the fc layer input features or reshape interaction tensor
        # weights = torch.sigmoid(self.fc(interaction))
        #
        # # Ensure the dimensions are correct for the weighted sum
        # class_feature = torch.sum(weights * expanded_patch_embeddings, dim=2)
        #
        # return class_feature