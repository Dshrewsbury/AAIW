from collections import OrderedDict

import torch
import torch.nn as nn
import math
from src.models.components import clip
from src.models.components.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
#from SemanticDecoupling import TransformerSemanticDecoupling

class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, device):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        tokenized_prompts = tokenized_prompts.to(device)
        # indices = torch.arange(x.shape[0], device='cuda')
        # x = x.half()
        # self.text_projection = nn.Parameter(self.text_projection.to(x.dtype))

        #x = x[indices, tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model, device):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = "a photo of a"
        n_ctx = 4
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = clip.tokenize(ctx_init)
        prompt = prompt.to(device)
        clip_model = clip_model.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],  # type: ignore
            dim=1,
        )
        return prompts

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #input = input.float()
        #adj = adj.float()
        #self.weight = self.weight.half()
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
# This is largely gotten from the VisualTransformer class of CLIP
# Thus I should adapt the ModifiedResNet portion of the code if I want to use resnet
class CLIPVIT(nn.Module):

    def __init__(self, topk, clip_model, device, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.clipzero = False
        self.device = device

        #self.semantic_decoupler = TransformerSemanticDecoupling()

        self.use_clip_proj = False

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj
        self.topk = topk
        self.logit_scale = clip_model.logit_scale
        # relation+voc.npy

        self.classnames = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                               'train', 'tvmonitor']

        # self.prompt_learner = PromptLearner(self.classnames, clip_model, self.device)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.text_encoder = TextEncoder(clip_model)


        #SCP Stuff
        # self.gc1 = GraphConvolution(512, 1024)
        # self.gc2 = GraphConvolution(1024, 1024)
        # self.gc3 = GraphConvolution(1024, 512)
        # self.relu = nn.LeakyReLU(0.2)
        # self.relu2 = nn.LeakyReLU(0.2)
        # self.num_classes = 20
        #
        # self.relation = torch.Tensor(
        #     np.load("/home/dan/LabelNoise/src/MultiLabel/Non_Lightning/SCPNet/relation+voc.npy"))

        # sparse_topk = 20
        # reweight_p = 0.2
        # T = 0.3
        #
        # _, max_idx = torch.topk(self.relation, sparse_topk)
        # mask = torch.ones_like(self.relation).type(torch.bool)
        # for i, idx in enumerate(max_idx):
        #     mask[i][idx] = 0
        # self.relation[mask] = 0
        # sparse_mask = mask
        # dialog = torch.eye(self.num_classes).type(torch.bool)
        # self.relation[dialog] = 0
        # self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1) * reweight_p
        # self.relation[dialog] = 1 - reweight_p
        #
        # self.gcn_relation = self.relation.clone()
        # assert (self.gcn_relation.requires_grad == False)
        # self.relation = torch.exp(self.relation / T) / torch.sum(torch.exp(self.relation / T), dim=1).reshape(
        #     -1, 1)
        # self.relation[sparse_mask] = 0
        # self.relation = self.relation / torch.sum(self.relation, dim=1).reshape(-1, 1)
    
    def forward_features(self, x):
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x

    def forward(self, x, label_embed, device, norm_pred=True):
        #tokenized_prompts = self.tokenized_prompts # int64. [20, 77]
        x = self.forward_features(x) #float 32

        dist_feat = x[:, 0] @ self.projection_dist

        # For Global Head Only Ablation
        if self.global_only:
            score = dist_feat @ label_embed.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default
        else:
            if not self.use_clip_proj:
            # pred_feat = self.projection(x[:, 1:])
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist

            # label embed are 20 x 512, 512 because that's BERT's embed dimensions
            #prompts = self.prompt_learner() #float32, [20, 77, 512]

            #text_features = self.text_encoder(prompts, tokenized_prompts, device) # 20, 512
            #identity = text_features
            #text_features = label_embed

            # think the 3 convolutions is to mirror resnet
            # text_features = self.gc1(text_features, self.gcn_relation.to(device))
            # text_features = self.relu(text_features)
            # text_features = self.gc2(text_features, self.gcn_relation.to(device))
            # text_features = self.relu2(text_features)
            # text_features = self.gc3(text_features, self.gcn_relation.to(device))
            #
            # text_features += identity
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # getting the topk local patches
            # was doing cosine similarity with just label_embed
            #pred_feat = pred_feat
            #dist_feat = dist_feat

            #score1 = torch.topk(pred_feat @ text_features.t(),k=self.topk, dim=1)[0].mean(dim=1)
            score1 = torch.topk(pred_feat @ label_embed.t(), k=self.topk, dim=1)[0].mean(dim=1)

            # global head features
            score2 = dist_feat @ label_embed.t()
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            # logits_per_image = logit_scale * image_features @ text_features.t()
            
            score = (score1 + score2) / 2
            #scaled_score = score * 20

            #return score, scaled_score, dist_feat
            return score, pred_feat, dist_feat

    def encode_img(self, x):
        # import pdb; pdb.set_trace()
        x = self.forward_features(x)
        if self.clipzero:
            x = x @ self.proj
            return x[:, 1:, :], x[:, 0, :]
        else:
            pred_feat = x[:, 1:] @ self.projection_dist
            # dist_feat = self.projection_dist(x[:, 0])
            dist_feat = x[:, 0] @ self.projection_dist
            return pred_feat, dist_feat