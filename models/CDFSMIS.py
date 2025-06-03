import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Res50Encoder
from torchvision import transforms
from timm.models.layers import DropPath
from torch.nn import Sequential as Seq
from .gcn_lib import Grapher, act_layer
from torchvision.transforms import InterpolationMode

class SelfAttention(nn.Module):
    def __init__(self, in_dim, embed_dim, num_heads=1):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)  # [1, 512, N] -> [1, N, 512]
        x = self.input_proj(x)  # [1, N, embed_dim]
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(attn_output + x)
        x = self.proj(x) + x

        return self.norm2(x).permute(0, 2, 1)  # [1, N, 512] -> [1, 512, N]
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.query_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # Q
        self.key_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)    # K
        self.value_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)  # V
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm([embed_dim, 64, 64])
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  

    def forward(self, A, B):
        pe = self._positional_encoding(A.shape[-2], A.shape[-1]).to(A.device)
        A = A + pe  

        A_ = self.query_proj(A).view(1, 512, -1).permute(0, 2, 1)  # [1, 4096, 512]
        B_ = self.key_proj(B).permute(0, 2, 1)  # [1, N, 512]
        V_ = self.value_proj(B).permute(0, 2, 1)  # [1, N, 512]

        attn_output, _ = self.mha(A_, B_, V_)
        out = self.norm_1(attn_output + A_)
        out = out.permute(0, 2, 1).view(1, 512, 64, 64)
        out = self.proj(out) + out

        return self.norm_2(out)

    def _positional_encoding(self, H, W):
        y_pos = torch.arange(H).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W).unsqueeze(0).repeat(H, 1)
        div_term = torch.exp(torch.arange(0, 512, 2) * -(torch.log(torch.tensor(10000.0)) / 512))
        pe = torch.zeros(512, H, W)
        pe[0::2] = torch.sin(x_pos * div_term[:, None, None])
        pe[1::2] = torch.cos(y_pos * div_term[:, None, None])
        return pe.unsqueeze(0)
    
class Up(nn.Module):
    """Upscaling, concat, conv"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),   
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                )
        self.res = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),)
        
        self.head = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels//2, out_channels, kernel_size=1), 
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature):
        """feature : 1, 512, 64, 64, map : 64, 64"""
        """return : 1, 1, 256, 256"""

        x = map * feature
        x1 = self.up(x)
        x2 = self.res(x1) + x1
        x2 = self.head(x2)
        x2 = self.sigmoid(x2)

        return x2
    
class Bottleneck(nn.Module):
    def __init__(self, in_dim=512):
        super(Bottleneck, self).__init__()
        self.in_dim = in_dim
        self.res1 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim//2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_dim//2, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim//4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(self.in_dim//4, 1, kernel_size=1, padding=0, bias=False))

    def forward(self, x):
        """ N -> 1"""
        x = self.res1(x)
        x = self.res2(x) + x
        out = self.down(x)

        return out  
    
class SubgraphMatching(nn.Module):

    def __init__(self, hidden_dim, proj_dim, self_update_dim):
        super().__init__()
        self.support_proj = nn.Linear(hidden_dim, proj_dim)
        self.query_proj = nn.Linear(hidden_dim, proj_dim)
        self.self_update_proj = nn.Sequential(
            nn.Linear(hidden_dim, self_update_dim), 
            nn.ReLU(),
            nn.Linear(self_update_dim, hidden_dim))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dim = hidden_dim
        self.neck = Bottleneck(hidden_dim)
        self.up = Up(hidden_dim, hidden_dim//2, 1)
        self.model_init()

    def forward(self, query_graph, support_subgraph, spatial_shape):
        """
        Args:
            support_subgraph: [n, bs, c]
            query_graph: [hw, bs, c]
            spatial_shape: h, w
        """
        h, w = spatial_shape
        shotcut = query_graph
        query_graph = query_graph.view(1, self.dim, h*w).permute(2, 0, 1)
        query_graph = query_graph.transpose(0, 1)  # [bs, hw, c]
        support_subgraph = support_subgraph.transpose(0, 1)  # [bs, query, c]

        fs_proj = self.support_proj(support_subgraph)  # [bs, query, c]
        fq_proj = self.query_proj(query_graph)  # [bs, hw, c]
        channel_reweight = self.tanh(self.self_update_proj(fs_proj))  # [bs, query, c]

        fs_feat = (channel_reweight + 1) * fs_proj  # [bs, query, c]
        connectivity = torch.bmm(fq_proj, fs_feat.transpose(1, 2))  # [bs, hw, query]
        connectivity = connectivity.transpose(1, 2).reshape(-1, h, w)  # [query, h, w]
        connectivity = self.sigmoid(connectivity)
        prob_map = self.neck(connectivity)
        prob_map_up = self.up(prob_map, shotcut)
        return prob_map_up
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.InstanceNorm2d(hidden_features, affine=True),
            
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.InstanceNorm2d(out_features, affine=True),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class SPGLayer(nn.Module):
    def __init__(self, in_channels, num_knn, dilation, conv='mr', act='gelu', norm='batch',
                 bias=True, stochastic=False, epsilon=0.2, groups=1, drop_path=0.0):
        super(SPGLayer, self).__init__()
        self.self_attention = SelfAttention(in_channels, in_channels)
        self.cross_attention = CrossAttention(in_channels)
        self.gcn = Grapher(in_channels, num_knn, dilation, conv, act, norm,
                           bias, stochastic, epsilon, groups, drop_path)
        self.ffn = FFN(in_channels, in_channels * 4, act=act, drop_path=drop_path)
        self.layer_norm = nn.LayerNorm([in_channels, 64, 64])
        self.N = 512
        self.adapt_pooling = nn.AdaptiveAvgPool1d(self.N)
        self.dropout = nn.Dropout(0.1)
        self.model_init()

    def forward(self, support_graph, query_graph, support_mask):

        # subgraph extraction
        spt_subgraph = self.get_subgraph(support_graph, support_mask) # (1, 512, N)
        if spt_subgraph.shape[2] == 0:
            spt_subgraph = F.pad(spt_subgraph, (0, 1))

        # encoding
        spt_subgraph = self.adapt_pooling(spt_subgraph)
        spt_subgraph = self.self_attention(spt_subgraph)

        # interaction
        query_graph = self.cross_attention(query_graph, spt_subgraph)
        support_graph = self.cross_attention(support_graph, spt_subgraph) 

        # graph forward
        support_GCN = self.dropout(self.ffn(self.gcn(support_graph))) + support_graph
        query_GCN = self.dropout(self.ffn(self.gcn(query_graph))) + query_graph

        support_graph = self.layer_norm(support_GCN)
        query_graph = self.layer_norm(query_GCN)
    
        return support_graph, query_graph

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def get_subgraph(self, graph, mask):

        """
        :param graph: (1, C, H', W')
        :param mask: (1, H, W)
        :return:
        """

        mask = torch.round(mask)
        graph = F.interpolate(graph, size=mask.shape[-2:], mode='bilinear')
        mask = mask.unsqueeze(1).bool()
        result_list = []
        for batch_id in range(graph.shape[0]):
            tmp_tensor = graph[batch_id]  
            tmp_mask = mask[batch_id]  
            subgraph = tmp_tensor[:, tmp_mask[0]]  
            if subgraph.shape[1] == 1:  
                subgraph = torch.cat((subgraph, subgraph), dim=1)

            result_list.append(subgraph)  
        subgraph = torch.stack(result_list)

        return subgraph
           
class FewShotSeg(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Encoder
        self.ResNet = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  # or "ImageNet"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.scaler = 20.0
        self.resize = transforms.Resize((256, 256))
        self.resize_mask =  transforms.Resize((256, 256),
                        interpolation=InterpolationMode.NEAREST)
        self.preprocess = transforms.Compose([
            transforms.CenterCrop(64),
        ])

        self.channels = 512
        self.k = 9
        self.layer_depth = 3
        self.threshold = 0.2
        self.temperature = 0.1

        dpr = [x.item() for x in torch.linspace(0, 0.0, self.layer_depth)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(self.k, 2*self.k, self.layer_depth)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.channels, 64, 64))

        self.N = 512
        self.adapt_pooling = nn.AdaptiveAvgPool1d(self.N)

        self.SPGlayers = Seq(*[SPGLayer(self.channels, num_knn[i], min(i // 4 + 1, max_dilation), drop_path=dpr[i]) 
                               for i in range(self.layer_depth)])
        self.SMD = SubgraphMatching(self.channels, self.channels, self.channels // 2)


    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False):

        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors  (1, 3, 257, 257)
            qry_mask: label
                N x 2 x H x W, tensor
        """
        
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        # Extract features #


        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        imgs_concat = self.resize(imgs_concat)
        img_fts, tao = self.ResNet(imgs_concat) # torch.Size([2, 512, 65, 65])
        img_fts = self.preprocess(img_fts.squeeze(0))
        

        #****************************************************
        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])


        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        outputs_qry = []
        self.contrastive_loss = torch.tensor(0.0).cuda()
        for epi in range(supp_bs):

            """
            supp_fts[[epi], way, shot]: (B, C, H, W) 
            """

            if supp_mask[[0], 0, 0].max() > 0.:

                supp_mask = self.resize_mask(supp_mask[0][0])
                qry_mask = self.resize_mask(qry_mask)

                # ********************************************** graph processing *******************************************************
                support_graph = supp_fts[0][0] + self.pos_embed
                query_graph = qry_fts[0] + self.pos_embed
                B, C, H, W = support_graph.shape
                
                for i in range(self.layer_depth):
                    support_graph, query_graph = self.SPGlayers[i](support_graph, query_graph, supp_mask)

                # ***************************************** subgraph matching decoding ***************************************************
                spt_subgraph = self.get_subgraph(support_graph, supp_mask)# (1, 512, N)
                if spt_subgraph.shape[2] == 0:
                    spt_subgraph = F.pad(spt_subgraph, (0, 1))
                spt_subgraph = self.adapt_pooling(spt_subgraph).permute(2, 0, 1)  # (N, 512, 1)

                prob_map_up = self.SMD(query_graph, spt_subgraph, (H, W)) # (512, 64, 64)


                # ************************************** confusion-minimizing node contrast **********************************************

                if train:
                    uncertainty = self.calculate_confusion_map(prob_map_up.cpu()).squeeze(0)
                    uncertainty = F.interpolate(uncertainty.unsqueeze(0).unsqueeze(0), 
                                                size=(64, 64), mode='bilinear').squeeze(0).squeeze(0)

                    all_hard_indices = self.get_confusion_indices(uncertainty, self.threshold)

                    # avoid too many hard indices
                    if len(all_hard_indices) >= int(64 * 64 * 0.1):
                        random_indices = torch.randperm(len(all_hard_indices))[:int(64 * 64 * 0.1)]
                        all_hard_indices = all_hard_indices[random_indices]

                    fg_indices, bg_indices = self.split_indices_by_mask(all_hard_indices, qry_mask[0].cpu())


                    positive = self.extract_top_features(query_graph, fg_indices).permute(0, 2, 1).squeeze(0) # (1, 512, N)
                    negatives = self.extract_top_features(query_graph, bg_indices).permute(0, 2, 1).squeeze(0)

                    semantic_center = self.get_subgraph(query_graph, qry_mask).mean(dim=2)

                    semantic_center = F.normalize(semantic_center, dim=-1)
                    positive = F.normalize(positive, dim=-1)
                    negatives = F.normalize(negatives, dim=-1)


                    if len(positive) > 0:
                        for pos in positive:
                            self.contrastive_loss  = self.contrastive_loss + \
                                self.node_contrast(semantic_center ,pos.unsqueeze(0), negatives, self.temperature)
                        self.contrastive_loss = self.contrastive_loss / len(positive)
                    else:
                        self.contrastive_loss = self.contrastive_loss + \
                                self.node_contrast(semantic_center ,semantic_center, negatives, self.temperature)
                        self.contrastive_loss = self.contrastive_loss / 1 # for consistency

                    # avoid nan
                    if torch.isnan(self.contrastive_loss):
                        self.contrastive_loss = torch.zeros(1).to(self.device)


                pred_up = F.interpolate(prob_map_up, size=img_size, mode='bilinear', align_corners=True)
                pred = torch.cat((1.0 - pred_up, pred_up), dim=1)
                outputs_qry.append(pred)

            else:
                ########################acquiesce prototypical network ################
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
                ########################################################################

                # Combine predictions of different feature maps #
                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

                preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

                outputs_qry.append(preds)

        output_qry = torch.stack(outputs_qry, dim=1)
        output_qry = output_qry.view(-1, *output_qry.shape[2:])


        return output_qry, 0.01 * self.contrastive_loss
    
    def get_subgraph(self, graph, mask):

        """
        :param graph: (1, C, H', W')
        :param mask: (1, H, W)
        :return:
        """

        mask = torch.round(mask)
        graph = F.interpolate(graph, size=mask.shape[-2:], mode='bilinear')
        mask = mask.unsqueeze(1).bool()
        result_list = []
        for batch_id in range(graph.shape[0]):
            tmp_tensor = graph[batch_id]  
            tmp_mask = mask[batch_id]  
            subgraph = tmp_tensor[:, tmp_mask[0]]  
            if subgraph.shape[1] == 1:  
                subgraph = torch.cat((subgraph, subgraph), dim=1)

            result_list.append(subgraph)  
        subgraph = torch.stack(result_list)

        return subgraph

    def calculate_confusion_map(self, prob_maps):
        """
        :param prob_maps: (C, H, W)
        :return: (H, W)
        """
        entropy_map = -torch.sum(prob_maps * torch.log(torch.clamp(prob_maps, min=1e-10, max=1.0)), dim=0)

        return entropy_map

    def split_indices_by_mask(self, top_indices, binary_mask, scale_factor=4):

        mask_values = binary_mask[scale_factor * top_indices[:, 0], scale_factor * top_indices[:, 1]]
        foreground_indices = top_indices[mask_values == 1]
        background_indices = top_indices[mask_values == 0]

        return foreground_indices, background_indices

    def get_confusion_indices(self, confusion_map, threshold=0.2):
    
        indices = torch.nonzero(confusion_map > threshold)

        return indices

    def extract_top_features(self, features, top_indices):

        n = top_indices.shape[0]
        c = features.shape[1]
        top_features = features[0, :, top_indices[:, 0], top_indices[:, 1]].view(1, c, n)
        return top_features
    
    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def node_contrast(self, anchor, positive_key, negative_keys=None, temperature=0.1, reduction='mean'):
        pos_logit = (anchor * positive_key).sum(dim=1, keepdim=True)  # (|p|,1)
        neg_logits = anchor @ negative_keys.transpose(0, 1)           # (|p|, |n|)
        logits = torch.cat([pos_logit, neg_logits], dim=1)           # (|p|, 1+|n|)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)



  