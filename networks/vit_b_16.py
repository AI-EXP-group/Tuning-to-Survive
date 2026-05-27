import torch
import torch.nn as nn
import torch.nn.functional as F

class Vit(nn.Module):
    def __init__(self, classes:int, blocks:int, channels:int, height: int, width:int, patch_size:int, H:int, inner_dim:int, dropout:float) -> None:
        super().__init__()
        self.embeddings = Embeddings(channels, height, width, patch_size)
        self.encoder = nn.Sequential(
            *[Encoder(self.embeddings.D, H, inner_dim, dropout) for _ in range(blocks)]
        )
        self.pooler = nn.Linear(self.embeddings.D, self.embeddings.D)
        self.layernorm = nn.LayerNorm(normalized_shape=self.embeddings.D)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embeddings.D),
            nn.Linear(self.embeddings.D, classes)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(x)

        encoder = self.encoder(embeddings)
        # print(encoder)
        layernorm = self.layernorm(encoder)
        
        pooler = self.pooler(layernorm)

        cls_token = pooler[: , 0]

        mlp_head = self.mlp_head(cls_token)
        
        return mlp_head



class MultiHeadAttention(nn.Module):
    def __init__(self, D:int, H:int, dropout:float ) -> None:
        super().__init__()
        self.D = D
        self.H = H
        self.dropout = dropout
        assert self.D % self.H == 0, f"features {D} not divisible by heads {self.H}"
        self.d_k = D // H
        self.query = nn.Linear(self.D,self.D)
        self.key = nn.Linear(self.D,self.D)
        self.value = nn.Linear(self.D,self.D)
        self.output = nn.Linear(self.D, self.D)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch, N, _ = x.size() #[2, 145, 108]
        q = self.query(x) #[2, 145, 108]
        q = q.view(batch, self.H, N, self.d_k) #[2, 6, 145, 18]
        # print(q[0][0][0][0])
        k = self.key(x)
        k = k.view(batch, self.H, N, self.d_k)
        # print(k[0][0][0][0])
        v = self.value(x)
        v = v.view(batch, self.H, N, self.d_k)
        # print(v[0][0][0][0])
        dots = (q @ k.transpose(2,3)) / (self.d_k ** 0.5) #[2, 6, 145, 145]
        # print(dots[0][0][0][0])
        attn = F.softmax(dots, dim=3) #[2, 6, 145, 145]
        # print(attn[0][0][0][0])
        out = attn @ v #[2, 6, 145, 18]
        out = out.transpose(1, 2).reshape(batch, N, self.D) #[2, 145, 108]
        # print(out[0][0][0])
        out = self.output(out)
        # print(out[0][0][0])
        return self.dropout_layer(out) #[2, 145, 108]
    
    
    
class Encoder(nn.Module):
    def __init__(self, D:int, H:int, inner_dim:int, dropout:float) -> None:
        super().__init__()
        self.layernorm_before = nn.LayerNorm(normalized_shape=D)
        self.msa = MultiHeadAttention(D, H, dropout)
        self.layernorm_after = nn.LayerNorm(normalized_shape=D)
        self.intermediate = nn.Linear(D, inner_dim)
        self.gelu = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.output = nn.Linear(inner_dim, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x[0][0][0])
        layernorm_before = self.layernorm_before(x)
        # print(layernorm_before[0][0][0])
        msa = self.msa(layernorm_before)
        # print(layernorm_before[0][0][0])
        residual_1 = msa + x
        layernorm_after = self.layernorm_after(residual_1)
        intermediate = self.intermediate(layernorm_after)
        gelu = self.gelu(intermediate)
        output = self.output(gelu)
        residual_2 = output + residual_1

        return residual_2
    
    
class Embeddings(nn.Module):
    def __init__(self, channels:int, height: int, width:int, patch_size:int) -> None:
        super().__init__()
        assert height % patch_size == 0, "height is not divible by patch_size"
        self.N = (height * width) // (patch_size ** 2)
        self.D = (patch_size ** 2) * channels
        self.projection = nn.Conv2d(channels, self.D, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.D))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.N + 1, self.D))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x->[2, 3, 72, 72]
        out = self.projection(x) #[2, 108, 12, 12]
        out = out.flatten(2) #[2, 108, 144]
        out = out.transpose(1,2) #[2, 144, 108]
        #repeat() function is used to add cls_tonken to all images in the batch
        out = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), out], dim=1) #[2, 145, 108]
        out = out + self.position_embeddings #[2, 145, 108]
        return out #[2, 145, 108]
    
    
# model = Vit(classes=200, blocks=12, channels=3, height=224, width=224, 
#             patch_size=16,
#             H=12, inner_dim=3072, dropout=0.1)
# model.eval()

# tensor = torch.rand([4, 3, 224, 224])

# print(model)