
import torch
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
	def __init__(self, in_channels, patch_size, emb_size, img_size):#in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224
		super().__init__()
		self.patch_size = patch_size
		self.nPatches = (img_size*img_size)//((patch_size)**2)
		self.projection = nn.Sequential(
			Rearrange('b c (h p1)(w p2) -> b (h w) (p1 p2 c)',p1 = patch_size,p2 = patch_size),
			nn.Linear(patch_size * patch_size * in_channels, emb_size)
		)
		self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
		#self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1,emb_size))
                
	def forward(self, x: Tensor):
		b,c,h,w = x.shape
		x = self.projection(x)
		cls_tokens = repeat(self.cls_token,'() n e -> b n e', b=b)#repeat the cls tokens for all patch set in 
		x = torch.cat([cls_tokens,x],dim=1)
		#x+=self.positions
		return x

class multiHeadAttention(nn.Module):
	def __init__(self, emb_size, heads, dropout):
		super().__init__()
		self.heads = heads
		self.emb_size = emb_size
		self.query = nn.Linear(emb_size,emb_size)
		self.key = nn.Linear(emb_size,emb_size)
		self.value = nn.Linear(emb_size,emb_size)
		self.drop_out = nn.Dropout(dropout)
		self.projection = nn.Linear(emb_size,emb_size)

	def forward(self,x):
		#splitting the single input int number of heads
		queries = rearrange(self.query(x),"b n (h d) -> b h n d", h = self.heads)
		keys = rearrange(self.key(x),"b n (h d) -> b h n d", h = self.heads)
		values = rearrange(self.value(x),"b n (h d) -> b h n d", h = self.heads)
		attention_maps = torch.einsum("bhqd, bhkd -> bhqk",queries,keys)
		scaling_value = self.emb_size**(1/2)
		attention_maps = F.softmax(attention_maps,dim=-1)/scaling_value
		attention_maps = self.drop_out(attention_maps)##might be deleted
		output = torch.einsum("bhal, bhlv -> bhav",attention_maps,values)
		output  = rearrange(output,"b h n d -> b n (h d)")
		output = self.projection(output)
		return output
class residual(nn.Module):
	def __init__(self,fn):
		super().__init__()
		self.fn = fn
	def forward(self,x):
		identity = x
		res = self.fn(x)
		out = res + identity
		return out


class DeepBlock(nn.Sequential):
	def __init__(self,emb_size:int =256 ,drop_out:float=0.0):#64
		super().__init__(
        		residual(
            			nn.Sequential(
                			nn.LayerNorm(emb_size),
                			multiHeadAttention(emb_size,2,drop_out),
                			nn.LayerNorm(emb_size)
            			)
        		)
    		)

class Classification(nn.Sequential):
	def __init__(self, emb_size:int=256, n_classes:int=2):
		super().__init__(
			Reduce('b n e -> b e', reduction='mean'),
			nn.LayerNorm(emb_size), 
			nn.Linear(emb_size, n_classes))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
  def __init__(self,emb_size,drop_out, n_classes,in_channels,patch_size1,patch_size2,image_size):
    super().__init__()
    self.PatchEmbedding1 = PatchEmbedding(in_channels,patch_size1,emb_size,image_size)
    self.PatchEmbedding2 = PatchEmbedding(in_channels,patch_size2,emb_size,image_size)
    self.DeepBlock = DeepBlock()#Transformer()
    self.Classification = Classification(n_classes=2)
  def forward(self,x):
    patch1 = self.PatchEmbedding1(x)
    patch2 = self.PatchEmbedding2(x)
     # Resize tensor2 along the second dimension to match the size of tensor1
    desired_size = patch1.shape[1]  #patchEmbeddings1
    indices = torch.linspace(0, patch2.shape[1] - 1, desired_size, device=device).long()
    patch2_resized = torch.index_select(patch2, 1, indices)#.to(device)

    # Concatenate the tensors along the second dimension (dim=1)
    patchEmbeddings = torch.cat((patch2_resized, patch1), dim=1)#.to(device)

    DeepBlockOp = self.DeepBlock(patchEmbeddings)
    classificationOutput = self.Classification(DeepBlockOp)
    output = F.log_softmax(classificationOutput, dim=1)
    return output
    
  