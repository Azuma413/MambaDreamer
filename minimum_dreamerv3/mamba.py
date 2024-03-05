"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

import tools
from torch import distributions as torchd

# できるだけ元の枠組みを維持して実装する。
# 例えばstateはstochとdeterの辞書型であるが，今回の実装ではimg_obsのみを扱うため，stochとdeterにimg_obsを入れる。
class WrapMamba(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super().__init__()
        act = getattr(torch.nn, act)
        self._device = device
        self._num_actions = num_actions
        self._discrete = discrete
        self._initial = initial
        self._unimix_ratio = unimix_ratio
        
        # RSSMの代用としてのMamba
        self.args = ModelArgs()
        self.args.d_model = 384
        self.args.n_layer = 4
        self.args.vocab_size = stoch*discrete + deter + num_actions
        self.args.feat_size = stoch*discrete + deter
        self.mamba = Mamba(self.args)
        self.mamba.apply(tools.weight_init)
        
        # Mambaの出力とembedを結合してfeat_sizeの出力を得るFC層
        obs_out_layers = []
        inp_dim = self.args.feat_size + embed
        obs_out_layers.append(nn.Linear(inp_dim, self.args.feat_size, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self.args.feat_size, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)
        
    def initial(self, batch_size):
        """
        初期状態を設定する
        """
        deter = torch.zeros(batch_size, self.args.feat_size).to(self._device)
        state = dict(
            logit=deter,
            stoch=deter,
            deter=deter,
        )

        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            print("learned initial state には対応していない")
            return state
        else:
            raise NotImplementedError(self._initial)
    
    def observe(self, embed, action, is_first, state=None):
        """
        embed: CNNの出力の画像の特徴量
        action: 行動
        is_first: エピソードの最初のステップかどうか
        posteriorとpriorの配列を返す(確率状態の潜在表現stochと決定状態の潜在表現であるdeterの辞書)
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape)))) # ラムダ関数の定義 swap: (batch, time, ch) -> (time, batch, ch) 次元入れ替え
        embed, action, is_first = swap(embed), swap(action), swap(is_first) # embed, action, is_firstの次元入れ替え
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        # post: 1ステップ先の観測の情報を取り込んで計算した状態表現である"posterior"。つまり教師データ。 prior: RSSMによって出力された1ステップ先の未来の状態表現である"prior"
        post, prior = tools.static_scan( # 全ての時間ステップに対してops_step関数を適用。戻り値は第一引数の関数の出力であるposteriorとpriorのリスト
            lambda prev_state, prev_act, embed, is_first: self.obs_step(prev_state[0], prev_act, embed, is_first), # obs_step関数により状態を更新
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()} # postの次元を元に戻す
        prior = {k: swap(v) for k, v in prior.items()} # priorの次元を元に戻す
        return post, prior # posteriorとpriorの配列を返す
    
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """
        prev_state: 前の状態
        prev_action: 前の行動
        embed: CNNの出力の画像の特徴量
        is_first: エピソードの最初のステップかどうか
        1ステップ先の観測の情報を取り込んで計算した状態表現である"posterior"と，
        RSSMによって出力された1ステップ先の未来の状態表現である"prior"を返す
        どちらも確率的状態の潜在表現であるstochと，決定的な状態の潜在表現であるdeterを持つ辞書型であるが，本来はdeterは必要ない
        """
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first): # prev_stateがNoneまたはis_firstが全てTrueの場合
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions)).to(self._device)
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (val * (1.0 - is_first_r) + init_state[key] * is_first_r)

        prior = self.img_step(prev_state, prev_action) # 1ステップ先の状態表現を得る
        x = torch.cat([prior["deter"], embed], -1) # rnn_hidden(deter)を1ステップ先のCNNの出力の画像の特徴量と結合
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        next_img_obs = self._obs_out_layers(x) # xを全結合層に通してfeat_sizeの出力を得る
        post = {"stoch": next_img_obs, "deter": next_img_obs, "logit": next_img_obs} # 1ステップ先の観測の情報を取り込んで計算した状態表現である"posterior"を得る
        return post, prior # posteriorとpriorを返す
    
    def img_step(self, prev_state, prev_action, sample=True):
        """
        prev_state: 前の状態 確率的な状態の潜在表現であるstochと，決定的な状態の潜在表現であるdeterを持つ辞書
        prev_action: 前の行動
        状態遷移を用いた1ステップ先の未来の状態表現である"prior"を返す
        """
        prev_img_obs = prev_state["stoch"]
        x = torch.cat([prev_img_obs, prev_action], -1) # 前の潜在表現と前の行動を結合
        stats = self.mamba(x) # Mambaにより1ステップ先の未来の状態表現を得る
        stats = nn.LayerNorm(self.args.feat_size, eps=1e-03)(stats) # statsを正規化
        prior = {"stoch": stats, "deter": stats, "logit": stats} # 状態遷移を用いた1ステップ先の未来の状態表現である"prior"を得る
        return prior
    
    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior
    
    def get_feat(self, state):
        """
        stateから状態空間の潜在表現を取得する
        """
        return state["stoch"]
    
    def get_dist(self, state, dtype=None):
        return nn.Softmax(dim=-1)(state["stoch"])
    
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """
        kl平衡を計算する
        論文ではdyn_scaleとrep_scaleの和は1になるようにしている
        """
        kld = torchd.kl.kl_divergence
        # softmaxで確率分布に変換
        post_value = nn.Softmax(dim=-1)(post["stoch"])
        prior_value = nn.Softmax(dim=-1)(prior["stoch"])
        rep_loss = value = kld(
            post_value,
            prior_value.detach(),
        )
        dyn_loss = kld(
            post_value.detach(),
            prior_value,
        )
        rep_loss = torch.clip(rep_loss, min=free) # rep_lossの値がfree以下の場合はfreeにする free bitsの実装。論文ではfree=1
        dyn_loss = torch.clip(dyn_loss, min=free) # dyn_lossの値がfree以下の場合はfreeにする
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss

@dataclass
class ModelArgs:
    d_model: int # 最初のLinear層の出力サイズ, 残差ブロックへの入力サイズ
    n_layer: int # 残差ブロック層の数
    vocab_size: int # action + img_obs 最初のLinear層の入力サイズ
    feat_size: int # 最後のFC層の出力サイズ
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4  # mambaブロックの1次元畳み込み層のカーネルサイズ
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True # mambaブロックの1次元畳み込み層のバイアス on/off
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model) # mambaブロックの1次元畳み込み層の入出力サイズ
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        # 単語の埋め込み層 vocab_size: 単語の数, d_model: 埋め込みベクトルの次元数 vocab_size -> d_model
        # self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.fc_layer = nn.Linear(args.vocab_size, args.d_model) # vocab_sizeはaction + img_obs, d_modelは隠れ層の次元数 vmambaでは96, 192, 384, 768とか。この程度のサイズならok 
        # n_layer分の残差ブロック層
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        # 正規化層 d_model -> d_model
        self.norm_f = RMSNorm(args.d_model)
        # 言語モデルの出力層 d_model -> vocab_size
        self.lm_head = nn.Linear(args.d_model, args.feat_size, bias=False) # d_modelからimg_obsに変換する
        # self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights. これ必要なの？
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        # x = self.embedding(input_ids)
        x = self.fc_layer(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    
    # @staticmethod
    # def from_pretrained(pretrained_model_name: str):
    #     """Load pretrained weights from HuggingFace into model.
    
    #     Args:
    #         pretrained_model_name: One of
    #             * 'state-spaces/mamba-2.8b-slimpj'
    #             * 'state-spaces/mamba-2.8b'
    #             * 'state-spaces/mamba-1.4b'
    #             * 'state-spaces/mamba-790m'
    #             * 'state-spaces/mamba-370m'
    #             * 'state-spaces/mamba-130m'
                            
    #     Returns:
    #         model: Mamba model with weights loaded
    
    #     """
    #     from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    #     from transformers.utils.hub import cached_file
        
    #     def load_config_hf(model_name):
    #         resolved_archive_file = cached_file(model_name, CONFIG_NAME,
    #                                             _raise_exceptions_for_missing_entries=False)
    #         return json.load(open(resolved_archive_file))
        
        
    #     def load_state_dict_hf(model_name, device=None, dtype=None):
    #         resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
    #                                             _raise_exceptions_for_missing_entries=False)
    #         return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
    #     config_data = load_config_hf(pretrained_model_name)
    #     args = ModelArgs(
    #         d_model=config_data['d_model'],
    #         n_layer=config_data['n_layer'],
    #         vocab_size=config_data['vocab_size']
    #     )
    #     model = Mamba(args)
        
    #     state_dict = load_state_dict_hf(pretrained_model_name)
    #     new_state_dict = {}
    #     for key in state_dict:
    #         new_key = key.replace('backbone.', '')
    #         new_state_dict[new_key] = state_dict[key]
    #     model.load_state_dict(new_state_dict)
        
    #     return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias) # d_model -> d_inner*2 biasはFalse

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape # b: batch size, l: sequence length, d: hidden dim
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_inner)に変換
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1) # x_and_resを２分割してxとresにする

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l] # 1次元畳み込み層を通す (b, d_in, l) -> (b, d_in, l)
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x) # Swish activation function

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
        
