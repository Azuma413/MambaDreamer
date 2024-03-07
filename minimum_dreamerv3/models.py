import copy
import torch
from torch import nn
from mamba import WrapMamba

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder) # CNNによる画像の特徴量抽出
        self.embed_size = self.encoder.outdim # 特徴量の次元数
        self.dynamics = WrapMamba(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        # self.dynamics = networks.RSSM( # 状態空間モデルのインスタンス化
        #     config.dyn_stoch,
        #     config.dyn_deter,
        #     config.dyn_hidden,
        #     config.dyn_rec_depth,
        #     config.dyn_discrete,
        #     config.act,
        #     config.norm,
        #     config.dyn_mean_act,
        #     config.dyn_std_act,
        #     config.dyn_min_std,
        #     config.unimix_ratio,
        #     config.initial,
        #     config.num_actions,
        #     self.embed_size,
        #     config.device,
        # )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete: # DreamerV3では確率的状態の潜在表現を離散化する
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder( # 潜在空間から画像を生成するデコーダ
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP( # 報酬予測モデル
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP( # DreamerV3で追加されたもの。報酬をtwehot変換するモデル
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer( # モデルの最適化手法の設定 何も指定しなければAdam
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        """
        data: dict
        action (batch_size, batch_length, act_dim) 行動
        image (batch_size, batch_length, h, w, ch) 画像
        reward (batch_size, batch_length) 報酬
        discount (batch_size, batch_length) 割引率？
        
        推定された潜在状態postと，その他の情報を返す
        """

        data = self.preprocess(data) # 画像の前処理と，discountの計算

        with tools.RequiresGrad(self): # パラメータの勾配計算を有効にする
            with torch.cuda.amp.autocast(self._use_amp): # データ型を調整する
                embed = self.encoder(data) # 画像の特徴量を抽出
                # print("embed shape", embed.shape)
                # print("action shape", data["action"].shape)
                # print("is_first shape", data["is_first"].shape)
                # RSSMによる状態空間モデルのlossを計算
                # 出力: 現在の状態から予測された次の潜在状態表現, 次の状態から予測された次の潜在状態表現
                post, prior = self.dynamics.observe(embed, data["action"], data["is_first"]) # 潜在状態の取得 入力: 特徴量，行動，初期状態かどうか
                # print("post shape", post["stoch"].shape)
                # print("prior shape", prior["stoch"].shape)
                kl_free = self._config.kl_free # DreamerV3で実装されたFree Bitsの設定。KLが1以下なら1を返し，それ以外なら元の値を返す。
                dyn_scale = self._config.dyn_scale # 環境変動予測器の出力を符号化器の出力に近づける項の重み
                rep_scale = self._config.rep_scale # 環境変動予測器の出力を符号化器の出力から遠ざける項の重み dyn_scale > rep_scale, dyn_scale + rep_scale = 1
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(post, prior, kl_free, dyn_scale, rep_scale) # KL平衡の損失関数の計算
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape # 画像のバッチサイズと同じ次元数のKL損失が出力されているか確認
                # 画像のデコーダー，報酬予測モデル，報酬のtwo-hot変換モデルのlossを計算
                preds = {} # 予測値を格納する辞書型の変数
                for name, head in self.heads.items(): # headsはdecoder, reward, contのモデルを指す。
                    grad_head = name in self._config.grad_heads # headを学習するかどうか
                    feat = self.dynamics.get_feat(post) # 潜在状態から特徴量を取得
                    feat = feat if grad_head else feat.detach() # grad_headがTrueの場合は特徴量をそのまま使い，Falseの場合は特徴量をdetachする(学習を行わない)
                    pred = head(feat) # 特徴量から画像，報酬，報酬のtwo-hot変換モデルの出力を取得
                    if type(pred) is dict: # 出力が辞書型の場合
                        preds.update(pred) # 辞書型の出力をpredsに追加
                    else:
                        preds[name] = pred # 出力が辞書型でない場合はキーを指定してpredsに追加
                losses = {} # 損失関数を格納する辞書型の変数
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name]) # 予測値と真値の対数尤度を計算。負の対数尤度を損失関数とする。
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss # 損失関数をlossesに追加
                scaled = { # 損失関数に重みをかける
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss # 損失関数の合計にKL損失を加えてモデルの損失関数とする。
            metrics = self._model_opt(torch.mean(model_loss), self.parameters()) # モデルの最適化手法を適用
        # metricsは学習ログに出力するための情報を格納する辞書型の変数
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()}) # 損失関数をmetricsに追加
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp): # データ型を調整する
            metrics["prior_ent"] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy())) # 事前分布(状態空間モデルによって予測された次の潜在状態の分布)のエントロピー
            metrics["post_ent"] = to_np(torch.mean(self.dynamics.get_dist(post).entropy())) # 事後分布(次の状態から計算された次の潜在状態の分布)のエントロピー
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        
        # context = {}
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        """
        観測obsの前処理を行う
        """
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        """
        潜在状態から画像を生成し，予測された画像,実際の観測画像,その差分を結合した配列を返す
        """
        data = self.preprocess(data)
        embed = self.encoder(data) # CNNを用いて画像の特徴量を抽出
        states, _ = self.dynamics.observe(embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]) # postとpriorを取得するが，今回はpriorは使わない
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[:6] # デコーダーモデルを用いて画像を生成
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6] # 報酬予測モデルを用いて報酬を予測
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init) # 行動を考慮して予測された次の潜在状態を取得
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode() # 新たに取得した潜在状態からデコーダーモデルを用いて画像を生成
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode() # 新たに取得した潜在状態から報酬予測モデルを用いて報酬を予測
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1) # 予測された画像を連結(過去の5ステップは潜在状態から予測した画像，6ステップ目は予測された行動から予測された画像)
        truth = data["image"][:6] # 実際の観測画像
        model = model
        error = (model - truth + 1.0) / 2.0 # 予測された画像と実際の観測画像の差分

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False # presicionは精度 精度が16bitの場合はTrueにする
        self._config = config
        self._world_model = world_model # WorldModelのインスタンス
        if config.dyn_discrete: # DreamerV3では確率的状態の潜在表現を離散化する
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else: # 無視
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP( # actorのモデルを作成
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP( # criticのモデルを作成
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]: # criticのslow_targetがTrueの場合 slaw_targetとはCriticの更新頻度を下げて，Acter-Criticの学習安定性を向上させる手法
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer( # actorの最適化手法の設定 何も指定しなければAdam
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer( # criticの最適化手法の設定 何も指定しなければAdam
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA: # 指数移動平均(EMA)を用いた報酬の正規化を行うか否か。Trueの場合はRewardEMAのインスタンスを作成
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(self, start, objective):
        """
        start: 予想を開始する状態の辞書
        objective: feat, state, actionを入力として報酬を出力する関数。実際には報酬予測モデルをラップしたラムダ関数
        """
        self._update_slow_target()
        metrics = {}

        # Actorの損失関数を計算
        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp): # データ型を調整する
                # imag_horizon先までの軌道を予測
                imag_feat, imag_state, imag_action = self._imagine(start, self.actor, self._config.imag_horizon) # imag_horizon先の状態を予測
                reward = objective(imag_feat, imag_state, imag_action) # 予測された報酬を取得
                actor_ent = self.actor(imag_feat).entropy() # actorに状態表現を入力して行動を取得し，そのエントロピーを計算
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy() # 状態のエントロピーを計算
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(imag_feat, imag_state, reward) # Actorの損失関数を計算するためのターゲットを計算
                actor_loss, mets = self._compute_actor_loss(imag_feat, imag_action, target, weights, base) # Actorの損失関数を計算
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None] # SACに対応させるため，エントロピー項を損失関数に加える。
                actor_loss = torch.mean(actor_loss) # 損失関数の平均を取る
                metrics.update(mets) # 損失関数の情報をmetricsに追加
                value_input = imag_feat # criticの入力として状態表現を用いる

        # Criticの損失関数を計算
        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach()) # criticのモデルを用いて状態価値を予測
                target = torch.stack(target, dim=1) # imag_horizon-1までのラムダ報酬の配列をスタック
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach()) # λ報酬の対数尤度を計算して損失関数とする
                slow_target = self._slow_value(value_input[:-1].detach()) # _slow_valueはcriticのslow_targetがTrueの場合に用いるモデル
                if self._config.critic["slow_target"]: # criticのslow_targetがTrueの場合はslow_targetを用いて損失を計算
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None]) # TD(λ)誤差を計算して平均を取る

        # logの追加
        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self): # パラメータの勾配計算を有効にする
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters())) # actorのパラメータを更新
            metrics.update(self._value_opt(value_loss, self.value.parameters())) # criticのパラメータを更新
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        """
        start: 初期状態の辞書
        policy: actorのモデル
        horizon: imag_horizon つまりどれだけ先の状態まで予測するか。
        
        horizon先までの軌道を予測する
        feats: imag_horizon先までの特徴量の配列
        states: imag_horizon先までの状態の辞書(featsはstatesの値を結合しただけ)
        actions: imag_horizon先までの行動の配列
        """
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state) # 確率的な状態の潜在表現であるstochと，決定的な状態の潜在表現であるdeterを結合して状態表現を取得する関数
            inp = feat.detach()
            action = policy(inp).sample() # 状態表現から行動をサンプリング
            # prev_state: 前の状態 確率的な状態の潜在表現であるstochと，決定的な状態の潜在表現であるdeterを持つ辞書
            # prev_action: 前の行動
            # 状態遷移を用いた1ステップ先の未来の状態表現である"prior"を返す
            succ = dynamics.img_step(state, action) # サンプリングした行動と，前の状態から次の状態を予測
            return succ, feat, action

        succ, feats, actions = tools.static_scan(step, [torch.arange(horizon)], (start, None, None)) # horizon分の配列が得られる
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()} # 初期状態startと予測された状態の配列succを連結してstatesに格納

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        """
        Actorの損失関数を計算するためのターゲットを計算
        imag_feat: imag_horizon先までの特徴量の配列
        imag_state: imag_horizon先までの状態の辞書
        reward: imag_horizon先までの報酬の配列
        
        戻り値
        target: imag_horizon-1までのラムダ報酬の配列
        weights: imag_horizon-1までの重みの配列
        value: imag_horizon-1までの状態価値の配列
        """
        if "cont" in self._world_model.heads: # contモデル(つまり報酬のtwo-hot変換モデル)が存在する場合
            inp = self._world_model.dynamics.get_feat(imag_state) # 状態表現を取得
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean # ラムダ報酬の割引率を取得
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode() # criticのモデルを用いて状態価値を予測
        target = tools.lambda_return( # TD(λ)学習のためのラムダ報酬を計算
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(self, imag_feat, imag_action, target, weights, base):
        """
        Actorの損失関数を計算
        """
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA: # 指数移動平均(EMA)を用いた報酬の正規化を行う場合
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        """
        slow_targetがTrueの場合，更新回数がslow_target_updateの倍数の場合にslow_target_fractionの割合でslow_targetを更新する
        """
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
