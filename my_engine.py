import datetime
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
import wandb
from pydantic import BaseModel, ConfigDict
from tqdm import trange

from my_tasks import get_task_sampler
from nano_gpt import GPT2Config, GPT2Model


class CurriculumSchedule(BaseModel):
    start: int
    end: int
    inc: int
    interval: int


class CurriculumConfig(BaseModel):
    dims: CurriculumSchedule = CurriculumSchedule(start=5, end=20, inc=1, interval=5000)

    points: CurriculumSchedule = CurriculumSchedule(start=11, end=41, inc=2, interval=5000)

    loops: CurriculumSchedule = CurriculumSchedule(start=15, end=30, inc=2, interval=500)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    debug_mode: bool = True

    # Outputs and Wandb logging
    project: str = "alex_loop"
    notes: str = ""
    name: str = "noname_run"
    log_every_steps: int = 100

    timestamp: str = datetime.datetime.now().strftime("%m%d%H%M%S")
    run_id: str = f"{timestamp}-{name}-{str(uuid.uuid4())[:4]}"

    out_dir: Path = Path("./outputs") / run_id

    seed: int = 42

    # Net
    family: str = "gpt2_loop"
    # family: str = "gpt2"

    n_embd: int = 256
    n_layer: int = 1
    n_head: int = 8
    n_positions: int = 101
    n_dims: int = 20

    pred_type: str = "regression"  # conf.pred_type
    loop_func: str = "z=f(x+z)"  # conf.loop_func

    # Training: optimizers and scalers
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate: float = 0.0001  # args.training.learning_rate
    weight_decay: float = 0.0  # args.training.weight_decay

    epochs: int = 500000  # args.training.train_steps
    batch_size: int = 64  # training.batch_size
    sparsity: int = 100  # training.sparsity

    n_loop_window: int = 15

    task_name: str = "linear_regression"  # training.task_name

    curriculum: CurriculumConfig = CurriculumConfig()


class Curriculum:
    def __init__(self, config: CurriculumConfig):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter

        self.n_dims_truncated = config.dims.start
        self.n_points = config.points.start
        self.n_loops = config.loops.start

        self.n_dims_schedule = config.dims
        self.n_points_schedule = config.points
        self.n_loops_schedule = config.loops
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(self.n_dims_truncated, self.n_dims_schedule)
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)
        self.n_loops = self.update_var(self.n_loops, self.n_loops_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


class TransformerModel(torch.nn.Module):
    MAX_NUM_CLASS = 2  # for openML classification task

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config

        self.freq = 2
        self.ind = 0

        block_size = self.freq * config.n_positions + 1

        self.configuration = GPT2Config(
            block_size=block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )

        self.n_positions = config.n_positions  # n = points in this setting
        self.n_dims = config.n_dims  # input dimension, d_in
        self.n_embd = config.n_embd  # d
        self.n_layer = config.n_layer
        self._pred_type = config.pred_type

        self._read_in = torch.nn.Linear(config.n_dims, config.n_embd)
        self._backbone = GPT2Model(self.configuration)
        if self._pred_type == "regression":
            self._read_out = torch.nn.Linear(config.n_embd, 1)
        elif self._pred_type == "classification":
            self._read_out = torch.nn.Linear(config.n_embd, self.MAX_NUM_CLASS)  # NOTE: hard-code

        self.print_flag = False

    def _combine(self, xs_b, ys_b):
        """
        :param xs_b: shape [B, n, d_in]
        :param ys_b: shape [B, n]
        :return: shape [B, 2n, d_in + 1]
        """
        B, n, d = xs_b.shape
        device = xs_b.device

        ys_b_wide = torch.cat((ys_b.view(B, n, 1), torch.zeros(B, n, d - 1, device=device)), axis=2)

        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(B, self.freq * n, d)

        return zs

    def forward(self, xs, ys):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :return:
        """

        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]

        f_output = self._backbone(inputs_embeds=embeds, position_ids=None, rm_pos_embd=False)  # [B, 2n, d]
        prediction = self._read_out(f_output)  # [B, 2n, d] -> [B, 2n, 1]
        if self._pred_type == "regression":
            y = prediction[:, self.ind :: self.freq, 0]
        elif self._pred_type == "classification":
            y = prediction[:, self.ind :: self.freq]
        else:
            raise NotImplementedError

        return y


class TransformerModelLooped(TransformerModel):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.loop_func = config.loop_func

    def f(self, output, embeds):
        if self.loop_func == "z=f(x+z)":
            f_output = self._backbone(inputs_embeds=output + embeds)  # [B, 2n + 1, d]
        elif self.loop_func == "z=f(x*z)":
            f_output = self._backbone(inputs_embeds=output * embeds)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def forward(self, xs, ys, n_loop_start, n_loops):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        if self.loop_func in ["z=f(x+z)"]:
            output = torch.zeros_like(embeds)  # also of shape [B, 2n, d]
        elif self.loop_func in ["z=f(x*z)"]:
            output = torch.ones_like(embeds)  # also of shape [B, 2n, d]
        else:
            raise NotImplementedError("Currently we only support loop function z=f(x+z) or z=f(x*z).")

        pred_list = []
        for idx in range(n_loops):
            if idx < n_loop_start:  # this will save memory when n_loops large.
                with torch.no_grad():
                    output = self.f(output, embeds)
            else:
                output = self.f(output, embeds)
                prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
                if self._pred_type == "regression":
                    y = prediction[:, self.ind :: self.freq, 0]
                elif self._pred_type == "classification":
                    y = prediction[:, self.ind :: self.freq]
                else:
                    raise NotImplementedError
                pred_list.append(y)
            if not self.print_flag:
                print(idx)
                self.print_flag = True

        return pred_list


def create_model(config: ExperimentConfig):
    if config.family == "gpt2":
        model = TransformerModel(config)
    elif config.family == "gpt2_loop":
        model = TransformerModelLooped(config)
    else:
        raise NotImplementedError

    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    return model, optimizer


def train_batch(
    xs: torch.Tensor,
    ys: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum: Curriculum,
    config: ExperimentConfig,
):
    xs, ys = xs.to(config.device), ys.to(config.device)

    if config.family == "gpt2":
        y_pred = model(xs, ys)  # [B, n]
        # list of [B, n], length K + 1, get rid of the 0-th one
        loss = (ys - y_pred).square().mean()  # auto on both K and n (number of in context samples)
    elif config.family == "gpt2_loop":
        n_loops = curriculum.n_loops  # K

        horizon_start = max(0, n_loops - config.n_loop_window)
        y_pred_list = model(xs, ys, horizon_start, n_loops)
        # list of [B, n], length K
        y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
        y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)  # [B * K, n]
        loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
        y_pred = y_pred_list[-1]  # [B, n]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()

    return loss.detach(), y_pred.detach()


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum: Curriculum,
    config: ExperimentConfig,
):
    # wandb.watch(model, loss_func, log="all", log_freq=100)

    for epoch in trange(config.epochs):
        model.train()

        task_sampler = get_task_sampler(
            task_name=config.task_name,
            batch_size=config.batch_size,
            n_points=curriculum.n_points,
            n_dims=config.n_dims,
            n_dims_truncated=curriculum.n_dims_truncated,
            device=config.device,
            sparsity=config.sparsity,
        )

        real_task = task_sampler()
        xs, ys = real_task.xs.float(), real_task.ys.float()

        loss, output = train_batch(xs, ys, model, optimizer, curriculum, config)

        # test(train_loader, val_loader, model, accuracy_calculator, epoch, config)

        point_wise_tags = list(range(curriculum.n_points))  # [0, 1, 2, ..., n-1]
        if epoch % config.log_every_steps == 0:
            point_wise_loss = (output - ys).square().mean(dim=0)  # [n,]

            wandb.log(
                {
                    "overall_loss": loss,
                    "loop_times": curriculum.n_loops,
                    "pointwise/loss": dict(zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

        curriculum.update()

    model_path = config.out_dir / f"model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)


def setup_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    config = ExperimentConfig()

    config.out_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        dir=config.out_dir,
        project=config.project,
        config=config.model_dump(),
        notes=config.notes,
        name=config.name,
        mode="disabled" if config.debug_mode else "online",
        resume=True,
    )

    setup_seed(config.seed)

    # torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    curriculum = Curriculum(config.curriculum)

    model, optimizer = create_model(config)

    train(model, optimizer, curriculum, config)

    wandb.finish()
