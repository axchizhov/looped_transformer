import os
import random

import numpy as np
import torch

from pydantic import BaseModel, ConfigDict

# torch.backends.cudnn.benchmark = True
import datetime
from tqdm import trange

import wandb
import datetime
import uuid

from my_utils import TransformerModel, TransformerModelLooped, Curriculum
from my_tasks import get_task_sampler


from pathlib import Path


class GenericSchedule(BaseModel):
    start: int
    end: int
    inc: int
    interval: int


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dims: GenericSchedule = GenericSchedule(start=5, end=20, inc=1, interval=5000)

    points: GenericSchedule = GenericSchedule(start=11, end=41, inc=2, interval=5000)

    loops: GenericSchedule = GenericSchedule(start=20, end=350, inc=2, interval=500)


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


def setup_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_model(config: ExperimentConfig):
    if config.family == "gpt2":
        model = TransformerModel(**dict(config))
    elif config.family == "gpt2_loop":
        model = TransformerModelLooped(**dict(config))
    else:
        raise NotImplementedError

    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    return model, optimizer


def train_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum,
    config: ExperimentConfig,
):
    X, y = X.to(config.device), y.to(config.device)

    if config.family == "gpt2":
        y_pred = model(X, y, add_inputs_embeds=False)  # [B, n]
        # list of [B, n], length K + 1, get rid of the 0-th one
        loss = (y - y_pred).square().mean()  # auto on both K and n (number of in context samples)
    elif config.family == "gpt2_loop":
        n_loops = curriculum.n_loops  # K

        horizon_start = max(0, n_loops - config.n_loop_window)
        y_pred_list = model(X, y, horizon_start, n_loops)
        # list of [B, n], length K
        y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
        y_star_arr = torch.cat([y] * len(y_pred_list), dim=0)  # [B * K, n]
        loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
        y_pred = y_pred_list[-1]  # [B, n]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()

    return loss.detach(), y_pred.detach()


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum,
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    model, optimizer = create_model(config)

    curriculum = Curriculum(config.curriculum)

    train(
        model,
        optimizer,
        curriculum,
        config,
    )

    wandb.finish()
