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

class TrainConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Outputs and Wandb logging
    project = args.wandb.project
    notes = args.wandb.notes
    name = args.wandb.name
    
    timestamp: str = datetime.datetime.now().strftime('%m%d%H%M%S')
    run_id: str = f"{timestamp}-{name}-{str(uuid.uuid4())[:4]}"
    
    out_dir: Path = Path("outputs") / run_id
    
    seed = 42
    # debug_mode: bool = args.debug_mode
    
    # Net
    model_family: str = "gpt2_loop"
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    n_dims = conf.n_dims
    n_positions = conf.n_positions
    n_embd = conf.n_embd
    n_layer = conf.n_layer
    n_head = conf.n_head
    pred_type = conf.pred_type
    
    loop_func = conf.loop_func
    
    # Training: optimizers and scalers
    
    learning_rate: float = args.training.learning_rate
    weight_decay: float = args.training.weight_decay
    
    
    epochs = args.training.train_steps # 500 000
    
    
    
    
    
    
    
    
    

    # Net
    architecture: str = "vit_b_16 @ IMAGENET1K_V1"
    learning_rate: float = 0.0001
    epochs: int = 100

    # Dataset
    root_dir: str = "data/reference_and_train_set"
    annotation_csv: str = "labels.csv"
    dataset_name: str = Path(root_dir).name
    image_size: tuple = (192, 128)

    # Dataloader
    batch_size: int = 64
    num_workers: int = 8

    # Metrics
    k_neighbors: int = 1
    metrics: tuple = ("precision_at_1", "mean_average_precision")

    # Other
    
    model_dir: str = "/workspaces/classifier/output/january_filtered_set"
    weights_name: str = "model_refset_january"


def setup_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# def create_dataloaders(config: TrainConfig):
#     rotations = [lambda image: T.functional.rotate(image, angle) for angle in [0, 90, 180, 270]]
#     train_transform = torchvision.transforms.Compose(
#         [
#             torchvision.transforms.RandomChoice(rotations),
#             torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
#         ]
#     )

#     val_transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
#     test_transform = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()

#     train_set = ProductClassificationDataset(config, split="train", transform=train_transform)
#     val_set = ProductClassificationDataset(config, split="val", transform=val_transform)
#     test_set = ProductClassificationDataset(config, split="test", transform=test_transform)

#     train_loader = DataLoader(
#         train_set,
#         batch_size=config.batch_size,
#         num_workers=config.num_workers,
#         # pin_memory=True,
#     )

#     val_loader = DataLoader(
#         val_set,
#         batch_size=config.batch_size,
#         num_workers=config.num_workers,
#         # pin_memory=True,
#     )

#     test_loader = DataLoader(
#         test_set,
#         batch_size=config.batch_size,
#         num_workers=config.num_workers,
#         # pin_memory=True,
#     )

#     return train_loader, val_loader, test_loader

def create_model(config: TrainConfig):
    if config.model_family == "gpt2":
         model = TransformerModel(**dict(config))
    elif config.model_family == "gpt2_loop":
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
    # loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    curriculum,
    config: TrainConfig,
):
    X, y = X.to(config.device), y.to(config.device)
    
    if config.model_family == "gpt2":
            y_pred = model(X, y, add_inputs_embeds=args.training.add_inputs_embeds)  # [B, n]
            # list of [B, n], length K + 1, get rid of the 0-th one
            loss = (y - y_pred).square().mean()  # auto on both K and n (number of in context samples)
    elif config.model_family == "gpt2_loop":
        n_loops = curriculum.n_loops  # K
        
        horizon_start = max(0, n_loops - args.training.n_loop_window)
        y_pred_list = model(X, y, horizon_start, n_loops)
        # list of [B, n], length K
        y_pred_arr = torch.cat(y_pred_list, dim=0)  # [B * K, n]
        y_star_arr = torch.cat([y] * len(y_pred_list), dim=0)  # [B * K, n]
        loss = (y_star_arr - y_pred_arr).square().mean()  # auto on both K and n (number of in context samples)
        y_pred = y_pred_list[-1]  # [B, n]

    # loss = loss_func(embeddings, y, hard_pairs)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()

    return loss.detach(), y_pred.detach()


def log_training(loss: torch.Tensor, num_triplets, example_count, epoch):
    loss = float(loss)

    wandb.log({"epoch": epoch, "loss": loss}, step=example_count)
    # print(f"Loss after {str(example_count).zfill(5)} examples: {loss:.3f}")
    # print(f"Number of mined triplets = {num_triplets}")


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


# def test(
#     curriculum,
#     epoch,
    
    
    
#     train_loader: DataLoader,
#     test_loader: DataLoader,
#     model: nn.Module,
#     accuracy_calculator,
#     epoch: int,
#     config: TrainConfig,
# ):

    
    
    
    
    
    
#     train_set = train_loader.dataset
#     test_set = test_loader.dataset

#     tester = testers.GlobalEmbeddingSpaceTester(
#         dataloader_num_workers=config.num_workers,
#         accuracy_calculator=accuracy_calculator,
#     )

#     dataset_dict = {"validation_set": test_set, "train_set": train_set}
#     accuracies = tester.test(dataset_dict, epoch, model)
#     print(accuracies)

#     # train_embeddings, train_labels = get_all_embeddings(train_set, model)
#     # test_embeddings, test_labels = get_all_embeddings(test_set, model)
#     # train_labels = train_labels.squeeze(1)
#     # test_labels = test_labels.squeeze(1)

#     # accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels, train_embeddings, train_labels, False)

#     wandb.log(accuracies)


def train(
    model: torch.nn.Module,
    # loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    # scaler,
    # train_loader: DataLoader,
    # val_loader: DataLoader,
    # accuracy_calculator,
    curriculum,
    config: TrainConfig,
):
    # wandb.watch(model, loss_func, log="all", log_freq=100)

    example_count = 0
    batch_count = 0
    for epoch in trange(config.epochs):
        model.train()
        
        task_sampler = get_task_sampler(
                task_name=args.training.task_name,
                batch_size=args.training.batch_size,
                n_points=curriculum.n_points,
                n_dims=args.model.n_dims,
                n_dims_truncated=curriculum.n_dims_truncated,
                device=config.device,
                sparsity=args.training.sparsity,
            )
        
        real_task = task_sampler()
        xs, ys = real_task.xs.float(), real_task.ys.float()
        
        loss, output = train_batch(xs, ys, model, optimizer, curriculum, config)
        # example_count += len(X)
        # batch_count += 1
        # if batch_count % 100 == 0:
        #     log_training(loss, miner.num_triplets, example_count, epoch)
    
        # test(train_loader, val_loader, model, accuracy_calculator, epoch, config)
        point_wise_tags = list(range(curriculum.n_points))  # [0, 1, 2, ..., n-1]
        if epoch % args.wandb.log_every_steps == 0:
            point_wise_loss = (output - ys).square().mean(dim=0)  # [n,]
            
            wandb.log(
                    {
                        "overall_loss": loss,
                        "loop_times": curriculum.n_loops,
                        "pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.detach().cpu().numpy())
                        ),
                        "n_points": curriculum.n_points,
                        "n_dims": curriculum.n_dims_truncated,
                        "lr": optimizer.param_groups[0]['lr'],
                    },
                    step=epoch,
                )
        
        curriculum.update()
        
        # if epoch % 4 == 0:
        #     model_path = Path(config.model_dir) / f"{config.weights_name}_e{epoch}.pth"
        #     torch.save(model.state_dict(), model_path)
        
        # if i % args.training.save_every_steps == 0:
        #     training_state = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "train_step": i,
        #     }
        #     torch.save(training_state, state_path)
        # if (
        #         args.training.keep_every_steps > 0
        #         and i % args.training.keep_every_steps == 0
        #         and i > 0
        # ) or (i == args.training.train_steps - 1):
        #     torch.save({'model': model.state_dict()},
        #                os.path.join(args.out_dir, f"model_{i}.pt"))
    
    model_path = config.out_dir / f"model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    config = TrainConfig()
    
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
    
    curriculum = Curriculum(args.training.curriculum)
    
    train(
        model,
        optimizer,
        curriculum,
        config,
    )


    wandb.finish()
