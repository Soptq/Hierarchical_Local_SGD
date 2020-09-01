import math
import os
import time

import numpy as np

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process

import parser
import dataloader
import hlsgd_logging

args = parser.get_parser().parse_args()

"""
##########################################################################################
#
#   Get Arguments From Parser.
#
##########################################################################################
"""

debug_mode_enabled = args.debug
world_size = args.world_size
batch_size = args.batch_size
lr = args.learning_rate
epoch_size = args.epoch_size
gpu = args.gpu
training_model = args.model

"""
##########################################################################################
#
#   Initialize Useful Variables
#
##########################################################################################
"""

# Configure Processing Unit
if debug_mode_enabled:
    DEVICE = "cpu"
elif isinstance(args.gpu, int):
    DEVICE = "cuda:{}".format(args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    # Will configure it when the worker process is spawned.
    DEVICE = None

# HLSGD algorithm-related variables
G_local_latest = {}  # Lastest local gradients used to update
G_sum = {}  # Local gradients sum
backup_state_dict = {}  # Local snapshot of the global model
init = True     # flag that indicates if it is the first iteration of a epoch
global_rep = {}     # Response array
counter = 0 # Counter of local training

# Log-Related Variables
logger = None


"""
##########################################################################################
#
#   Model Validation
#
##########################################################################################
"""


def validate(val_loader, model, criterion, epoch, num_batches):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, target = data
            inputs = inputs.to(DEVICE)
            target = target.to(DEVICE)
            output = model(inputs)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_loss /= total
    accuracy = 100 * correct / total
    logger.info(
        f'Rank {dist.get_rank()}, epoch {epoch}, val_loss {val_loss / num_batches}, accuracy {accuracy}')
    dist.barrier()


"""
##########################################################################################
#
#   Model Training
#
##########################################################################################
"""


def train(trainloader, model, optimizer, criterion, epoch, num_batches):
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    average_time = 0.0
    dist.barrier()
    start_time = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, target = data
        inputs = inputs.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        wait_time = HLSGD(model, (i == num_batches - 1))   # Model averaging
        optimizer.step()
        epoch_loss += loss.item()
        running_loss += loss.item()
        train_time = time.time() - start_time
        average_time += wait_time
        if i % 10 == 0 and i > 0:
            logger.info(
                f'Rank {dist.get_rank()}, epoch {epoch}: {i}, train_time {train_time}, average_time {average_time}, train_loss {running_loss / 10.0}')
            running_loss = 0.0
    train_time = time.time() - start_time
    logger.info(
        f'Rank {dist.get_rank()}, epoch {epoch}, train_time {train_time}, train_loss {epoch_loss / num_batches}')


"""
##########################################################################################
#
#   HLSGD Algorithm
#
##########################################################################################
"""


def HLSGD(model, force_async=False):
    global G_local_latest, global_rep, init, G_sum, backup_state_dict, counter
    local_train = init or rep_not_ready(global_rep)
    b_rep = dist.barrier(async_op=True)
    sync_start_time = time.time()
    b_rep.wait()
    wait_time = time.time() - sync_start_time

    if local_train and not force_async:
        for name, param in model.named_parameters():
            if init:
                backup_state_dict[name] = param.data.clone()

            if name in G_sum:
                G_sum[name] += param.grad.data
            else:
                G_sum[name] = param.grad.data.clone()
            counter += 1
        if init:
            init = False
    else:
        if force_async:
            wait_time += rep_wait(global_rep)
        for name, param in model.named_parameters():
            grad_temp = G_sum[name] + param.grad.data

            if name in G_local_latest:
                param.data.copy_(backup_state_dict[name].data)
                param.grad.data.copy_(G_local_latest[name].data / float(world_size))
                init = True

            G_local_latest[name] = grad_temp
            G_sum[name].zero_()
            global_rep[name] = dist.all_reduce(G_local_latest[name], op=dist.ReduceOp.SUM, async_op=True)
            counter = 0
            if force_async:
                force_update_start = time.time()
                global_rep[name].wait()
                wait_time += (time.time() - force_update_start)
                param.grad.data.copy_(G_local_latest[name].data / float(world_size))

    return wait_time


def rep_not_ready(rep_dict):
    for _, rep in rep_dict.items():
        if not rep.is_completed():
            return True
    return False


def rep_wait(rep_dict):
    start_time = time.time()
    for _, rep in rep_dict.items():
        rep.wait()
    return time.time() - start_time


"""
##########################################################################################
#
#   Distributed Simulating Code
#
##########################################################################################
"""


def run(rank, size):
    global lr, debug_mode, dbs_enable, acc_threshold, gdataset, best_acc
    logger.info(f'Initiating Rank {rank}, World Size {size}')
    torch.manual_seed(1234)

    if debug_mode_enabled:
        import Net.MnistNet
        model = Net.MnistNet.MnistNet()
    else:
        if args.model == "resnet":
            import Net.Resnet
            model = Net.Resnet.ResNet101()
        elif args.model == "densenet":
            import Net.Densenet
            model = Net.Densenet.DenseNet121()
        elif args.model == "resnext":
            import Net.ResneXt
            model = Net.ResneXt.ResNeXt29_2x64d()
        elif args.model == "vgg":
            import Net.VGG
            model = Net.VGG.VGG('VGG19')
        elif args.model == "mobilenet":
            import Net.MobileNetV2
            model = Net.MobileNetV2.MobileNetV2()
        elif args.model == "effecientnet":
            import Net.EffecientNet
            model = Net.EffecientNet.EfficientNetB0()
        elif args.model == "googlenet":
            import Net.GoogleNet
            model = Net.GoogleNet.GoogLeNet()
        elif args.model == "shufflenet":
            import Net.ShuffleNetV2
            model = Net.ShuffleNetV2.ShuffleNetV2(1)
        elif args.model == "regnet":
            import Net.RegNet
            model = Net.RegNet.RegNetY_400MF()
        elif args.model == "dpn":
            import Net.DPN
            model = Net.DPN.DPN92()
        else:
            raise Exception(
                "resnet, densenet and resnext, vgg, mobilenet, effecientnet, googlenet, shufflenet, regnet, dpn")
    model = model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    partition_size = np.array([1.0 / size for _ in range(size)])    # Equally split the dataset to workers

    # Start training
    logger.info(f'Rank {rank} start training')
    total_train_time = 0  # Count total train time

    for epoch in range(epoch_size):
        train_set, val_set, bsz = dataloader.partition_dataset(partition_size, rank, debug_mode_enabled, batch_size)
        num_batches = math.ceil(len(train_set.dataset) / float(bsz))
        logger.info(
            f"Rank {rank}, number of batches {num_batches}, batch size {train_set.batch_size}, length {train_set.batch_size * num_batches}")
        epoch_start_time = time.time()
        train(train_set, model, optimizer, F.cross_entropy, epoch, num_batches)
        total_train_time += time.time() - epoch_start_time  # Get time that includes communication time.
        validate(val_set, model, F.cross_entropy, epoch, num_batches)

    logger.info(f'Rank {rank} Terminated')
    logger.info(f'Rank {rank} Total Time:')
    logger.info(total_train_time)


def init_processes(rank, size, fn, backend='gloo'):
    global DEVICE, logger
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    # Configuring multiple GPU
    if not debug_mode_enabled and isinstance(gpu, list):
        DEVICE = "cuda:{}".format(gpu[rank])
        torch.cuda.set_device(gpu[rank])

    logger = hlsgd_logging.init_logger(args, rank)
    fn(rank, size)


if __name__ == "__main__":
    processes = []
    for rank in range(world_size):
        p = Process(target=init_processes, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

