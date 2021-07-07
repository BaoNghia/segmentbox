import os, time
import logging
import argparse
from datetime import datetime

import torch
from data_loader.dataloader import get_data_loader
from utils import general, metrics_loader, callbacks
from utils.general import (yaml_loader, model_loader,  get_optimizer,
    get_lr_scheduler, get_loss_fn)

import trainer
import segmentation_models_pytorch as smp


# from torchsampler import ImbalancedDatasetSampler
def main(config, model, log_dir, checkpoint=None,):     
    if checkpoint is not None:
        print("...Load checkpoint from {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("...Checkpoint loaded")

    # Checking cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {} ".format(device))

    # Convert to suitable device
    # logger.info(cls_model)
    model = model.to(device)
    logger.info("Number parameters of model: {:,}".format(sum(p.numel() for p in model.parameters())))

    # Using parsed configurations to create a dataset
    # Create dataset
    num_of_class = len(config["data"]["label_dict"])
    train_loader, valid_loader, test_loader = get_data_loader(config)
    print("Dataset and Dataloaders created")

    # create a metric for evaluating
    metric_names = config["train"]["metrics"]
    train_metrics = metrics_loader.Metrics(metric_names)
    val_metrics = metrics_loader.Metrics(metric_names)
    print("Metrics implemented successfully")

    ## read settings from json file
    ## initlize optimizer from config
    optimizer_module, optimizer_params = get_optimizer(config)
    optimizer = optimizer_module(model.parameters(), **optimizer_params)
    ## initlize sheduler from config
    scheduler_module, scheduler_params = get_lr_scheduler(config)
    scheduler = scheduler_module(optimizer, **scheduler_params)
    # scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)

    ## Initlize loss function
    loss_fn = get_loss_fn(config)
    criterion = loss_fn()

    print("\nTraing shape: {} samples".format(len(train_loader.dataset)))
    print("Validation shape: {} samples".format(len(valid_loader.dataset)))
    print("Beginning training...")

    # initialize the early_stopping callback
    save_mode = config["train"]["mode"]
    early_patience = config["train"]["patience"]
    checkpoint_path = os.path.join(log_dir, "Checkpoint.ckpt")
    early_stopping = callbacks.EarlyStopping(patience=early_patience, mode = save_mode, path = checkpoint_path)

    # training models
    logger.info("-"*100)
    num_epochs = int(config["train"]["num_epochs"])
    t0 = time.time()
    for epoch in range(num_epochs):
        t1 = time.time()
        print(('\n' + '%13s' * 3) % ('Epoch', 'gpu_mem', 'mean_loss'))
        train_loss, val_loss, train_result, val_result = trainer.train_one_epoch(
            epoch, num_epochs,
            model, device,
            train_loader, valid_loader,
            criterion, optimizer,
            train_metrics, val_metrics, 
        )

        train_checkpoint = {
            'epoch': epoch,
            'valid_loss': val_loss,
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        ## lr decay
        scheduler.step(val_loss)

        ## log to file 
        logger.info("\n------Epoch {} / {}, Training time: {:.4f} seconds------".format(epoch, num_epochs, (time.time() - t1)))
        logger.info(f"Training loss: {train_loss} \n Training metrics: {train_result}")
        logger.info(f"Validation loss: {val_loss} \n Validation metrics: {val_result}")
        
        ## tensorboard writer
        tb_writer.add_scalar("Training Loss", train_loss, epoch)
        tb_writer.add_scalar("Valid Loss", val_loss, epoch)
        for metric_name in metric_names:
            tb_writer.add_scalar(f"Training {metric_name}", train_result[metric_name], epoch)
            tb_writer.add_scalar(f"Validation {metric_name}", val_result[metric_name], epoch)

        # Save model
        early_stopping(val_loss, train_checkpoint)
        if early_stopping.early_stop:
            logging.info("Early Stopping!!!")
            break

    # testing on test set
    # load the test model and making inference
    print("\n==============Inference on the testing set==============")
    best_checkpoint = torch.load(checkpoint_path)
    test_model = best_checkpoint['model']
    test_model.load_state_dict(best_checkpoint['state_dict'])
    test_model = test_model.to(device)
    test_model.eval()

    # # logging report
    # report = tester.test_result(test_model, test_loader, device, cfg)
    # logging.info("\nClassification Report: \n {}".format(report))
    # logging.info('Completed in %.3f seconds.' % (time.time() - t0))

    # print("Classification Report: \n{}".format(report))
    # print('Completed in %.3f seconds.' % (time.time() - t0))
    # print('Start Tensorboard with tensorboard --logdir {}, view at http://localhost:6006/'.format(log_dir))
    # # # saving torch models

    print(f"-------- Checkpoints and logs are saved in ``{log_dir}`` --------")
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NA')
    parser.add_argument('-cfg', '--configure', default='cfgs/tenes.yaml', help='YAML file')
    parser.add_argument('-cp', '--checkpoint', default=None, help = 'checkpoint path')
    args = parser.parse_args()
    checkpoint = args.checkpoint

# read configure file
    config = yaml_loader(args.configure) # cfg dict
    ## comment for this experiment: leave here
    comment = config["session"]["_comment_"]

    ## create dir to save log and checkpoint
    save_path = config['session']['save_path']
    time_str = str(datetime.now().strftime("%Y%m%d-%Hh%M"))
    project_name = config["session"]["project_name"]
    log_dir = os.path.join(save_path, project_name, time_str)

    ## create logger
    tb_writer = general.make_writer(log_dir = log_dir)
    logger = general.log_initilize(log_dir)
    logger.info(f"Start Tensorboard with tensorboard --logdir {log_dir}, view at http://localhost:6006/")
    logger.info(f"Project name: {project_name}")
    logger.info(f"CONFIGS: \n {config}")

    ## Create model
    model = model_loader(config)
    print(f"Create model Successfully")
    print(("Number parameters of model: {:,}".format(sum(p.numel() for p in model.parameters()))))
    time.sleep(1.8)

    best_ckpt = main(
        config = config,
        model = model,
        log_dir = log_dir,
        checkpoint=checkpoint,
    )