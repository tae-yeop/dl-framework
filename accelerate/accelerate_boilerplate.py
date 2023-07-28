from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


logger = get_logger(__name__)

args.mixed_precision = ["no", "fp16", "bf16"]


def main():
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Acceleartor(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=["tensorboard","wandb"],
                              logging_dir=logging_dir)

    if args.seed is not None:
        set_seed(args.seed)

    #######################################################
    #####################  Model ###########################
    if args.init_empty_weights:
        from accelerate import init_empty_weights
        model = Model(...)


        
    # Model
    model = Model(...)
    model2 = Model2(...) # Inference only

    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    model2.to(accelerator.deivce, dtype=weight_dtype)
    # Optimizer
    # Dataset, DataLoader
    # LR scheduler
    
    

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    device = accelerator.device
    # 이게 필요? prepare()가 모두 담당하는거 아닌가?
    # https://huggingface.co/docs/accelerate/v0.13.2/en/usage_guides/tracking#integrated-trackers
    # https://huggingface.co/docs/accelerate/v0.13.2/en/basic_tutorials/migration
    # 여기에는 prepare()이후에 to()를 하지 않음
    model.to(device)

    
    ########## train preparation #########
    num_update_steps_per_epoch =
    
    ######################################

    ########## log tracker ###############
    if accelerator.is_main_process:
        accelerator.init_trackers("experiment-1", config=vars(args))
        # or
        hps = {"num_iterations": 5, "learning_rate": 1e-2}
        accelerator.init_trackers("my_project", config=hps)


    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


    ####### training pregress bar ###########
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    ########## train loop ####################
    ##### epoch 기준
    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            x, y = batch
            pred = model(x)
            loss = loss_func(y, pred)
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        #### global step 기준 logging ####
        if not global_step % args.log_interval:
            logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
            # set_postfix는 뭐하는건지?
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.update(1)
        global_step += 1

        if global_step >= args.max_train_steps:
            break

@hydra.main(config=)
def main():
    # parser
    

if __name__ == '__main__':
    main()