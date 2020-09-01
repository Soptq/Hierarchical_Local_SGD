import logging, socket, os

# Log-Related Variables

def init_logger(args, rank, output_dir="./logs"):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    extra = {
        "world_size": args.world_size,
        "lr": args.learning_rate,
    }

    logger = logging.getLogger(socket.gethostname())
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(world_size)s:%(lr)s] [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    log_file = os.path.join(output_dir, '%s-debug_%d-n_%d-bs_%d-lr_%.4f-ep_%d-node%d.log'
                            % (args.model, int(args.debug), args.world_size, args.batch_size,
                               args.learning_rate, args.epoch_size, rank))
    fh = logging.FileHandler(log_file, 'w+')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger = logging.LoggerAdapter(logger, extra)
    return logger