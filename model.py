from RGAN import RGAN
from TimeGAN import TimeGAN

def select_model(args, data):
    '''
    Initializes model
    '''
    print("Initializing model..")

    if args.model == 'RGAN':
        return RGAN(data, 
                    device=args.device, 
                    input_dim=args.input_dim, 
                    hidden_dim=args.hidden_dim, 
                    seed=args.seed, 
                    batch_size=args.batch_size)
    elif args.model == 'TimeGAN':
        return TimeGAN(data, 
                       device=args.device, 
                       input_dim=args.input_dim, 
                       hidden_dim=args.hidden_dim,
                       latent_dim=args.latent_dim,
                       seed=args.seed,
                       batch_size=args.batch_size)

    raise Exception("An implementation for the given model is missing")