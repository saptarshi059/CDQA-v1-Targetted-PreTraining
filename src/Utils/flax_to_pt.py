from transformers import AutoModel
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flax_model_checkpoint_folder', type=str)
    parser.add_argument('--save_encoder_only', type=str2bool)
    args = parser.parse_args()

    print('Loading Flax Checkpoint for Conversion to PyTorch...')
    flax_model = AutoModel.from_pretrained(args.flax_model_checkpoint_folder, from_flax=True)

    if args.save_encoder_only:
        s2s_encoder = flax_model.encoder
        print('Saving only encoder Flax checkpoint to PyTorch')
        s2s_encoder.save_pretrained(args.flax_model_checkpoint_folder)
    else:
        print('Saving the entire Seq-2-Seq Flax Checkpoint to PyTorch...')
        flax_model.save_pretrained(args.flax_model_checkpoint_folder)
