from transformers import AutoModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flax_model_checkpoint_folder', type=str)
    parser.add_argument('--pytorch_model_checkpoint_folder', type=str)
    args = parser.parse_args()

    print('Loading Flax Checkpoint for Conversion to PyTorch...')
    flax_model = AutoModel.from_pretrained(args.flax_model_checkpoint_folder, from_flax=True)

    print('Saving the Flax Checkpoint to PyTorch...')
    flax_model.save_pretrained(args.pytorch_model_checkpoint_folder)

