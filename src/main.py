import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
OUTPUT_DIR = os.path.join(parent_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from metric import evaluate_model
from torch.utils.data import DataLoader
from co_teaching_net import CoTeachingNet
from prepare_dataset import StandardParams, PrepareTrainDataset, PrepareTestDataset

def main(args):
    train_preparator = PrepareTrainDataset(train_clean_csv_path=args.train_clean_csv_path,
                                           train_noise_csv_path=args.train_csv_noise_path,
                                           augment=True,
                                           is_change_path=args.is_change_path,
                                           image_dir=args.image_dir,
                                           output_dir=args.output_dir,
                                           save_cls_distribution=args.save_cls_distribution,
                                           fix_paths_in_memory=False)
    
    test_preparator = PrepareTestDataset(csv_path=args.test_csv_path,
                                        is_change_path=args.is_change_path,
                                        image_dir=args.image_dir,
                                        output_dir=args.output_dir,
                                        save_cls_distribution=args.save_cls_distribution,
                                        fix_paths_in_memory=False)
    
    train_dataset = train_preparator.trainer()
    #test_dataset = test_preparator.tester()

    train_loader = DataLoader(train_dataset, batch_size=StandardParams.BATCH_SIZE.value, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=StandardParams.BATCH_SIZE.value, shuffle=False)


    network_A = CoTeachingNet(path_to_save_model_weight=args.output_dir, 
                            model_name=args.model_name, 
                            class_number=train_dataset.nb_class,
                            how_many_layers_to_unfreeze=4)
    
    network_B = CoTeachingNet(path_to_save_model_weight=args.output_dir, 
                            model_name=args.model_name, 
                            class_number=train_dataset.nb_class,
                            how_many_layers_to_unfreeze=4)

    print('Network A and B are ready for use.')

    def per_sample_loss(logits, labels, pos_weight=None):
        """ Computes the binary cross-entropy loss for each sample without reduction.
        
        Args:
            logits (torch.Tensor): Logits from the model of shape (B, C).
            labels (torch.Tensor): Ground truth labels of shape (B, C).
            pos_weight (torch.Tensor, optional): A weight of positive examples. Defaults to None.
        
        Returns:
            torch.Tensor: Loss values for each sample of shape (B, C).
        """
        criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        loss = criterion(logits, labels)
        return loss
    
    def flip_hard_labels(labels, loss, R_T):
        """ This function flips the labels of the samples with the largest losses.
        # Small loss trick is used to select clean samples, so we flip the labels of the samples with the largest losses.
        Args:
            labels (torch.Tensor): Original labels of shape (B, C).
            loss (torch.Tensor): Loss values of shape (B, C).
            R_T (float): Fraction of samples to flip.

        Returns:
            torch.Tensor: Labels with flipped values.
        """
        B, C = loss.shape
        k = max(1, int(B * R_T))
        out = labels.clone()
        for c in range(C):
            idx = torch.topk(loss[:, c], k=k, largest=True).indices
            out[idx, c] = 1.0 - out[idx, c]
        return out
    
    def get_networks_loss_vec(logits_A, logits_B, labels):
        """ Computes the loss vectors for both networks.
        
        Args:
            logits_A (torch.Tensor): Logits from network A of shape (B, C).
            logits_B (torch.Tensor): Logits from network B of shape (B, C).
            labels (torch.Tensor): Ground truth labels of shape (B, C).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss vectors for network A and B, each of shape (B, C)."""
        loss_vec_A = per_sample_loss(logits_A, labels)
        loss_vec_B = per_sample_loss(logits_B, labels)
        return loss_vec_A, loss_vec_B

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    network_A.to(device)
    network_B.to(device)

    optimizer_A = torch.optim.Adam(network_A.parameters(), lr=args.learning_rate)
    optimizer_B = torch.optim.Adam(network_B.parameters(), lr=args.learning_rate)
    
    Tk = 40
    EPOCH_SWITCHER = 1
    print('🚔 Starting training...')

    for epoch in range(args.epochs):
        network_A.train(), network_B.train()
        R_T = min((epoch + 1) * args.noise_rate / Tk, args.noise_rate)
        
        epoch_loss_A, epoch_loss_B = 0.0, 0.0
        n_batches = 0

        for i, (images, labels_noise, labels_clean) in enumerate(train_loader):
            noisy_img = images.to(device)
            noisy_lbl = labels_noise.to(device)
            clean_lbl = labels_clean.to(device)

            _, logits_A = network_A(noisy_img)
            _, logits_B = network_B(noisy_img)

            if epoch < EPOCH_SWITCHER:
                loss_vec_A, loss_vec_B = get_networks_loss_vec(logits_A, logits_B, noisy_lbl)
                loss_A, loss_B = loss_vec_A.mean(), loss_vec_B.mean()

                optimizer_A.zero_grad()
                loss_A.backward()
                optimizer_A.step()

                optimizer_B.zero_grad()
                loss_B.backward()
                optimizer_B.step()

            else:
                #Co-Teaching step
                loss_vec_A, loss_vec_B = get_networks_loss_vec(logits_A, logits_B, noisy_lbl)

                flipped_labels_A = flip_hard_labels(noisy_lbl, loss_vec_A, R_T)
                flipped_labels_B = flip_hard_labels(noisy_lbl, loss_vec_B, R_T)

                loss_vec_A = per_sample_loss(logits_A, flipped_labels_B)
                loss_vec_B = per_sample_loss(logits_B, flipped_labels_A)

                loss_A, loss_B = loss_vec_A.mean(), loss_vec_B.mean()

                optimizer_A.zero_grad(), loss_A.backward(), optimizer_A.step()
                optimizer_B.zero_grad(), loss_B.backward(), optimizer_B.step()

            epoch_loss_A += loss_A.item()
            epoch_loss_B += loss_B.item()
            n_batches += 1

        avg_loss_A = epoch_loss_A / n_batches
        avg_loss_B = epoch_loss_B / n_batches
        print(f"Epoch [{epoch+1}/{args.epochs}] | Current R_T: {R_T:.4f} | Mini Batch {i+1}/{len(train_loader)} | Loss A: {avg_loss_A:.4f} | Loss B: {avg_loss_B:.4f}")
        
        # Evaluate the model on the test set
        test_dataset = test_preparator.tester()
        test_loader = DataLoader(test_dataset, batch_size=StandardParams.BATCH_SIZE.value, shuffle=False)
        metrics = evaluate_model(test_loader, network_A, network_B, threshold=0.5, device=device)
        dico = {**metrics, "epoch": epoch + 1}
        with open(os.path.join(args.output_dir, 'evaluation_log.txt'), 'a') as f:
            f.write(str(dico) + '\n')
        print(f"Evaluation Metrics after Epoch {epoch+1}: F1 Micro: {metrics['f1_micro']:.4f}, F1 Macro: {metrics['f1_macro']:.4f}, Subset Acc: {metrics['subset_acc']:.4f}, Sample Acc: {metrics['sample_acc']:.4f}")
    print('✅ Training completed.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noisy Labels Training")
    parser.add_argument('--train_clean_csv_path', default='D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/dataset/arvore_dataset_train_clean.csv', type=str, required=False, help='Path to the training CSV file with clean labels')
    parser.add_argument('--train_csv_noise_path', default="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/dataset/arvore_dataset_train_noise_25.csv", type=str, required=False, help='Path to the training CSV file with noisy labels')
    parser.add_argument('--test_csv_path', default="D:/meus_codigos_doutourado/Doutourado_2025_2/clean_code/Learning-By-Small-Loss-Approach-Co-teaching/dataset/arvore_dataset_test_clean.csv", type=str, required=False, help='Path to the test CSV file')
    parser.add_argument('--image_dir', default="D:/meus_codigos_doutourado/Doutourado_2025_2/s1/s1/200m", type=str, required=False, help='Path to the directory containing images')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Path to the output directory')
    #parser.add_argument('--img_size', type=int, nargs=2, default=(224, 224), help='Image size (height, width)')
    parser.add_argument('--is_change_path', action='store_true', help='Whether to change image paths in the CSV files. Useful when you have to match the current directory of the images with the annotations')
    parser.add_argument('--save_cls_distribution', default=False, action='store_true', help='Save class distribution histograms for datasets')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='Model architecture to use')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.00025, help='Learning rate for the optimizer')
    parser.add_argument('--noise_rate', type=float, default=0.2, help='Estimated noise rate in the training data')
    args = parser.parse_args()
    main(args)