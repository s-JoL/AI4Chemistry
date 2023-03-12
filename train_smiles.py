import os
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

import torch
from torch.utils.data import DataLoader

import wandb
import numpy as np
from accelerate import Accelerator
from x_transformers import Decoder
from torchvision import transforms
from transformers import AutoImageProcessor, Swinv2Model, ViTModel
from transformers import AutoTokenizer, BertModel, PreTrainedTokenizerFast

from data.contants import RGroupSymbols
from data.smiles_dataset import (Image2SmilesDataset, SmilesRemoveNum, to_canonical,
SmilesReplaceWithR, TransformV1, molecule_abbrevs, ImageTransform, RandomTransform, RandomNoise, cxsmiles_decode)
from model.molecule_model import MolecularExpert


image_size = (384, 384)
image_model_name = 'microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft'
text_model_name = 'bert-base-cased'

accelerator = Accelerator()
if accelerator.is_main_process:
    wandb.init(
        # set the wandb project where this run will be logged
        project='AI for Chemistry-Molecular Image Recognition',
        
        # track hyperparameters and run metadata
        config={
            'image_model': image_model_name,
            'text_model': text_model_name, 
            'multimodal_model': 'gpt'
        }
    )

with open('uspto_data/train.txt', 'r') as fp:
    train_reaction_smarts = fp.readlines()
train_smiles = set()
for line in train_reaction_smarts:
    reaction = Chem.AllChem.ReactionFromSmarts(line.strip())
    for mol in reaction.GetReactants():
        smiles = Chem.MolToSmiles(mol)
        train_smiles.add(smiles)
    for mol in reaction.GetProducts():
        smiles = Chem.MolToSmiles(mol)
        train_smiles.add(smiles)
    # if len(train_smiles) > 100:
    #     break
train_smiles = list(train_smiles)

with open('uspto_data/valid.txt', 'r') as fp:
    valid_reaction_smarts = fp.readlines()
valid_smiles = set()
for line in valid_reaction_smarts:
    reaction = Chem.AllChem.ReactionFromSmarts(line.strip())
    for mol in reaction.GetReactants():
        smiles = Chem.MolToSmiles(mol)
        valid_smiles.add(smiles)
    for mol in reaction.GetProducts():
        smiles = Chem.MolToSmiles(mol)
        valid_smiles.add(smiles)
    # if len(valid_smiles) > 100:
    #     break
valid_smiles = list(valid_smiles)

smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file="config/smiles_tokenizer.json")
smiles_tokenizer.mask_token = '[MASK]'
smiles_tokenizer.pad_token = '[EOS]'
smiles_tokenizer.unk_token = '[UNK]'
smiles_tokenizer.sep_token = '[SEP]'
smiles_tokenizer.cls_token = '[CLS]'
smiles_tokenizer.bos_token = '[BOS]'
smiles_tokenizer.eos_token = '[EOS]'
# 加入固定的R基团名字，避免被分词切开
for symbol in RGroupSymbols.keys():
    if symbol not in smiles_tokenizer.vocab:
        smiles_tokenizer.add_tokens(symbol, special_tokens=True)

# 加入数据增强
smiles_transform = transforms.Compose([SmilesRemoveNum(0.9), SmilesReplaceWithR(0.4, 5)])
image_preprocesser = AutoImageProcessor.from_pretrained(image_model_name)
image_transform = ImageTransform(
    transforms.Compose([
        RandomTransform(transforms.GaussianBlur((5, 5), 1), 0.2),
        transforms.RandomPerspective(0.3, 0.1),
        transforms.RandomAdjustSharpness(0, 0.3),
        RandomNoise(0.3)
        ]), 
        image_preprocesser)

train_transform_smiles = TransformV1(
    mol_size=image_size,
    smiles_tokenizer=smiles_tokenizer,
    smiles_transform=smiles_transform, 
    mol_transform=molecule_abbrevs, 
    image_transform=image_transform, 
    return_graph=False, 
    num_r_groups_in_text=0)

valid_transform_smiles = TransformV1(
    mol_size=image_size,
    smiles_tokenizer=smiles_tokenizer,
    smiles_transform=smiles_transform, 
    mol_transform=molecule_abbrevs, 
    image_transform=ImageTransform(None, image_preprocesser), 
    return_graph=False, 
    num_r_groups_in_text=0)

train_set = Image2SmilesDataset(train_smiles, transform=train_transform_smiles)
valid_set = Image2SmilesDataset(valid_smiles, transform=valid_transform_smiles)

train_loader = DataLoader(train_set, batch_size=32, shuffle=False, drop_last=True, num_workers=8)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, drop_last=True)

image_model = Swinv2Model.from_pretrained(image_model_name)

tokenizer = None# AutoTokenizer.from_pretrained(text_model_name)
text_model = None# BertModel.from_pretrained(text_model_name)

gpt = Decoder(
    dim = 768,
    depth = 6,
    heads = 8
)
raw_model = MolecularExpert(False, image_model, tokenizer, text_model, smiles_tokenizer, gpt, max_seq_len=512)
accelerator.print(raw_model)
total_step = 100001

def linear_lr_with_warmup(total_step, warmup_rate=0.05):
    def lr_scheduler(step):
        step = step / accelerator.state.num_processes
        wanmup_step = int(total_step*warmup_rate)
        if step < wanmup_step:
            return step / wanmup_step + 1e-6
        else:
            return 1 - (step - wanmup_step) / (total_step - warmup_rate)
    return lr_scheduler
optim = torch.optim.Adam(raw_model.parameters(), lr=4e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, linear_lr_with_warmup(total_step))

train_loader, valid_loader, model, optim, scheduler = accelerator.prepare(
    train_loader, valid_loader, raw_model, optim, scheduler
)
train_loader_iter = iter(train_loader)
valid_loader_iter = iter(valid_loader)

for step in range(total_step):
    batch = next(train_loader_iter)
    optim.zero_grad()
    out = model(batch, is_train=True)
    losses = model.get_smiles_loss(out, batch)
    accelerator.backward(losses['total_loss'])
    optim.step()
    scheduler.step()

    if step % 100 == 0 and accelerator.is_main_process:
        for k, v in losses.items():
            wandb.log({'Losses/{}'.format(k): v})
        current_lr = optim.param_groups[0]['lr']
        wandb.log({'Training/lr': current_lr})
        wandb.log({'Training/Global Step': step})
        accelerator.print('Step: {}, Loss: {}'.format(step, losses['total_loss']))
    if step % 500 == 0 and step > 0 and accelerator.is_main_process:
        raw_images = []
        cutted_images = []
        pred_images = []
        for val_step in range(5):
            data = next(valid_loader_iter)
            pred = model(data, is_train=False)
            
            raw_smiles = data['smiles'][0]
            raw_mol = Chem.MolFromSmiles(raw_smiles)
            if raw_mol is None:
                raw_mol = Chem.MolFromSmiles('* |$Failed$|')
            raw_image = Draw.MolToImage(raw_mol, size=image_size)
            raw_image = wandb.Image(raw_image, caption=raw_smiles)
            raw_images.append(raw_image)

            cutted_image = torch.permute(data['image'][0], (1, 2, 0))
            cutted_image = cutted_image.cpu().numpy()
            text = data['text'][0]
            image_mean = np.asarray([[image_preprocesser.image_mean]])
            image_std = np.asarray([[image_preprocesser.image_std]])
            cutted_image = wandb.Image((cutted_image * image_std + image_mean), caption=text)
            cutted_images.append(cutted_image)

            pred_smiles = pred['smiles']
            pred_smiles = ''.join(pred_smiles.split(' '))
            pred_smiles = cxsmiles_decode(pred_smiles, smiles_tokenizer)
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                pred_mol = Chem.MolFromSmiles('* |$Failed$|')
            pred_image = Draw.MolToImage(pred_mol, size=image_size)
            pred_image = wandb.Image(pred_image, caption=pred_smiles)
            pred_images.append(pred_image)
        wandb.log({'raw molecular': raw_images, 'cutted molecular': cutted_images, 'pred molecular': pred_images})
    if step % 2000 == 0 and step > 0 and accelerator.is_main_process:
        if not os.path.isdir('saved_ckpt'):
            os.mkdir('saved_ckpt')
        torch.save(raw_model.state_dict(), 'saved_ckpt/{}.pt'.format(step))
    if step % 10000 == 0 and step > 0 and accelerator.is_main_process:
        canonical_cxsmiles = []
        canonical_decoded_cxsmiles = []
        correct_num = 0
        for val_step in range(200):
            data = next(valid_loader_iter)
            pred = model(data, is_train=False)
            raw_smiles = data['smiles'][0]
            raw_smiles = to_canonical(raw_smiles)

            pred_smiles = pred['smiles']
            pred_smiles = ''.join(pred_smiles.split(' '))
            pred_smiles = cxsmiles_decode(pred_smiles, smiles_tokenizer)
            pred_smiles = to_canonical(pred_smiles)

            canonical_cxsmiles.append(raw_smiles)
            canonical_decoded_cxsmiles.append(pred_smiles)
            if raw_smiles == pred_smiles:
                correct_num += 1
        wandb.log({'Training/Accuracy': correct_num/200})
if accelerator.is_main_process:
    wandb.finish()