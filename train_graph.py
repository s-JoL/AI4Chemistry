import os
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

import torch
from torch.utils.data import DataLoader

import wandb
from accelerate import Accelerator
from x_transformers import Decoder
from torchvision.transforms import Compose
from transformers import AutoImageProcessor, Swinv2Model, ViTModel
from transformers import AutoTokenizer, BertModel, PreTrainedTokenizerFast

from utils.hdfs_io import hopen
from data.smiles_dataset import (Image2SmilesDataset, SmilesRemoveNum, 
SmilesReplaceWithR, TransformV1, smiles_abbrevs, ImageTransform, collate_fn_with_bond, graph_to_cxsmiles)
from model.molecule_model import MolecularExpert


accelerator = Accelerator()

if accelerator.is_main_process:
    wandb.init(
        # set the wandb project where this run will be logged
        project='AI for Chemistry-Molecular Image Recognition',
        
        # track hyperparameters and run metadata
        config={
            'image_model': 'google/vit-base-patch16-224-in21k',
            'text_model': 'bert-base-cased', 
            'multimodal_model': 'gpt'
        }
    )

with hopen('hdfs://haruna/byte_search/aweme_relevance/roformer/data/train.txt', 'r') as fp:
    train_reaction_smarts = fp.readlines()
train_smiles = set()
for line in train_reaction_smarts:
    reaction = Chem.AllChem.ReactionFromSmarts(line.strip().decode('utf-8'))
    for mol in reaction.GetReactants():
        smiles = Chem.MolToSmiles(mol)
        train_smiles.add(smiles)
    for mol in reaction.GetProducts():
        smiles = Chem.MolToSmiles(mol)
        train_smiles.add(smiles)
    # if len(train_smiles) > 100:
    #     break
train_smiles = list(train_smiles)

with hopen('hdfs://haruna/byte_search/aweme_relevance/roformer/data/valid.txt', 'r') as fp:
    valid_reaction_smarts = fp.readlines()
valid_smiles = set()
for line in valid_reaction_smarts:
    reaction = Chem.AllChem.ReactionFromSmarts(line.strip().decode('utf-8'))
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

smiles_transform = Compose([SmilesRemoveNum(0.7), SmilesReplaceWithR(0.3, 2)])
image_preprocesser = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

transform_smiles = TransformV1(
    smiles_tokenizer=smiles_tokenizer,
    smiles_transform=smiles_transform, 
    mol_transform=smiles_abbrevs, 
    image_transform=ImageTransform(image_preprocesser), 
    return_graph=True, 
    num_r_groups_in_text=2)

train_set = Image2SmilesDataset(train_smiles, transform=transform_smiles)
valid_set = Image2SmilesDataset(valid_smiles, transform=transform_smiles)

train_loader = DataLoader(train_set, batch_size=48, shuffle=False, drop_last=True, collate_fn=collate_fn_with_bond)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn_with_bond)

image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
text_model = BertModel.from_pretrained("bert-base-cased")

gpt = Decoder(
    dim = 768,
    depth = 6,
    heads = 8
)
raw_model = MolecularExpert(True, image_model, tokenizer, text_model, smiles_tokenizer, gpt, max_seq_len=128)
accelerator.print(raw_model)
total_step = 100000

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
if accelerator.is_main_process:
    wandb.watch(model, log='all', log_freq=200)

for step in range(total_step):
    batch = next(train_loader_iter)
    optim.zero_grad()
    out = model(batch, is_train=True)
    losses = model.get_graph_loss(out, batch)
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
            raw_image = Draw.MolToImage(raw_mol, size=(224, 224))
            raw_image = wandb.Image(raw_image, caption=raw_smiles)
            raw_images.append(raw_image)

            cutted_image = torch.permute(data['image'][0], (1, 2, 0))
            text = data['text'][0]
            cutted_image = wandb.Image((cutted_image * 0.5 + 0.5).cpu().numpy(), caption=text)
            cutted_images.append(cutted_image)

            pred = model(data, is_train=False)
            atoms = pred['atoms']
            bonds_graph = pred['bonds_graph']
            pred_smiles = graph_to_cxsmiles(atoms, bonds_graph, smiles_tokenizer)
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is None:
                pred_mol = Chem.MolFromSmiles('* |$Failed$|')
            pred_image = Draw.MolToImage(pred_mol, size=(224, 224))
            pred_image = wandb.Image(pred_image, caption=pred_smiles)
            pred_images.append(pred_image)
        wandb.log({'raw molecular': raw_images, 'cutted molecular': cutted_images, 'pred molecular': pred_images})
    if step % 1000 == 0 and step > 0 and accelerator.is_main_process:
        if not os.path.isdir('saved_ckpt'):
            os.mkdir('saved_ckpt')
        torch.save(raw_model.state_dict(), 'saved_ckpt/{}.pt'.format(step))
if accelerator.is_main_process:
    wandb.finish()