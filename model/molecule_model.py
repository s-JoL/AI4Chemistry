import torch
from torch import nn


class MolecularExpert(nn.Module):
    def __init__(self, graph_mode, image_model, 
    text_tokenizer, text_model, 
    pred_tokenizer, pred_model, 
    max_seq_len=128, hidden_size=768):
        super().__init__()
        self.graph_mode = graph_mode
        self.image_model = image_model
        self.image_projector = nn.Linear(1536, hidden_size)

        self.text_tokenizer = text_tokenizer
        self.text_model = text_model
        
        self.pred_tokenizer = pred_tokenizer
        self.pred_vocab_size = len(pred_tokenizer.vocab)
        self.pred_model = pred_model
        
        self.token_embedding = nn.Embedding(self.pred_vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(8, hidden_size)
        self.position_embedding = nn.Embedding(4096, hidden_size)

        self.max_seq_len = max_seq_len
        self.criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        self.hidden_size = hidden_size
        if self.graph_mode:
            self.atom_projector = nn.Linear(hidden_size, self.pred_vocab_size)
            self.bond_projector = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size), 
                nn.ReLU(), 
                nn.Linear(hidden_size, 7) 
            )
        else:
            self.smiles_projector = nn.Linear(hidden_size, self.pred_vocab_size)
        
    def get_embedding(self, inputs):
        # vit获取图片表示
        image = inputs['image']
        image_feat = self.image_model(pixel_values=image).last_hidden_state
        image_feat = self.image_projector(image_feat)
        bs, image_seq_len, image_hidden = image_feat.shape
        # # bert获取文字表示
        # text = inputs['text']
        # text_input = self.text_tokenizer(text, padding=True, return_tensors="pt")
        # text_input = {k: v[:, : 512].to(image.device) for k, v in text_input.items()}
        # text_mask = text_input['attention_mask']
        # text_feat = self.text_model(**text_input).last_hidden_state
        # 屏蔽BERT
        text_mask = torch.zeros((bs, 0), dtype=torch.int64, device=image.device)
        text_feat = torch.zeros((bs, 0, self.hidden_size), dtype=image_feat.dtype, device=image_feat.device)
        _, text_seq_len, text_hidden = text_feat.shape
        # 不同模式输入不同
        if self.graph_mode:
            pred_name = 'atoms'
        else:
            pred_name = 'processed_smiles'
        pred = inputs[pred_name]
        # pred进行分词
        pred_input = self.pred_tokenizer(pred, 
        padding=True, return_token_type_ids=False, return_tensors='pt')
        pred_input = {k: v[:, : self.max_seq_len].type(text_mask.dtype).to(image.device) for k, v in pred_input.items()}
        pred_input_ids = pred_input['input_ids']
        pred_mask = pred_input['attention_mask']
        _, pred_seq_len = pred_input_ids.shape
        # pred获取词向量
        pred_token_embedding = self.token_embedding(pred_input_ids)
        # 获取一下可能用到的特殊字符id和词向量
        sep_ids = torch.ones((bs, 1), dtype=pred_input_ids.dtype, device=image.device) * self.pred_tokenizer.sep_token_id
        sep_embedding = self.token_embedding(sep_ids)
        bos_ids = torch.ones((bs, 1), dtype=pred_input_ids.dtype, device=image.device) * self.pred_tokenizer.bos_token_id
        bos_embedding = self.token_embedding(bos_ids)
        eos_ids = torch.ones((bs, 1), dtype=pred_input_ids.dtype, device=image.device) * self.pred_tokenizer.eos_token_id
        # gpt前面一部分不需要预测，使用mask表示，从bos开始预测，最后一个是eos
        mask_ids = torch.ones((bs, image_seq_len+text_seq_len+1), dtype=pred_input_ids.dtype, device=image.device) * self.pred_tokenizer.mask_token_id
        gt_ids = torch.cat([mask_ids, pred_input_ids, eos_ids], dim=1)
        gt_mask = torch.cat([torch.zeros_like(mask_ids), torch.ones_like(bos_ids), pred_mask], dim=1)
        # concat起来的视觉 文本 pred向量，准备输入gpt
        concat_token_embedding = torch.cat([image_feat, sep_embedding, 
                                            text_feat, bos_embedding, 
                                            pred_token_embedding], dim=1)
        # 计算segment
        segment_id = [0] * image_seq_len + [1] +\
        [2] * text_seq_len + [3] + \
        [4] * pred_seq_len
        segment_ids = torch.tensor([segment_id] * bs, dtype=pred_input_ids.dtype, device=image.device)
        segment_embedding = self.segment_embedding(segment_ids)
        # 计算mask
        image_mask = torch.ones((bs, image_seq_len), dtype=text_mask.dtype,  device=image.device)
        sep_mask = torch.ones((bs, 1), dtype=text_mask.dtype,  device=image.device)
        bos_mask = torch.ones((bs, 1), dtype=text_mask.dtype,  device=image.device)
        concat_mask = torch.cat([image_mask, sep_mask, text_mask, bos_mask, pred_mask], dim=1)
        # 累加计算position ids，避免中间存在的pad造成影响
        position_ids = torch.cumsum(concat_mask, dim=1)
        position_embedding = self.position_embedding(position_ids)
        # 最终的向量
        total_embedding = concat_token_embedding + segment_embedding + position_embedding
        embedding = self.pred_model(total_embedding, mask=concat_mask.bool())

        if self.graph_mode:
            atom_logits = self.atom_projector(embedding)

            context_length = image_seq_len + text_seq_len + 1
            out = {
                'embedding': embedding,
                'atom_logits': atom_logits, 
                'gt_ids': gt_ids, 
                'gt_mask': gt_mask
            }
            for k, v in out.items():
                out[k] = v[:, context_length:]

            bond_embedding = out['embedding']
            seq_len = bond_embedding.shape[1]
            bond_embedding = [bond_embedding.unsqueeze(1).repeat(1, seq_len, 1, 1), bond_embedding.unsqueeze(2).repeat(1, 1, seq_len, 1)]
            bond_embedding = torch.cat(bond_embedding, dim=-1)
            bond_logits = self.bond_projector(bond_embedding)
            # 对称造作，避免不一致
            bond_logits = bond_logits + torch.permute(bond_logits, (0, 2, 1, 3))
            out['bond_logits'] = bond_logits
        else:
            smiles_logits = self.smiles_projector(embedding)
            out = {
                'embedding': embedding,
                'smiles_logits': smiles_logits, 
                'gt_ids': gt_ids, 
                'gt_mask': gt_mask
            }
        return out
    
    def predict(self, inputs):
        assert(len(inputs['text'])==1)
        context_len = None
        normal_end = False
        # 不同模式输入不同
        if self.graph_mode:
            input_name = 'atoms'
            pred_name = 'atom_logits'
        else:
            input_name = 'processed_smiles'
            pred_name = 'smiles_logits'
        # 预测置空
        inputs[input_name] = ['']
        while True:
            if context_len is None:
                embeddings = self.get_embedding(inputs)
                pred_logits = embeddings[pred_name]
                context_len = pred_logits.shape[1] - 1
            else:
                inputs[input_name] = ''.join(pred.split(' '))
                embeddings = self.get_embedding(inputs)
                pred_logits = embeddings[pred_name]
            seq_len = pred_logits.shape[1]
            pred_ids = pred_logits.argmax(dim=-1).cpu()
            pred_ids = pred_ids[0, context_len:]
            pred = self.pred_tokenizer.decode(pred_ids)
            if pred.split(' ')[-1] == '[EOS]' or pred[-2:] == '  ':
                # 如果最后一个是EOS就截掉
                pred = ' '.join(pred.split(' ')[: -1])
                normal_end = True
                break
            if seq_len >= 4096 or seq_len > self.max_seq_len + context_len: 
                break
        if self.graph_mode:
            embedding = embeddings['embedding']
            if normal_end:
                # 1, seqlen, hidden
                bond_embedding = embedding[:, context_len: -1]
            else:
                bond_embedding = embedding[:, context_len:]
            seq_len = bond_embedding.shape[1]
            bond_embedding = [bond_embedding.unsqueeze(1).repeat(1, seq_len, 1, 1), bond_embedding.unsqueeze(2).repeat(1, 1, seq_len, 1)]
            # 1, seqlen, seqlen, 2*hidden
            bond_embedding = torch.cat(bond_embedding, dim=-1)
            bond_logits = self.bond_projector(bond_embedding)
            # 对称造作，避免不一致
            pred_bonds = bond_logits + torch.permute(bond_logits, (0, 2, 1, 3))
            pred_bonds = pred_bonds.argmax(dim=-1)[0].cpu().numpy()
            out = {
                'atoms': pred,
                'bonds_graph': pred_bonds
            }
        else:
            out = {
                'smiles': pred
            }
        return out
        
    def forward(self, inputs, is_train=True):
        if is_train:
            return self.get_embedding(inputs)
        else:
            return self.predict(inputs)

    def get_smiles_loss(self, preds, target):
        # bs, seqlen, num
        smiles_logits = preds['smiles_logits']
        smiles_logits = torch.permute(smiles_logits, (0, 2, 1))
        # bs, seqlen
        smiles_target = preds['gt_ids']
        smiles_mask = preds['gt_mask'].type(smiles_logits.dtype)
        smiles_loss = self.criterion(smiles_logits, smiles_target)
        smiles_loss = smiles_loss * smiles_mask
        smiles_loss = smiles_loss.sum() / (smiles_mask.sum() + 1e-6)
        losses = {
            'total_loss': smiles_loss,
            'smiles_loss': smiles_loss
        }
        return losses

    def get_graph_loss(self, preds, target):
        # bs, seqlen, num
        atom_logits = preds['atom_logits']
        atom_logits = torch.permute(atom_logits, (0, 2, 1))
        # bs, seqlen
        atom_target = preds['gt_ids']
        atom_mask = preds['gt_mask'].type(atom_logits.dtype)
        atom_loss = self.criterion(atom_logits, atom_target)
        atom_loss = atom_loss * atom_mask
        atom_loss = atom_loss.sum() / (atom_mask.sum() + 1e-6)

        # bs, seqlen, seqlen, num
        bond_logits = preds['bond_logits']
        bond_logits = torch.permute(bond_logits, (0, 3, 1, 2))
        
        # bs, seqlen, seqlen
        bond_target = target['bonds_graph']
        # 包含R group和pad mask
        bond_r_group_mask = target['bonds_mask']
        # 只计算原子对的
        # pred_atom = atom_logits.argmax(dim=1)
        # bond_atom_mask = (pred_atom == atom_target).type(atom_mask.dtype)
        # bond_atom_mask = bond_atom_mask.unsqueeze(1) * bond_atom_mask.unsqueeze(2)
        # 只计算上三角
        bs, seq_len = atom_target.shape
        triu_mask = torch.ones((seq_len, seq_len), dtype=bond_r_group_mask.dtype, device=bond_r_group_mask.device)
        triu_mask = torch.triu(triu_mask, diagonal=1)
        bond_triu_mask = triu_mask.unsqueeze(0)

        bond_mask = bond_r_group_mask * bond_triu_mask
        bond_mask = bond_mask.detach()
        bond_loss = self.criterion(bond_logits, bond_target)
        bond_loss = bond_loss * bond_mask
        bond_loss = bond_loss.sum() / (bond_mask.sum() + 1e-6)

        total_loss = atom_loss + bond_loss
        losses = {
            'total_loss': total_loss,
            'atom_loss': atom_loss,
            'bond_loss': bond_loss
        }
        return losses

if __name__ == '__main__':
    from rdkit import Chem
    from torch.utils.data import DataLoader
    from x_transformers import Decoder
    from transformers import AutoImageProcessor, Swinv2Model
    from transformers import AutoTokenizer, BertModel, PreTrainedTokenizerFast
    from torchvision.transforms import Compose

    from data.smiles_dataset import (Image2SmilesDataset, SmilesRemoveNum, 
    SmilesReplaceWithR, TransformV1, smiles_abbrevs, ImageTransform, collate_fn_with_bond)

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
        if len(train_smiles) > 100:
            break
    train_smiles = list(train_smiles)

    smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file="config/smiles_tokenizer.json")
    smiles_tokenizer.mask_token = '[MASK]'
    # use [EOS] to pad，这样eos的位置才正确
    smiles_tokenizer.pad_token = '[EOS]'
    smiles_tokenizer.unk_token = '[UNK]'
    smiles_tokenizer.sep_token = '[SEP]'
    smiles_tokenizer.cls_token = '[CLS]'
    smiles_tokenizer.bos_token = '[BOS]'
    smiles_tokenizer.eos_token = '[EOS]'

    smiles_transform = Compose([SmilesRemoveNum(1), SmilesReplaceWithR(1)])
    image_preprocesser = AutoImageProcessor.from_pretrained("microsoft/swinv2-base-patch4-window16-256")

    transform_smiles = TransformV1(
        smiles_tokenizer=smiles_tokenizer,
        smiles_transform=smiles_transform, 
        mol_transform=smiles_abbrevs, 
        image_transform=ImageTransform(image_preprocesser), 
        return_graph=False)

    transform_graph= TransformV1(
        smiles_tokenizer=smiles_tokenizer,
        smiles_transform=smiles_transform, 
        mol_transform=smiles_abbrevs, 
        image_transform=ImageTransform(image_preprocesser), 
        return_graph=True)

    smiles_train_set = Image2SmilesDataset(train_smiles, transform=transform_smiles)
    smiles_train_loader = DataLoader(smiles_train_set, batch_size=16, shuffle=False)
    smiles_valid_loader = DataLoader(smiles_train_set, batch_size=1, shuffle=False)

    graph_train_set = Image2SmilesDataset(train_smiles, transform=transform_graph)
    graph_train_loader = DataLoader(graph_train_set, batch_size=16, shuffle=False, collate_fn=collate_fn_with_bond)
    graph_valid_loader = DataLoader(graph_train_set, batch_size=1, shuffle=False, collate_fn=collate_fn_with_bond)

    image_model = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window16-256")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    text_model = BertModel.from_pretrained("bert-base-cased")

    gpt = Decoder(
        dim = 768,
        depth = 6,
        heads = 8
    )
    smiles_model = MolecularExpert(False, image_model, tokenizer, text_model, smiles_tokenizer, gpt, max_seq_len=128)
    graph_model = MolecularExpert(True, image_model, tokenizer, text_model, smiles_tokenizer, gpt, max_seq_len=128)

    for data in smiles_train_loader:
        break
    train_out = smiles_model(data, is_train=True)
    for k, v in train_out.items():
        print(k, v.shape)

    losses = smiles_model.get_smiles_loss(train_out, data)
    for k, v in losses.items():
        print(k, v.shape)

    for data in graph_train_loader:
        break
    train_out = graph_model(data, is_train=True)
    for k, v in train_out.items():
        print(k, v.shape)

    losses = graph_model.get_graph_loss(train_out, data)
    for k, v in losses.items():
        print(k, v.shape)

    for data in smiles_valid_loader:
        break
    valid_out = smiles_model(data, is_train=False)
    for k, v in valid_out.items():
        print(k, v)

    for data in graph_valid_loader:
        break
    valid_out = graph_model(data, is_train=False)
    for k, v in valid_out.items():
        print(k, v)
