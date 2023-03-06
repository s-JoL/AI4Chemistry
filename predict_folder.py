import sys
import torch
from glob import glob
from PIL import Image
from absl import flags
from x_transformers import Decoder
from transformers import AutoImageProcessor, Swinv2Model, ViTModel
from transformers import AutoTokenizer, BertModel, PreTrainedTokenizerFast

from rdkit import Chem
from rdkit.Chem import Draw

from data.contants import RGroupSymbols
from data.smiles_dataset import graph_to_cxsmiles, cxsmiles_decode
from model.molecule_model import MolecularExpert


FLAGS = flags.FLAGS
flags.DEFINE_string('folder_path', None, '')
flags.mark_flag_as_required('folder_path')
flags.DEFINE_string('ckpt_path', None, '')
flags.mark_flag_as_required('ckpt_path')

flags.DEFINE_string('text', '', '')
flags.DEFINE_bool('graph_mode', False,
                    'use graph mode or smiles mode')
flags.DEFINE_integer('max_seq_len', 512, '')
flags.DEFINE_string('image_model_name', 'microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft',
                    'image model in huggingface')
flags.DEFINE_string('text_model_name', 'bert-base-cased',
                    'text model in huggingface')
flags.DEFINE_string('smiles_tokenizer_path', 'config/smiles_tokenizer.json', '')

FLAGS(sys.argv)

smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file=FLAGS.smiles_tokenizer_path)
smiles_tokenizer.mask_token = '[MASK]'
smiles_tokenizer.pad_token = '[EOS]'
smiles_tokenizer.unk_token = '[UNK]'
smiles_tokenizer.sep_token = '[SEP]'
smiles_tokenizer.cls_token = '[CLS]'
smiles_tokenizer.bos_token = '[BOS]'
smiles_tokenizer.eos_token = '[EOS]'
smiles_tokenizer.add_tokens(list(RGroupSymbols.keys()), special_tokens=True)

image_preprocesser = AutoImageProcessor.from_pretrained(FLAGS.image_model_name)
image_model = Swinv2Model.from_pretrained(FLAGS.image_model_name)

tokenizer = None# AutoTokenizer.from_pretrained(FLAGS.text_model_name)
text_model = None# BertModel.from_pretrained(FLAGS.text_model_name)

gpt = Decoder(
    dim = 768,
    depth = 6,
    heads = 8
)
raw_model = MolecularExpert(FLAGS.graph_mode, image_model, tokenizer, text_model, smiles_tokenizer, gpt, max_seq_len=FLAGS.max_seq_len)
state_dict = torch.load(FLAGS.ckpt_path, map_location='cpu')
raw_model.load_state_dict(state_dict)

for path in glob('{}/*'.format(FLAGS.folder_path)):
    image = Image.open(path)
    if image.mode == 'RGBA':
        image.load() # required for png.split()
        new_img = Image.new("RGB", image.size, (255, 255, 255))
        new_img.paste(image, mask=image.split()[3]) # 3 is the alpha channel
        image = new_img

    text = FLAGS.text
    image = image_preprocesser(image, return_tensors='pt')['pixel_values'].cuda()
    model = raw_model.cuda()
    out = model({'image': image, 'text': [text]}, is_train=False)

    if FLAGS.graph_mode:
        smiles = graph_to_cxsmiles(out['atoms'], out['bonds_graph'], smiles_tokenizer)
    else:
        smiles = out['smiles']
        smiles = ''.join(smiles.split(' '))
        smiles = cxsmiles_decode(smiles, smiles_tokenizer)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Error: ', path)
        continue
    pred_image = Draw.MolToImage(mol, (384, 384))
    pred_image.save(path[: -4] + '_pred.png')
    print(smiles)